import datetime
import inspect
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import time
import torch
from ..helpers import max_diff, string_type, string_diff
from ..helpers.helper import flatten_object
from ..helpers.ort_session import make_feeds
from ..helpers.torch_test_helper import to_any, torch_deepcopy
from ..torch_export_patches import bypass_export_some_errors
from .hghub import get_untrained_model_with_inputs
from .hghub.model_inputs import random_input_kwargs


def empty(value: Any) -> bool:
    """Tells if the value is empty."""
    if isinstance(value, (str, list, dict, tuple, set)):
        return bool(value)
    if value is None:
        return True
    return False


def _ds_clean(v):
    return (
        str(v)
        .replace(",min=None", "")
        .replace(",max=None", "")
        .replace(",_factory=True", "")
        .replace("<class 'onnx_diagnostic.torch_models.hghub.model_inputs.", "")
        .replace("'>", "")
        .replace("_DimHint(type=<_DimHintType.DYNAMIC: 3>)", "DYNAMIC")
        .replace("_DimHint(type=<_DimHintType.AUTO: 3>)", "AUTO")
    )


def get_inputs_for_task(task: str, config: Optional[Any] = None) -> Dict[str, Any]:
    """
    Returns dummy inputs for a specific task.

    :param task: requested task
    :param config: returns dummy inputs for a specific config if available
    :return: dummy inputs and dynamic shapes
    """
    kwargs, f = random_input_kwargs(config, task)
    return f(model=None, config=config, **kwargs)


def split_args_kwargs(inputs: Any) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    """Splits into args, kwargs."""
    if isinstance(inputs, dict):
        return (), inputs
    if isinstance(inputs, tuple) and len(inputs) == 2 and isinstance(inputs[1], dict):
        return inputs
    assert isinstance(inputs, tuple), f"Unexpected inputs {string_type(inputs)}"
    return inputs, {}


def make_inputs(
    args: Optional[Tuple[Any, ...]], kwargs: Optional[Dict[str, Any]] = None
) -> Any:
    """Returns either args, kwargs or both depending on which ones are empty."""
    assert args or kwargs, "No input was given."
    if not args:
        return kwargs
    if not kwargs:
        return args
    return args, kwargs


def filter_inputs(
    inputs: Any,
    drop_names: List[str],
    model: Optional[Union[torch.nn.Module, List[str]]] = None,
    dynamic_shapes: Optional[Any] = None,
):
    """
    Drops some inputs from the given inputs.
    It updates the dynamic shapes as well.
    """
    args, kwargs = split_args_kwargs(inputs)
    set_drop_names = set(drop_names)
    kwargs = {k: v for k, v in kwargs.items() if k not in set_drop_names}
    dyn = (
        {k: v for k, v in dynamic_shapes.items() if k not in set_drop_names}
        if dynamic_shapes and isinstance(dynamic_shapes, dict)
        else dynamic_shapes
    )
    if not args or all(i in kwargs for i in set_drop_names):
        return make_inputs(args, kwargs), dyn
    assert model, (
        f"we need the model to get the parameter name but model is None, "
        f"input_names={drop_names} and args={string_type(args)}"
    )
    pnames = (
        list(inspect.signature(model.forward).parameters)
        if isinstance(model, torch.nn.Module)
        else model
    )
    new_args = []
    new_ds = []
    for i, a in enumerate(args):
        if isinstance(dynamic_shapes, tuple):
            new_ds.append(None if pnames[i] in set_drop_names else dynamic_shapes[i])
        new_args.append(None if pnames[i] in set_drop_names else a)
    new_inputs = make_inputs(tuple(new_args), kwargs)
    if new_ds:
        return new_inputs, tuple(new_ds)
    return new_inputs, dyn


def _make_folder_name(
    model_id: str,
    exporter: Optional[str],
    optimization: Optional[str] = None,
    dtype: Optional[Union[str, torch.dtype]] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> str:
    "Creates a filename unique based on the given options."
    els = [model_id.replace("/", "_")]
    if exporter:
        els.append(exporter)
    if optimization:
        els.append(optimization)
    if dtype is not None and dtype:
        stype = dtype if isinstance(dtype, str) else str(dtype)
        stype = stype.replace("float", "f").replace("uint", "u").replace("int", "i")
        els.append(stype)
    if device is not None and device:
        sdev = device if isinstance(device, str) else str(device)
        sdev = sdev.lower()
        if "cpu" in sdev:
            sdev = "cpu"
        elif "cuda" in sdev:
            sdev = "cuda"
        else:
            raise AssertionError(f"unexpected value for device={device}, sdev={sdev!r}")
        els.append(sdev)
    return "-".join(els)


def version_summary() -> Dict[str, Union[int, float, str]]:
    """
    Example:

    .. runpython::
        :showcode:

        import pprint
        from onnx_diagnostic.torch_models.test_helper import version_summary

        pprint.pprint(version_summary())
    """
    import numpy

    summary: Dict[str, Union[int, float, str]] = {
        "version_torch": torch.__version__,
        "version_numpy": numpy.__version__,
    }
    try:
        import transformers

        summary["version_transformers"] = transformers.__version__
    except ImportError:
        pass
    try:
        import onnx

        summary["version_onnx"] = onnx.__version__
    except ImportError:
        pass
    try:
        import onnxscript

        summary["version_onnxscript"] = onnxscript.__version__
    except ImportError:
        pass
    try:
        import onnxruntime

        summary["version_onnxruntime"] = onnxruntime.__version__
    except ImportError:
        pass
    import onnx_diagnostic

    summary["version_onnx_diagnostic"] = onnx_diagnostic.__version__
    summary["version_date"] = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    return summary


def validate_model(
    model_id: str,
    task: Optional[str] = None,
    do_run: bool = False,
    exporter: Optional[str] = None,
    do_same: bool = False,
    verbose: int = 0,
    dtype: Optional[Union[str, torch.dtype]] = None,
    device: Optional[Union[str, torch.device]] = None,
    trained: bool = False,
    optimization: Optional[str] = None,
    quiet: bool = False,
    patch: bool = False,
    stop_if_static: int = 1,
    dump_folder: Optional[str] = None,
    drop_inputs: Optional[List[str]] = None,
) -> Tuple[Dict[str, Union[int, float, str]], Dict[str, Any]]:
    """
    Validates a model.

    :param model_id: model id to validate
    :param task: task used to generate the necessary inputs,
        can be left empty to use the default task for this model
        if it can be determined
    :param do_run: checks the model works with the defined inputs
    :param exporter: exporter the model using this exporter,
        available list: ``export-strict``, ``export-nostrict``, ``onnx``
    :param do_same: checks the discrepancies of the exported model
    :param verbose: verbosity level
    :param dtype: uses this dtype to check the model
    :param device: do the verification on this device
    :param trained: use the trained model, not the untrained one
    :param optimization: optimization to apply to the exported model,
        depend on the the exporter
    :param quiet: if quiet, catches exception if any issue
    :param patch: applies patches (``patch_transformers=True``) before exporting,
        see :func:`onnx_diagnostic.torch_export_patches.bypass_export_some_errors`
    :param stop_if_static: stops if a dynamic dimension becomes static,
        see :func:`onnx_diagnostic.torch_export_patches.bypass_export_some_errors`
    :param dump_folder: dumps everything in a subfolder of this one
    :param drop_inputs: drops this list of inputs (given their names)
    :return: two dictionaries, one with some metrics,
        another one with whatever the function produces
    """
    assert not trained, f"trained={trained} not supported yet"
    summary = version_summary()
    if dump_folder:
        folder_name = _make_folder_name(
            model_id, exporter, optimization, dtype=dtype, device=device
        )
        dump_folder = os.path.join(dump_folder, folder_name)
        if not os.path.exists(dump_folder):
            os.makedirs(dump_folder)
        summary["dump_folder"] = dump_folder
        summary["dump_folder_name"] = folder_name
        if verbose:
            print(f"[validate_model] dump into {folder_name!r}")
    else:
        folder_name = None
    if verbose:
        print(f"[validate_model] validate model id {model_id!r}")
        print("[validate_model] get dummy inputs...")
        summary["model_id"] = model_id
    begin = time.perf_counter()
    if quiet:
        try:
            data = get_untrained_model_with_inputs(model_id, verbose=verbose, task=task)
        except Exception as e:
            summary["ERR_create"] = str(e)
            data["ERR_create"] = e
            summary["time_create"] = time.perf_counter() - begin
            return summary, {}
    else:
        data = get_untrained_model_with_inputs(model_id, verbose=verbose, task=task)

    if drop_inputs:
        if verbose:
            print(f"[validate_model] drop inputs {drop_inputs!r}")
            print(f"[validate_model] current inputs: {string_type(data['inputs'])}")
            print(
                f"[validate_model] current dynnamic_shapes: "
                f"{_ds_clean(data['dynamic_shapes'])}"
            )
        data["inputs"], data["dynamic_shapes"] = filter_inputs(
            data["inputs"],
            drop_names=drop_inputs,
            model=data["model"],
            dynamic_shapes=data["dynamic_shapes"],
        )
        if verbose:
            print(f"[validate_model] new inputs: {string_type(data['inputs'])}")
            print(f"[validate_model] new dynamic_hapes: {_ds_clean(data['dynamic_shapes'])}")

    if not empty(dtype):
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        if verbose:
            print(f"[validate_model] dtype conversion to {dtype}")
        data["model"] = to_any(data["model"], dtype)  # type: ignore
        data["inputs"] = to_any(data["inputs"], dtype)  # type: ignore
        summary["model_dtype"] = str(dtype)

    if not empty(device):
        if verbose:
            print(f"[validate_model] device conversion to {device}")
        data["model"] = to_any(data["model"], device)  # type: ignore
        data["inputs"] = to_any(data["inputs"], device)  # type: ignore
        summary["model_device"] = str(device)

    summary["time_create"] = time.perf_counter() - begin
    for k in ["task", "size", "n_weights"]:
        summary[f"model_{k.replace('_','')}"] = data[k]
        summary["model_inputs"] = string_type(data["inputs"], with_shape=True)
        summary["model_shapes"] = _ds_clean(str(data["dynamic_shapes"]))
        summary["model_class"] = data["model"].__class__.__name__
        summary["model_config_class"] = data["configuration"].__class__.__name__
        summary["model_config"] = str(data["configuration"].to_dict()).replace(" ", "")
    summary["model_id"] = model_id

    if verbose:
        print(f"[validate_model] task={data['task']}")
        print(f"[validate_model] size={data['size'] / 2**20} Mb")
        print(f"[validate_model] n_weights={data['n_weights'] / 1e6} millions parameters")
        for k, v in data["inputs"].items():
            print(f"[validate_model] +INPUT {k}={string_type(v, with_shape=True)}")
        for k, v in data["dynamic_shapes"].items():
            print(f"[validate_model] +SHAPE {k}={_ds_clean(v)}")

    if do_run:
        if verbose:
            print("[validate_model] run the model...")
            print(f"[validate_model] inputs={string_type(data['inputs'], with_shape=True)}")
        # We make a copy of the input just in case the model modifies them inplace
        hash_inputs = string_type(data["inputs"], with_shape=True)
        inputs = torch_deepcopy(data["inputs"])
        begin = time.perf_counter()
        if quiet:
            try:
                expected = data["model"](**inputs)
            except Exception as e:
                summary["ERR_run"] = str(e)
                data["ERR_run"] = e
                summary["time_run"] = time.perf_counter() - begin
                return summary, data
        else:
            expected = data["model"](**inputs)
        summary["time_run"] = time.perf_counter() - begin
        summary["model_expected"] = string_type(expected, with_shape=True)
        if verbose:
            print("[validate_model] done (run)")
        data["expected"] = expected
        assert hash_inputs == string_type(data["inputs"], with_shape=True), (
            f"The model execution did modified the inputs:\n"
            f"before: {hash_inputs}\n"
            f" after: {string_type(data['inputs'], with_shape=True)}"
        )

    if exporter:
        print(
            f"[validate_model] export the model with {exporter!r}, "
            f"optimization={optimization!r}"
        )
        if patch:
            if verbose:
                print("[validate_model] applies patches before exporting")
            with bypass_export_some_errors(  # type: ignore
                patch_transformers=True,
                stop_if_static=stop_if_static,
                verbose=max(0, verbose - 1),
            ) as modificator:
                data["inputs_export"] = modificator(data["inputs"])  # type: ignore

                if do_run:
                    # We run a second time the model to check the patch did not
                    # introduce any discrepancies
                    if verbose:
                        print("[validate_model] run patched model...")
                        print(
                            f"[validate_model] patched inputs="
                            f"{string_type(data['inputs_export'], with_shape=True)}"
                        )
                    hash_inputs = string_type(data["inputs_export"], with_shape=True)

                    # We make a copy of the input just in case the model modifies them inplace
                    inputs = torch_deepcopy(data["inputs_export"])
                    begin = time.perf_counter()
                    if quiet:
                        try:
                            expected = data["model"](**inputs)
                        except Exception as e:
                            summary["ERR_run_patched"] = str(e)
                            data["ERR_run_patched"] = e
                            summary["time_run_patched"] = time.perf_counter() - begin
                            return summary, data
                    else:
                        expected = data["model"](**inputs)
                    summary["time_run_patched"] = time.perf_counter() - begin
                    disc = max_diff(data["expected"], expected)
                    for k, v in disc.items():
                        summary[f"disc_patched_{k}"] = v
                    if verbose:
                        print("[validate_model] done (patched run)")
                        print(f"[validate_model] patched discrepancies={string_diff(disc)}")
                    assert hash_inputs == string_type(
                        data["inputs_export"], with_shape=True
                    ), (
                        f"The model execution did modified the inputs:\n"
                        f"before: {hash_inputs}\n"
                        f" after: {string_type(data['inputs_export'], with_shape=True)}"
                    )

                # data is modified inplace
                summary_export, data = call_exporter(
                    exporter=exporter,
                    data=data,
                    quiet=quiet,
                    verbose=verbose,
                    optimization=optimization,
                    do_run=do_run,
                )
        else:
            data["inputs_export"] = data["inputs"]
            # data is modified inplace
            summary_export, data = call_exporter(
                exporter=exporter,
                data=data,
                quiet=quiet,
                verbose=verbose,
                optimization=optimization,
                do_run=do_run,
            )
        summary.update(summary_export)

    if dump_folder:
        if "exported_program" in data:
            ep = data["exported_program"]
            if verbose:
                print(f"[validate_model] dumps exported program in {dump_folder!r}...")
            with open(os.path.join(dump_folder, f"{folder_name}.ep"), "w") as f:
                f.write(str(ep))
            with open(os.path.join(dump_folder, f"{folder_name}.graph"), "w") as f:
                f.write(str(ep.graph))
            if verbose:
                print("[validate_model] done (dump ep)")
        if "onnx_program" in data:
            epo = data["onnx_program"]
            if verbose:
                print(f"[validate_model] dumps onnx program in {dump_folder!r}...")
            onnx_file_name = os.path.join(dump_folder, f"{folder_name}.onnx")
            epo.save(onnx_file_name, external_data=True)
            if verbose:
                print("[validate_model] done (dump onnx)")
        if verbose:
            print(f"[validate_model] dumps statistics in {dump_folder!r}...")
        with open(os.path.join(dump_folder, f"{folder_name}.stats"), "w") as f:
            for k, v in sorted(summary.items()):
                f.write(f":{k}:{v};\n")
        if verbose:
            print("[validate_model] done (dump)")

    if exporter and exporter.startswith("onnx-") and do_run:
        summary_valid, data = validate_onnx_model(
            data=data,
            quiet=quiet,
            verbose=verbose,
            optimization=optimization,
        )
        summary.update(summary_valid)

    if verbose:
        print("[validate_model] done (final)")
    return summary, data


def call_exporter(
    data: Dict[str, Any],
    exporter: str,
    quiet: bool = False,
    verbose: int = 0,
    optimization: Optional[str] = None,
    do_run: bool = False,
) -> Tuple[Dict[str, Union[int, float, str]], Dict[str, Any]]:
    """
    Calls an exporter on a model;
    If a patch must be applied, it should be before this functions.

    :param data: dictionary with all the necessary inputs
    :param exporter: exporter to call
    :param quiet: catch exception or not
    :param verbose: verbosity
    :param optimization: optimization to do
    :param do_run: runs and compute discrepancies
    :return: two dictionaries, one with some metrics,
        another one with whatever the function produces
    """
    if exporter.startswith("export-"):
        # torch export
        summary, data = call_torch_export_export(
            exporter=exporter,
            data=data,
            quiet=quiet,
            verbose=verbose,
            optimization=optimization,
            do_run=do_run,
        )
        return summary, data
    if exporter.startswith("onnx-"):
        # torch export
        summary, data = call_torch_export_onnx(
            exporter=exporter,
            data=data,
            quiet=quiet,
            verbose=verbose,
            optimization=optimization,
        )
        return summary, data
    raise NotImplementedError(
        f"export with {exporter!r} and optimization={optimization!r} not implemented yet"
    )


def call_torch_export_export(
    data: Dict[str, Any],
    exporter: str,
    quiet: bool = False,
    verbose: int = 0,
    optimization: Optional[str] = None,
    do_run: bool = False,
):
    """
    Exports a model with :func:`torch.export.export`.
    If a patch must be applied, it should be before this functions.

    :param data: dictionary with all the necessary inputs, the dictionary must
        contains keys ``model`` and ``inputs_export``
    :param exporter: exporter to call
    :param quiet: catch exception or not
    :param verbose: verbosity
    :param optimization: optimization to do
    :param do_run: runs and compute discrepancies
    :return: two dictionaries, one with some metrics,
        another one with whatever the function produces
    """
    assert exporter in {
        "export-strict",
        "export-nostrict",
    }, f"Unexpected value for exporter={exporter!r}"
    assert "model" in data, f"model is missing from data: {sorted(data)}"
    assert "inputs_export" in data, f"inputs_export is missing from data: {sorted(data)}"
    summary: Dict[str, Union[str, int, float]] = {}
    strict = "nostrict" not in exporter
    args, kwargs = split_args_kwargs(data["inputs_export"])
    ds = data.get("dynamic_shapes", None)
    if verbose:
        print(
            f"[call_torch_export_export] exporter={exporter!r}, "
            f"strict={strict}, optimization={optimization!r}"
        )
        print(f"[call_torch_export_export] args={string_type(args, with_shape=True)}")
        print(f"[call_torch_export_export] kwargs={string_type(kwargs, with_shape=True)}")
        print(f"[call_torch_export_export] dynamic_shapes={_ds_clean(ds)}")
        print("[call_torch_export_export] export...")
    summary["export_exporter"] = exporter
    summary["export_optimization"] = optimization or ""
    summary["export_strict"] = strict
    summary["export_args"] = string_type(args, with_shape=True)
    summary["export_kwargs"] = string_type(kwargs, with_shape=True)

    begin = time.perf_counter()
    if quiet:
        try:
            ep = torch.export.export(
                data["model"], args, kwargs=kwargs, dynamic_shapes=ds, strict=strict
            )
        except Exception as e:
            summary["ERR_export_export"] = str(e)
            data["ERR_export_export"] = e
            summary["time_export_export"] = time.perf_counter() - begin
            return summary, data
    else:
        ep = torch.export.export(
            data["model"], args, kwargs=kwargs, dynamic_shapes=ds, strict=strict
        )

    summary["time_export_export"] = time.perf_counter() - begin
    summary["export_graph_nodes"] = len(ep.graph.nodes)
    if verbose:
        print(
            f"[call_torch_export_export] done (export) "
            f"with {summary['export_graph_nodes']} nodes"
        )
    data["exported_program"] = ep
    if verbose > 1:
        print("[call_torch_export_export] -- ExportedProgram")
        print(ep)
        print("[call_torch_export_export] -- End of ExportedProgram")

    if do_run:
        # We check for discrepancies.
        if verbose:
            print("[validate_model] run exported model...")
            print(
                f"[validate_model] patched inputs="
                f"{string_type(data['inputs_export'], with_shape=True)}"
            )
        hash_inputs = string_type(data["inputs_export"], with_shape=True)

        # We make a copy of the input just in case the model modifies them inplace
        inputs = torch_deepcopy(data["inputs_export"])
        model = ep.module()
        begin = time.perf_counter()
        if quiet:
            try:
                expected = model(**inputs)
            except Exception as e:
                summary["ERR_run_exported"] = str(e)
                data["ERR_run_exported"] = e
                summary["time_run_exported"] = time.perf_counter() - begin
                return summary, data
        else:
            expected = model(**inputs)
        summary["time_run_exported"] = time.perf_counter() - begin
        disc = max_diff(data["expected"], expected)
        for k, v in disc.items():
            summary[f"disc_exported_{k}"] = v
        if verbose:
            print("[validate_model] done (exported run)")
            print(f"[validate_model] exported discrepancies={string_diff(disc)}")
        assert hash_inputs == string_type(data["inputs_export"], with_shape=True), (
            f"The exported model execution did modified the inputs:\n"
            f"before: {hash_inputs}\n"
            f" after: {string_type(data['inputs_export'], with_shape=True)}"
        )
    return summary, data


def call_torch_export_onnx(
    data: Dict[str, Any],
    exporter: str,
    quiet: bool = False,
    verbose: int = 0,
    optimization: Optional[str] = None,
):
    """
    Exports a model into onnx.
    If a patch must be applied, it should be before this functions.

    :param data: dictionary with all the necessary inputs, the dictionary must
        contains keys ``model`` and ``inputs_export``
    :param exporter: exporter to call
    :param quiet: catch exception or not
    :param verbose: verbosity
    :param optimization: optimization to do
    :return: two dictionaries, one with some metrics,
        another one with whatever the function produces
    """
    assert optimization in {
        "",
        "ir",
        None,
    }, f"unexpected value for optimization={optimization}"
    assert exporter in {
        "onnx-dynamo",
        "onnx-script",
    }, f"Unexpected value for exporter={exporter!r}"
    assert "model" in data, f"model is missing from data: {sorted(data)}"
    assert "inputs_export" in data, f"inputs_export is missing from data: {sorted(data)}"
    summary: Dict[str, Union[str, int, float]] = {}
    dynamo = "nostrict" not in exporter
    args, kwargs = split_args_kwargs(data["inputs_export"])
    ds = data.get("dynamic_shapes", None)
    if verbose:
        print(
            f"[call_torch_export_onnx] exporter={exporter!r}, "
            f"optimization={optimization!r}"
        )
        print(f"[call_torch_export_onnx] args={string_type(args, with_shape=True)}")
        print(f"[call_torch_export_onnx] kwargs={string_type(kwargs, with_shape=True)}")
        print(f"[call_torch_export_onnx] dynamic_shapes={_ds_clean(ds)}")
        print("[call_torch_export_onnx] export...")
    summary["export_exporter"] = exporter
    summary["export_optimization"] = optimization or ""
    summary["export_dynamo"] = dynamo
    summary["export_args"] = string_type(args, with_shape=True)
    summary["export_kwargs"] = string_type(kwargs, with_shape=True)

    begin = time.perf_counter()
    if quiet:
        try:
            epo = torch.onnx.export(
                data["model"],
                args,
                kwargs=kwargs,
                dynamic_shapes=ds,
                dynamo=dynamo,
            )
        except Exception as e:
            summary["ERR_export_export"] = str(e)
            data["ERR_export_export"] = e
            summary["time_export_export"] = time.perf_counter() - begin
            return summary, data
    else:
        epo = torch.onnx.export(
            data["model"],
            args,
            kwargs=kwargs,
            dynamic_shapes=ds,
            dynamo=dynamo,
        )

    summary["time_export_export"] = time.perf_counter() - begin
    assert epo is not None, "no onnx export was found"
    if verbose:
        print("[call_torch_export_onnx] done (export)")
    data["onnx_program"] = epo
    if verbose > 1:
        print("[call_torch_export_onnx] -- ONNXProgram")
        print(epo)
        print("[call_torch_export_onnx] -- End of ONNXProgram")

    begin = time.perf_counter()
    if optimization == "ir":
        if verbose:
            print(f"[call_torch_export_onnx] starts optimization={optimization!r}...")
        if quiet:
            try:
                epo.optimize()
            except Exception as e:
                summary["ERR_export_optimize_ir"] = str(e)
                data["ERR_export_optimize_ir"] = e
                summary["time_export_optimize_ir"] = time.perf_counter() - begin
                return summary, data
        else:
            epo.optimize()
        summary["time_export_optimize_ir"] = time.perf_counter() - begin
        if verbose:
            print("[call_torch_export_onnx] done (optimization)")

    return summary, data


def validate_onnx_model(
    data: Dict[str, Any],
    quiet: bool = False,
    verbose: int = 0,
    optimization: Optional[str] = None,
):
    """
    Verifies that an onnx model produces the same
    expected outputs.

    :param data: dictionary with all the necessary inputs, the dictionary must
        contains keys ``model`` and ``inputs_export``
    :param quiet: catch exception or not
    :param verbose: verbosity
    :param optimization: optimization to do
    :return: two dictionaries, one with some metrics,
        another one with whatever the function produces
    """
    import onnxruntime

    summary = {}
    flat_inputs = flatten_object(data["inputs"], drop_keys=True)
    d = flat_inputs[0].get_device()
    providers = (
        ["CPUExecutionProvider"]
        if d < 0
        else ["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    if "onnx_file_name" in data:
        source = data["onnx_file_name"]
        summary["onnx_filename"] = source
        summary["onnx_size"] = os.stats(source).st_size
    else:
        assert (
            "onnx_program" in data
        ), f"onnx_program is missing from data which has {sorted(data)}"
        source = data["onnx_program"].model_proto.SerializeToString()
        assert len(source) < 2**31, f"The model is highger than 2Gb: {len(source) / 2**30} Gb"
        summary["onnx_size"] = len(source)
    if verbose:
        print(f"[validate_onnx_model] verify onnx model with providers {providers}...")

    begin = time.perf_counter()
    if quiet:
        try:
            sess = onnxruntime.InferenceSession(source, providers=providers)
        except Exception as e:
            summary["ERR_onnx_ort_create"] = str(e)
            data["ERR_onnx_ort_create"] = e
            summary["time_onnx_ort_create"] = time.perf_counter() - begin
            return summary, data
    else:
        sess = onnxruntime.InferenceSession(source, providers=providers)

    summary["time_onnx_ort_create"] = time.perf_counter() - begin
    data["onnx_ort_sess"] = sess
    if verbose:
        print("[validate_onnx_model] done (ort_session)")

    # make_feeds
    if verbose:
        print("[validate_onnx_model] make_feeds...")
        print(f"[validate_onnx_model] inputs={string_type(data['inputs'], with_shape=True)}")
    feeds = make_feeds([i.name for i in sess.get_inputs()], data["inputs"], use_numpy=True)
    if verbose:
        print(f"[validate_onnx_model] ort inputs={string_type(feeds, with_shape=True)}")
    summary["onnx_ort_inputs"] = string_type(feeds, with_shape=True)
    if verbose:
        print("[validate_onnx_model] done (make_feeds)")

    # run ort
    if verbose:
        print("[validate_onnx_model] run session...")
    begin = time.perf_counter()
    if quiet:
        try:
            got = sess.run(None, feeds)
        except Exception as e:
            summary["ERR_onnx_ort_run"] = str(e)
            data["ERR_onnx_ort_run"] = e
            summary["time_onnx_ort_run"] = time.perf_counter() - begin
            return summary, data
    else:
        got = sess.run(None, feeds)
    if verbose:
        print("[validate_onnx_model] done (run)")
        print(f"[validate_onnx_model] got={string_type(got, with_shape=True)}")

    # compute discrepancies
    disc = max_diff(data["expected"], got, flatten=True)
    if verbose:
        print(f"[validate_onnx_model] discrepancies={string_diff(disc)}")
    for k, v in disc.items():
        summary[f"disc_onnx_ort_run_{k}"] = v
    return summary, data
