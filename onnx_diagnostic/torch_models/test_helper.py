import os
from typing import Any, Dict, Optional, Tuple, Union
import time
import torch
from ..helpers import max_diff, string_type, string_diff
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
    dump_folder: Optional[str] = None,
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
    :param patch: applies patches before exporting
    :param dump_folder: dumps everything in a subfolder of this one
    :return: two dictionaries, one with some metrics,
        another one with whatever the function produces
    """
    assert not trained, f"trained={trained} not supported yet"
    summary: Dict[str, Union[int, float, str]] = {}
    if dump_folder:
        folder_name = f"{model_id.replace('/','-')}-{exporter}-{optimization or ''}"
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
                patch_transformers=True, verbose=max(0, verbose - 1)
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
        if verbose:
            print(f"[validate_model] dumps statistics in {dump_folder!r}...")
        with open(os.path.join(dump_folder, f"{folder_name}.stats"), "w") as f:
            for k, v in sorted(summary.items()):
                f.write(f":{k}:{v};\n")
        if verbose:
            print("[validate_model] done (dump)")

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
    :param patch: apply patches
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
    raise NotImplementedError(
        f"export with {exporter!r} and optimization={optimization!r} not implemented yet"
    )


def split_args_kwargs(inputs: Any) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    """
    Splits into args, kwargs.
    """
    if isinstance(inputs, dict):
        return (), inputs
    if isinstance(inputs, tuple) and len(inputs) == 2 and isinstance(inputs[1], dict):
        return inputs
    assert isinstance(inputs, tuple), f"Unexpected inputs {string_type(inputs)}"
    return inputs, {}


def call_torch_export_export(
    data: Dict[str, Any],
    exporter: str,
    quiet: bool = False,
    verbose: int = 0,
    optimization: Optional[str] = None,
    do_run: bool = False,
):
    """
    Calls an exporter on a model;
    If a patch must be applied, it should be before this functions.

    :param data: dictionary with all the necessary inputs
    :param exporter: exporter to call
    :param quiet: catch exception or not
    :param verbose: verbosity
    :param patch: apply patches
    :param optimization: optimization to do
    :param do_run: runs and compute discrepancies
    :return: two dictionaries, one with some metrics,
        another one with whatever the function produces
    """
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
        print(f"[call_torch_export_export] args={string_type(args)}")
        print(f"[call_torch_export_export] kwargs={string_type(kwargs)}")
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
