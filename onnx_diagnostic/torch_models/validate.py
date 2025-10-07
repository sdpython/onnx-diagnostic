import gc
import datetime
import inspect
import os
import pprint
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import time
import numpy as np
import onnx
import torch
from ..export import CoupleInputsDynamicShapes
from ..helpers import max_diff, string_type, string_diff
from ..helpers.helper import flatten_object
from ..helpers.rt_helper import make_feeds, reorder_modelbuilder_cache_to_torch
from ..helpers.torch_helper import to_any, torch_deepcopy
from ..helpers.cache_helper import flatten_unflatten_for_dynamic_shapes
from ..tasks import random_input_kwargs
from ..torch_export_patches import torch_export_patches
from ..torch_export_patches.patch_inputs import use_dyn_not_str
from .hghub import get_untrained_model_with_inputs


def empty(value: Any) -> bool:
    """Tells if the value is empty."""
    if isinstance(value, (str, list, dict, tuple, set)):
        return not bool(value)
    if value is None:
        return True
    return False


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
    subfolder: Optional[str] = None,
    opset: Optional[int] = None,
    drop_inputs: Optional[List[str]] = None,
    same_as_pretrained: bool = False,
    use_pretrained: bool = False,
) -> str:
    "Creates a filename unique based on the given options."
    els = [model_id.replace("/", "_")]
    if subfolder:
        els.append(subfolder.replace("/", "_"))
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
    if opset is not None:
        els.append(f"op{opset}")
    if drop_inputs:
        ii = "-".join(f"{s[0]}{s[-1]}" for s in drop_inputs)
        els.append(f"I-{ii.upper()}")
    if use_pretrained:
        els.append("TRAINED")
    elif same_as_pretrained:
        els.append("SAMESIZE")
    return "-".join(els)


def version_summary() -> Dict[str, Union[int, float, str]]:
    """
    Example:

    .. runpython::
        :showcode:

        import pprint
        from onnx_diagnostic.torch_models.validate import version_summary

        pprint.pprint(version_summary())
    """
    import numpy

    summary: Dict[str, Union[int, float, str]] = {
        "version_torch": torch.__version__,
        "version_numpy": numpy.__version__,
    }
    try:
        import scipy

        summary["version_scipy"] = getattr(scipy, "__version__", "?")
    except ImportError:
        pass
    try:
        import transformers

        summary["version_transformers"] = getattr(transformers, "__version__", "?")
    except ImportError:
        pass
    try:
        import onnx

        summary["version_onnx"] = getattr(onnx, "__version__", "?")
    except ImportError:
        pass
    try:
        import onnxscript

        summary["version_onnxscript"] = getattr(onnxscript, "__version__", "?")
    except ImportError:
        pass
    try:
        import onnxruntime

        summary["version_onnxruntime"] = getattr(onnxruntime, "__version__", "?")
    except ImportError:
        pass
    try:
        import onnx_ir

        summary["version_onnx_ir"] = getattr(onnx_ir, "__version__", "?")
    except ImportError:
        pass
    import onnx_diagnostic

    summary["version_onnx_diagnostic"] = onnx_diagnostic.__version__
    summary["version_date"] = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    return summary


def _quiet_or_not_quiet(
    quiet: bool,
    suffix: str,
    summary: Dict[str, Any],
    data: Optional[Dict[str, Any]],
    fct: Callable,
    repeat: int = 1,
    warmup: int = 0,
) -> Any:
    begin = time.perf_counter()
    if quiet:
        try:
            res = fct()
            summary[f"time_{suffix}"] = time.perf_counter() - begin
            if warmup + repeat == 1:
                return res
        except Exception as e:
            summary[f"ERR_{suffix}"] = str(e)
            summary[f"time_{suffix}"] = time.perf_counter() - begin
            if data is None:
                return {f"ERR_{suffix}": e}
            data[f"ERR_{suffix}"] = e
            return None
    else:
        res = fct()
    summary[f"time_{suffix}"] = time.perf_counter() - begin
    if warmup + repeat > 1:
        if suffix == "run":
            res = torch_deepcopy(res)
        summary[f"{suffix}_output"] = string_type(res, with_shape=True, with_min_max=True)
        summary[f"{suffix}_warmup"] = warmup
        summary[f"{suffix}_repeat"] = repeat
        last_ = None
        end_w = max(0, warmup - 1)
        for _w in range(end_w):
            t = fct()
            _ = string_type(t, with_shape=True, with_min_max=True)
            if _ != last_ or _w == end_w - 1:
                summary[f"io_{suffix}_{_w+1}"] = _
                last_ = _
        summary[f"time_{suffix}_warmup"] = time.perf_counter() - begin
        times = []
        for _r in range(repeat):
            begin = time.perf_counter()
            t = fct()
            times.append(time.perf_counter() - begin)
        a = np.array(times, dtype=np.float64)
        a.sort()
        i5 = max(1, a.shape[0] * 5 // 100)
        i2 = max(1, a.shape[0] * 2 // 100)
        summary[f"time_{suffix}_latency"] = a.mean()
        summary[f"time_{suffix}_latency_std"] = a.std()
        summary[f"time_{suffix}_latency_min"] = a.min()
        summary[f"time_{suffix}_latency_max"] = a.max()
        summary[f"time_{suffix}_latency_098"] = a[-i2]
        summary[f"time_{suffix}_latency_095"] = a[-i5]
        summary[f"time_{suffix}_latency_005"] = a[i5]
        summary[f"time_{suffix}_latency_002"] = a[i2]
        summary[f"time_{suffix}_n"] = len(a)
        summary[f"time_{suffix}_latency_m98"] = a[i2:-i2].mean()

    return res


def shrink_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Shrinks the configuration before it gets added to the information to log."""
    new_cfg = {}
    for k, v in cfg.items():

        new_cfg[k] = (
            v
            if (not isinstance(v, (list, tuple, set, dict)) or len(v) < 50)
            else (v.__class__("...") if isinstance(v, (list, tuple)) else "...")
        )
    return new_cfg


def _preprocess_model_id(
    model_id: str, subfolder: Optional[str], same_as_pretrained: bool, use_pretrained: bool
) -> Tuple[str, Optional[str], bool, bool]:
    if subfolder or "//" not in model_id:
        return model_id, subfolder, same_as_pretrained, use_pretrained
    spl = model_id.split("//")
    if spl[-1] == "pretrained":
        return _preprocess_model_id("//".join(spl[:-1]), "", True, True)
    if spl[-1] in {"transformer", "vae"}:
        # known subfolder
        return "//".join(spl[:-1]), spl[-1], same_as_pretrained, use_pretrained
    return model_id, subfolder, same_as_pretrained, use_pretrained


def validate_model(
    model_id: str,
    task: Optional[str] = None,
    do_run: bool = False,
    exporter: Optional[str] = None,
    do_same: bool = False,
    verbose: int = 0,
    dtype: Optional[Union[str, torch.dtype]] = None,
    device: Optional[Union[str, torch.device]] = None,
    same_as_pretrained: bool = False,
    use_pretrained: bool = False,
    optimization: Optional[str] = None,
    quiet: bool = False,
    patch: Union[bool, str, Dict[str, bool]] = False,
    rewrite: bool = False,
    stop_if_static: int = 1,
    dump_folder: Optional[str] = None,
    drop_inputs: Optional[List[str]] = None,
    ortfusiontype: Optional[str] = None,
    input_options: Optional[Dict[str, Any]] = None,
    model_options: Optional[Dict[str, Any]] = None,
    subfolder: Optional[str] = None,
    opset: Optional[int] = None,
    runtime: str = "onnxruntime",
    repeat: int = 1,
    warmup: int = 0,
    inputs2: int = 1,
    output_names: Optional[List[str]] = None,
    ort_logs: bool = False,
) -> Tuple[Dict[str, Union[int, float, str]], Dict[str, Any]]:
    """
    Validates a model.
    The function can also be called through the command line
    :ref:`l-cmd-validate`.

    :param model_id: model id to validate
    :param task: task used to generate the necessary inputs,
        can be left empty to use the default task for this model
        if it can be determined
    :param do_run: checks the model works with the defined inputs
    :param exporter: exporter the model using this exporter,
        available list: ``export-strict``, ``export-nostrict``, ...
        see below
    :param do_same: checks the discrepancies of the exported model
    :param verbose: verbosity level
    :param dtype: uses this dtype to check the model
    :param device: do the verification on this device
    :param same_as_pretrained: use a model equivalent to the trained,
        this is not always possible
    :param use_pretrained: use the trained model, not the untrained one
    :param optimization: optimization to apply to the exported model,
        depend on the the exporter
    :param quiet: if quiet, catches exception if any issue
    :param patch: applies patches (``patch_transformers=True, path_diffusers=True``)
        if True before exporting
        see :func:`onnx_diagnostic.torch_export_patches.torch_export_patches`,
        a string can be used to specify only one of them
    :param rewrite: applies known rewriting (``patch_transformers=True``) before exporting,
        see :func:`onnx_diagnostic.torch_export_patches.torch_export_patches`
    :param stop_if_static: stops if a dynamic dimension becomes static,
        see :func:`onnx_diagnostic.torch_export_patches.torch_export_patches`
    :param dump_folder: dumps everything in a subfolder of this one
    :param drop_inputs: drops this list of inputs (given their names)
    :param ortfusiontype: runs ort fusion, the parameters defines the fusion type,
        it accepts multiple values separated by ``|``,
        see :func:`onnx_diagnostic.torch_models.validate.run_ort_fusion`
    :param input_options: additional options to define the dummy inputs
        used to export
    :param model_options: additional options when creating the model such as
        ``num_hidden_layers`` or ``attn_implementation``
    :param subfolder: version or subfolders to uses when retrieving a model id
    :param opset: onnx opset to use for the conversion
    :param runtime: onnx runtime to use to check about discrepancies,
        possible values ``onnxruntime``, ``torch``, ``orteval``,
        ``orteval10``, ``ref`` only if `do_run` is true
    :param repeat: number of time to measure the model
    :param warmup: warmup the model first
    :param inputs2: checks that other sets of inputs are running as well,
        this ensures that the model does support dynamism, the value is used
        as an increment to the first set of values (added to dimensions),
        or an empty cache for example
    :param output_names: output names the onnx exporter should use
    :param ort_logs: increases onnxruntime verbosity when creating the session
    :return: two dictionaries, one with some metrics,
        another one with whatever the function produces

    The following environment variables can be used to print out some
    information:

    * ``PRINT_CONFIG``: prints the model configuration

    The following exporters are available:

    * ``export-nostrict``: run :func:`torch.export.export` (..., strict=False)
    * ``onnx-dynamo``: run :func:`torch.onnx.export` (...),
      models can be optimized with ``optimization`` in ``("ir", "os_ort")``
    * ``modelbuilder``: use :epkg:`ModelBuilder` to builds the onnx model
    * ``custom``: custom exporter (see :epkg:`experimental-experiment`),
      models can be optimized with ``optimization`` in
      ``("default", "default+onnxruntime", "default+os_ort", "default+onnxruntime+os_ort")``

    The default runtime, :epkg:`onnxruntime` is used to validate a model and check the
    exported model returns the same outputs as the original one, otherwise,
    :class:`onnx_diagnostic.reference.TorchOnnxEvaluator`
    if ``runtime == 'torch'`` or
    :class:`onnx_diagnostic.reference.OnnxruntimeEvaluator`
    if ``runtime == 'orteval'`` or
    :class:`onnx_diagnostic.reference.ExtendedReferenceEvaluator`
    if ``runtime == 'ref'``,
    ``orteval10`` increases the verbosity.

    .. versionchanged:: 0.7.13
        *inputs2* not only means a second set of inputs but many
        such as ``input_empty_cache``
        which refers to a set of inputs using an empty cache.
    """
    main_validation_begin = time.perf_counter()
    model_id, subfolder, same_as_pretrained, use_pretrained = _preprocess_model_id(
        model_id,
        subfolder,
        same_as_pretrained=same_as_pretrained,
        use_pretrained=use_pretrained,
    )
    time_preprocess_model_id = time.perf_counter() - main_validation_begin
    default_patch = dict(patch_transformers=True, patch_diffusers=True, patch=True)
    if isinstance(patch, bool):
        patch_kwargs = default_patch if patch else dict(patch=False)
    elif isinstance(patch, str):
        patch_kwargs = {"patch": True, **{p: True for p in patch.split(",")}}  # noqa: C420
    else:
        assert isinstance(patch, dict), f"Unable to interpret patch={patch!r}"
        patch_kwargs = patch.copy()
        if "patch" not in patch_kwargs:
            if any(patch_kwargs.values()):
                patch_kwargs["patch"] = True
        elif len(patch) == 1 and patch.get("patch", False):
            patch_kwargs.update(default_patch)

    assert not rewrite or patch_kwargs.get("patch", False), (
        f"rewrite={rewrite}, patch={patch}, patch_kwargs={patch_kwargs} "
        f"patch must be True to enable rewriting, "
        f"if --patch=0 was specified on the command line, rewrites are disabled."
    )
    summary = version_summary()
    summary.update(
        dict(
            version_model_id=model_id,
            version_do_run=str(do_run),
            version_dtype=str(dtype or ""),
            version_device=str(device or ""),
            version_same_as_pretrained=str(same_as_pretrained),
            version_use_pretrained=str(use_pretrained),
            version_optimization=optimization or "",
            version_quiet=str(quiet),
            version_patch=str(patch),
            version_patch_kwargs=str(patch_kwargs).replace(" ", ""),
            version_rewrite=str(rewrite),
            version_dump_folder=dump_folder or "",
            version_drop_inputs=str(list(drop_inputs or "")),
            version_ortfusiontype=ortfusiontype or "",
            version_stop_if_static=str(stop_if_static),
            version_exporter=exporter or "",
            version_runtime=runtime,
            version_inputs2=inputs2,
            time_preprocess_model_id=time_preprocess_model_id,
        )
    )
    if opset:
        summary["version_opset"] = opset

    folder_name = None
    if dump_folder:
        folder_name = _make_folder_name(
            model_id,
            exporter,
            optimization,
            dtype=dtype,
            device=device,
            subfolder=subfolder,
            opset=opset,
            drop_inputs=drop_inputs,
            use_pretrained=use_pretrained,
            same_as_pretrained=same_as_pretrained,
        )
        dump_folder = os.path.join(dump_folder, folder_name)
        if not os.path.exists(dump_folder):
            os.makedirs(dump_folder)
        summary["dump_folder"] = dump_folder
        summary["dump_folder_name"] = folder_name
        if verbose:
            print(f"[validate_model] dump into {folder_name!r}")

    if verbose:
        if subfolder:
            print(f"[validate_model] validate model id {model_id!r}, subfolder={subfolder!r}")
        else:
            print(f"[validate_model] validate model id {model_id!r}")
        print(f"[validate_model] patch={patch!r}")
        if model_options:
            print(f"[validate_model] model_options={model_options!r}")
        print(f"[validate_model] get dummy inputs with input_options={input_options}...")
        print(
            f"[validate_model] rewrite={rewrite}, patch_kwargs={patch_kwargs}, "
            f"stop_if_static={stop_if_static}"
        )
        print(f"[validate_model] exporter={exporter!r}, optimization={optimization!r}")
        print(f"[validate_model] dump_folder={dump_folder!r}")
        print(f"[validate_model] output_names={output_names}")
        summary["model_id"] = model_id
        summary["model_subfolder"] = subfolder or ""

    iop = input_options or {}
    mop = model_options or {}
    data = _quiet_or_not_quiet(
        quiet,
        "create_torch_model",
        summary,
        None,
        (
            lambda mid=model_id, v=verbose, task=task, uptr=use_pretrained, tr=same_as_pretrained, iop=iop, sub=subfolder, i2=inputs2: (  # noqa: E501
                get_untrained_model_with_inputs(
                    mid,
                    verbose=v,
                    task=task,
                    use_pretrained=uptr,
                    same_as_pretrained=tr,
                    inputs_kwargs=iop,
                    model_kwargs=mop,
                    subfolder=sub,
                    add_second_input=i2,
                )
            )
        ),
    )

    second_input_keys = [k for k in data if k.startswith("inputs") and k != "inputs"]

    if dump_folder:
        with open(os.path.join(dump_folder, "model_config.txt"), "w") as f:
            f.write(f"model_id: {model_id}\n------\n")
            f.write(
                pprint.pformat(
                    data["configuration"]
                    if type(data["configuration"]) is dict
                    else data["configuration"].to_dict()
                )
            )
        dump_info = data.get("dump_info", None)
        if dump_info:
            with open(os.path.join(dump_folder, "model_dump_info.txt"), "w") as f:
                f.write(f"model_id: {model_id}\n------\n")
                f.write(pprint.pformat(dump_info))

    if exporter == "modelbuilder":
        # Models used with ModelBuilder do not like batch size > 1.
        # Let's change that.
        for k in ["inputs", "inputs2"]:
            if k not in data:
                continue
            if verbose:
                print(f"[validate_model] set batch=1 for data[{k!r}]")
                print(f"[validate_model] batch=1 === {string_type(data[k], with_shape=True)}")
            cpl = CoupleInputsDynamicShapes(
                tuple(), data[k], dynamic_shapes=data["dynamic_shapes"]
            )
            if patch_kwargs.get("patch", False):
                with torch_export_patches(**patch_kwargs):  # type: ignore[arg-type]
                    data[k] = cpl.change_dynamic_dimensions(
                        desired_values=dict(batch=1), only_desired=True
                    )
            else:
                data[k] = cpl.change_dynamic_dimensions(
                    desired_values=dict(batch=1), only_desired=True
                )
            if verbose:
                print(f"[validate_model] batch=1 --> {string_type(data[k], with_shape=True)}")

    # modelbuilder needs different treatments sometimes, so
    # we mark it for later usage.
    # for example, it has different past_kv ordering than
    # flattened CacheObject
    data["exporter"] = exporter
    data["input_options"] = iop
    data["model_options"] = mop
    data["model_dump_folder"] = dump_folder
    if dtype:
        data["model_dtype"] = dtype if isinstance(dtype, str) else str(dtype)
    if device:
        data["model_device"] = str(device)
    if opset:
        data["model_opset"] = opset
    if "rewrite" in data:
        if rewrite:
            summary["model_rewrite"] = str(data["rewrite"])
            if verbose:
                print(f"[validate_model] model_rewrite={summary['model_rewrite']}")
        else:
            del data["rewrite"]
            if verbose:
                print("[validate_model] no rewrite")
    if os.environ.get("PRINT_CONFIG", "0") in (1, "1"):
        print("[validate_model] -- PRINT CONFIG")
        print("-- type(config)", type(data["configuration"]))
        print(data["configuration"])
        print("[validate_model] -- END PRINT CONFIG")
    if iop:
        summary["input_options"] = str(iop)
    if mop:
        summary["model_options"] = str(mop)
    if "ERR_create" in summary:
        return summary, data

    if drop_inputs:
        if verbose:
            print(f"[validate_model] -- drop inputs: {drop_inputs!r}")
            print(f"[validate_model] current inputs: {string_type(data['inputs'])}")
            print(
                f"[validate_model] current dynnamic_shapes: "
                f"{string_type(data['dynamic_shapes'])}"
            )
        data["inputs"], data["dynamic_shapes"] = filter_inputs(
            data["inputs"],
            drop_names=drop_inputs,
            model=data["model"],
            dynamic_shapes=data["dynamic_shapes"],
        )
        if verbose:
            print(f"[validate_model] new inputs: {string_type(data['inputs'])}")
            print(f"[validate_model] new dynamic_hapes: {string_type(data['dynamic_shapes'])}")
        if second_input_keys:
            for k in second_input_keys:
                data[k], _ = filter_inputs(
                    data[k],
                    drop_names=drop_inputs,
                    model=data["model"],
                    dynamic_shapes=data["dynamic_shapes"],
                )

    if not empty(dtype):
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        if verbose:
            print(f"[validate_model] dtype conversion to {dtype}")
        data["model"] = to_any(data["model"], dtype)  # type: ignore
        data["inputs"] = to_any(data["inputs"], dtype)  # type: ignore
        summary["model_dtype"] = str(dtype)
        if second_input_keys:
            for k in second_input_keys:
                data[k] = to_any(data[k], dtype)  # type: ignore

    if not empty(device):
        if verbose:
            print(f"[validate_model] device conversion to {device}")
        data["model"] = to_any(data["model"], device)  # type: ignore
        data["inputs"] = to_any(data["inputs"], device)  # type: ignore
        summary["model_device"] = str(device)
        if second_input_keys:
            for k in second_input_keys:
                data[k] = to_any(data[k], device)  # type: ignore

    for k in ["task", "size", "n_weights"]:
        summary[f"model_{k.replace('_','')}"] = data[k]
    summary["second_input_keys"] = ",".join(second_input_keys)
    summary["model_inputs_options"] = str(input_options or "")
    summary["model_inputs"] = string_type(data["inputs"], with_shape=True)
    summary["model_shapes"] = string_type(data["dynamic_shapes"])
    summary["model_class"] = data["model"].__class__.__name__
    summary["model_module"] = str(data["model"].__class__.__module__)
    if summary["model_module"] in sys.modules:
        summary["model_file"] = str(sys.modules[summary["model_module"]].__file__)  # type: ignore[index]
    summary["model_config_class"] = data["configuration"].__class__.__name__
    summary["model_config"] = str(
        shrink_config(
            data["configuration"]
            if type(data["configuration"]) is dict
            else data["configuration"].to_dict()
        )
    ).replace(" ", "")
    summary["model_id"] = model_id

    if verbose:
        print("[validate_model] --")
        print(f"[validate_model] task={data['task']}")
        print(f"[validate_model] size={data['size'] / 2**20} Mb")
        print(f"[validate_model] n_weights={data['n_weights'] / 1e6} millions parameters")
        for k, v in data["inputs"].items():
            print(f"[validate_model] +INPUT {k}={string_type(v, with_shape=True)}")
        for k, v in data["dynamic_shapes"].items():
            print(f"[validate_model] +SHAPE {k}={string_type(v)}")
        print(f"[validate_model] second_input_keys={second_input_keys}")
        print("[validate_model] --")

    if do_run:
        validation_begin = time.perf_counter()

        _validate_do_run_model(
            data, summary, "inputs", "run", "run_expected", verbose, repeat, warmup, quiet
        )
        if second_input_keys:
            for k in second_input_keys:
                _validate_do_run_model(
                    data,
                    summary,
                    k,
                    f"run2{k[6:]}",
                    f"run_expected2{k[6:]}",
                    verbose,
                    1,
                    0,
                    quiet,
                )

        summary["time_total_validation_torch"] = time.perf_counter() - validation_begin

    if exporter:
        print(
            f"[validate_model] -- export the model with {exporter!r}, "
            f"optimization={optimization!r}"
        )
        exporter_begin = time.perf_counter()
        if patch_kwargs:
            if verbose:
                print(
                    f"[validate_model] applies patches before exporting "
                    f"stop_if_static={stop_if_static}"
                )
            with torch_export_patches(  # type: ignore
                stop_if_static=stop_if_static,
                verbose=max(0, verbose - 1),
                rewrite=data.get("rewrite", None),
                dump_rewriting=(os.path.join(dump_folder, "rewrite") if dump_folder else None),
                **patch_kwargs,  # type: ignore[arg-type]
            ) as modificator:
                data["inputs_export"] = modificator(data["inputs"])  # type: ignore

                if do_run:
                    _validate_do_run_exported_program(data, summary, verbose, quiet)

                # data is modified inplace
                summary_export, data = call_exporter(
                    exporter=exporter,
                    data=data,
                    quiet=quiet,
                    verbose=verbose,
                    optimization=optimization,
                    do_run=do_run,
                    dump_folder=dump_folder,
                    output_names=output_names,
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
                dump_folder=dump_folder,
                output_names=output_names,
            )

        summary.update(summary_export)
        summary["time_total_exporter"] = time.perf_counter() - exporter_begin

    dump_stats = None
    if dump_folder:
        if "exported_program" in data:
            ep = data["exported_program"]
            if verbose:
                print(f"[validate_model] -- dumps exported program in {dump_folder!r}...")
            with open(os.path.join(dump_folder, f"{folder_name}.ep"), "w") as f:
                f.write(str(ep))
            torch.export.save(ep, os.path.join(dump_folder, f"{folder_name}.pt2"))
            with open(os.path.join(dump_folder, f"{folder_name}.graph"), "w") as f:
                f.write(str(ep.graph))
            if verbose:
                print("[validate_model] done (dump ep)")
        if "onnx_program" in data:
            epo = data["onnx_program"]
            if verbose:
                print(f"[validate_model] dumps onnx program in {dump_folder!r}...")
            onnx_filename = os.path.join(dump_folder, f"{folder_name}.onnx")
            begin = time.perf_counter()
            if isinstance(epo, onnx.model_container.ModelContainer):
                epo.save(onnx_filename, all_tensors_to_one_file=True)
            elif isinstance(epo, onnx.ModelProto):
                if os.path.exists(f"{onnx_filename}.data"):
                    os.remove(f"{onnx_filename}.data")
                onnx.save(
                    epo,
                    onnx_filename,
                    save_as_external_data=True,
                    all_tensors_to_one_file=True,
                    location=f"{os.path.split(onnx_filename)[-1]}.data",
                )
            else:
                epo.save(onnx_filename, external_data=True)
            duration = time.perf_counter() - begin
            if verbose:
                print(f"[validate_model] done (dump onnx) in {duration}")
            data["onnx_filename"] = onnx_filename
            summary["time_onnx_save"] = duration
            summary.update(compute_statistics(onnx_filename))
            del epo

        if verbose:
            print(f"[validate_model] dumps statistics in {dump_folder!r}...")
        dump_stats = os.path.join(dump_folder, f"{folder_name}.stats")
        with open(dump_stats, "w") as f:
            for k, v in sorted(summary.items()):
                f.write(f":{k}:{v};\n")
        if verbose:
            print("[validate_model] done (dump)")

    if not exporter or (
        not exporter.startswith(("onnx-", "custom-"))
        and exporter not in ("custom", "modelbuilder")
    ):
        if verbose:
            print("[validate_model] -- done (final)")
        if dump_stats:
            with open(dump_stats, "w") as f:
                for k, v in sorted(summary.items()):
                    f.write(f":{k}:{v};\n")
        return summary, data

    if do_run:
        # Let's move the model to CPU to make sure it frees GPU memory.
        if verbose:
            # It does not really work for the time being and the model
            # gets loaded twice, one by torch, one by onnxruntime
            print("[validation_model] -- delete the model")
            for key in ["model", "onnx_program", "config"]:
                if key in data:
                    del data[key]
            if device is not None and "cuda" in str(device).lower():
                torch.cuda.empty_cache()
            gc.collect()
            print("[validation_model] -- done")

        validation_begin = time.perf_counter()
        summary_valid, data = validate_onnx_model(
            data=data,
            quiet=quiet,
            verbose=verbose,
            runtime=runtime,
            repeat=repeat,
            warmup=warmup,
            second_input_keys=second_input_keys,
            ort_logs=ort_logs,
        )
        summary.update(summary_valid)
        summary["time_total_validation_onnx"] = time.perf_counter() - validation_begin

    if ortfusiontype and "onnx_filename" in data:
        assert (
            "configuration" in data
        ), f"missing configuration in data, cannot run ort fusion for model_id={model_id}"
        config = data["configuration"]
        assert hasattr(
            config, "hidden_size"
        ), f"Missing attribute hidden_size in configuration {config}"
        hidden_size = config.hidden_size
        assert hasattr(
            config, "num_attention_heads"
        ), f"Missing attribute num_attention_heads in configuration {config}"
        num_attention_heads = config.num_attention_heads

        if ortfusiontype == "ALL":
            from onnxruntime.transformers.optimizer import MODEL_TYPES

            model_types = sorted(MODEL_TYPES)
        else:
            model_types = ortfusiontype.split("|")
        for model_type in model_types:
            flavour = f"ort{model_type}"
            summary[f"version_{flavour}_hidden_size"] = hidden_size
            summary[f"version_{flavour}_num_attention_heads"] = num_attention_heads

            begin = time.perf_counter()
            if verbose:
                print(f"[validate_model] run onnxruntime fusion for {model_type!r}")
            input_filename = data["onnx_filename"]
            output_path = f"{os.path.splitext(input_filename)[0]}.ort.{model_type}.onnx"
            ort_sum, ort_data = run_ort_fusion(
                input_filename,
                output_path,
                model_type=model_type,
                num_attention_heads=num_attention_heads,
                hidden_size=hidden_size,
            )
            summary.update(ort_sum)
            data.update(ort_data)
            data[f"onnx_filename_{flavour}"] = output_path
            duration = time.perf_counter() - begin
            summary[f"time_ortfusion_{flavour}"] = duration
            if verbose:
                print(
                    f"[validate_model] done {model_type!r} in {duration}, "
                    f"saved into {output_path!r}"
                )

            if do_run:
                summary_valid, data = validate_onnx_model(
                    data=data,
                    quiet=quiet,
                    verbose=verbose,
                    flavour=flavour,
                    runtime=runtime,
                    repeat=repeat,
                    warmup=warmup,
                    second_input_keys=second_input_keys,
                )
                summary.update(summary_valid)

    _compute_final_statistics(summary)
    summary["time_total"] = time.perf_counter() - main_validation_begin

    if verbose:
        print("[validate_model] -- done (final)")
    if dump_stats:
        # Dumps again the statistics.
        with open(dump_stats, "w") as f:
            for k, v in sorted(summary.items()):
                f.write(f":{k}:{v};\n")
    return summary, data


def compute_statistics(onnx_filename: str) -> Dict[str, Union[float, int]]:
    """Computes some statistics on the model itself."""
    onx = onnx.load(onnx_filename, load_external_data=False)
    cache_functions = {(f.domain, f.name): f for f in onx.functions}
    local_domains = set(f.domain for f in onx.functions)

    def node_iter(proto):
        if isinstance(proto, onnx.ModelProto):
            for f in proto.functions:
                yield from node_iter(f)
            yield from node_iter(proto.graph)
        elif isinstance(proto, (onnx.FunctionProto, onnx.GraphProto)):
            for node in proto.node:
                yield node

                # Let's inline the function
                key = node.domain, node.op_type
                if key in cache_functions:
                    yield from node_iter(cache_functions[key])

                # Let's continue
                for att in node.attribute:
                    if att.type == onnx.AttributeProto.GRAPH:
                        yield from node_iter(att.g)
            if hasattr(proto, "initializer"):
                yield from proto.initializer
        else:
            raise NotImplementedError(f"Unexpected type={type(proto)}")

    counts: Dict[str, Union[float, int]] = {}
    n_nodes = 0
    n_nodes_nocst = 0
    for proto in node_iter(onx):
        if isinstance(proto, onnx.NodeProto):
            key = f"n_node_{proto.op_type}"
            n_nodes += 1
            if proto.op_type != "Constant":
                n_nodes_nocst += 1
            if proto.domain in local_domains:
                key = "n_node_local_function"
                if key not in counts:
                    counts[key] = 0
                counts[key] += 1
        else:
            key = f"n_node_initializer_{proto.data_type}"

        if key not in counts:
            counts[key] = 0
        counts[key] += 1

    counts["n_node_nodes"] = n_nodes
    counts["n_node_nodes_nocst"] = n_nodes_nocst
    counts["n_node_functions"] = len(onx.functions)
    return counts


def _validate_do_run_model(
    data, summary, key, tag, expected_tag, verbose, repeat, warmup, quiet
):
    if verbose:
        print(f"[validate_model] -- run the model inputs={key!r}...")
        print(f"[validate_model] {key}={string_type(data[key], with_shape=True)}")
    # We make a copy of the input just in case the model modifies them inplace
    hash_inputs = string_type(data[key], with_shape=True)
    inputs = torch_deepcopy(data[key])
    model = data["model"]

    expected = _quiet_or_not_quiet(
        quiet,
        tag,
        summary,
        data,
        (lambda m=model, inp=inputs: m(**torch_deepcopy(inp))),
        repeat=repeat,
        warmup=warmup,
    )
    if f"ERR_{tag}" in summary:
        return summary, data

    summary[expected_tag] = string_type(expected, with_shape=True)
    if verbose:
        print(f"[validate_model] done ([{tag}])")
    data[expected_tag] = expected
    assert hash_inputs == string_type(data[key], with_shape=True), (
        f"The model execution did modified the inputs:\n"
        f"before: {hash_inputs}\n"
        f" after: {string_type(data[key], with_shape=True)}"
    )


def _validate_do_run_exported_program(data, summary, verbose, quiet):

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
    model = data["model"]

    expected = _quiet_or_not_quiet(
        quiet,
        "run_patched",
        summary,
        data,
        (lambda m=model, inp=inputs: m(**inp)),
    )
    if "ERR_run_patched" in summary:
        return summary, data

    disc = max_diff(data["run_expected"], expected)
    for k, v in disc.items():
        summary[f"disc_patched_{k}"] = str(v)
    if verbose:
        print("[validate_model] done (patched run)")
        print(f"[validate_model] patched discrepancies={string_diff(disc)}")
    assert hash_inputs == string_type(data["inputs_export"], with_shape=True), (
        f"The model execution did modified the inputs:\n"
        f"before: {hash_inputs}\n"
        f" after: {string_type(data['inputs_export'], with_shape=True)}"
    )


_cache_export_times = []
_main_export_function = torch.export.export


def _torch_export_export(*args, _export=_main_export_function, **kwargs):
    begin = time.perf_counter()
    res = _export(*args, **kwargs)
    duration = time.perf_counter() - begin
    _cache_export_times.append(duration)
    return res


def _restore_torch_export_export(summary):
    torch.export.export = _main_export_function
    if _cache_export_times:
        summary["time_torch_export_export"] = sum(_cache_export_times)
        summary["time_torch_export_export_n"] = len(_cache_export_times)
    _cache_export_times.clear()


def call_exporter(
    data: Dict[str, Any],
    exporter: str,
    quiet: bool = False,
    verbose: int = 0,
    optimization: Optional[str] = None,
    do_run: bool = False,
    dump_folder: Optional[str] = None,
    output_names: Optional[List[str]] = None,
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
    :param dump_folder: to dump additional information
    :param output_names: list of output names to use with the onnx exporter
    :return: two dictionaries, one with some metrics,
        another one with whatever the function produces
    """
    _cache_export_times.clear()
    torch.export.export = _torch_export_export

    if exporter == "export" or exporter.startswith("export-"):
        # torch export
        summary, data = call_torch_export_export(
            exporter=exporter,
            data=data,
            quiet=quiet,
            verbose=verbose,
            optimization=optimization,
            do_run=do_run,
        )
        _restore_torch_export_export(summary)
        return summary, data
    if exporter.startswith("onnx-"):
        # torch export
        summary, data = call_torch_export_onnx(
            exporter=exporter,
            data=data,
            quiet=quiet,
            verbose=verbose,
            optimization=optimization,
            output_names=output_names,
        )
        _restore_torch_export_export(summary)
        return summary, data
    if exporter == "custom" or exporter.startswith("custom"):
        # torch export
        summary, data = call_torch_export_custom(
            exporter=exporter,
            data=data,
            quiet=quiet,
            verbose=verbose,
            optimization=optimization,
            dump_folder=dump_folder,
            output_names=output_names,
        )
        _restore_torch_export_export(summary)
        return summary, data
    if exporter == "modelbuilder":
        # torch export
        summary, data = call_torch_export_model_builder(
            exporter=exporter,
            data=data,
            quiet=quiet,
            verbose=verbose,
            optimization=optimization,
            output_names=output_names,
        )
        _restore_torch_export_export(summary)
        return summary, data
    raise NotImplementedError(
        f"export with {exporter!r} and optimization={optimization!r} not implemented yet, "
        f"exporter must startswith 'onnx-', 'custom', 'export', 'modelbuilder' "
        f"(onnx-dynamo, custom, export), optimization can 'ir', "
        f"'default', 'default+onnxruntime', "
        f"'default+onnxruntime+os_ort', 'ir', 'os_ort'"
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
        "export",
        "export-strict",
        "export-nostrict",
    }, f"Unexpected value for exporter={exporter!r}"
    assert not optimization, f"No optimization is implemented for exporter={exporter!r}"
    assert "model" in data, f"model is missing from data: {sorted(data)}"
    assert "inputs_export" in data, f"inputs_export is missing from data: {sorted(data)}"
    summary: Dict[str, Union[str, int, float]] = {}
    strict = "-strict" in exporter
    args, kwargs = split_args_kwargs(data["inputs_export"])
    ds = data.get("dynamic_shapes", None)

    summary["export_exporter"] = exporter
    summary["export_optimization"] = optimization or ""
    summary["export_strict"] = strict
    summary["export_args"] = string_type(args, with_shape=True)
    summary["export_kwargs"] = string_type(kwargs, with_shape=True)
    summary["export_dynamic_shapes"] = string_type(ds)

    # There is an issue with DynamicShape [[],[]] becomes []
    dse = use_dyn_not_str(ds)
    # dse = CoupleInputsDynamicShapes(args, kwargs, ds).replace_string_by()

    summary["export_dynamic_shapes_export_export"] = string_type(dse)

    if verbose:
        print(
            f"[call_torch_export_export] exporter={exporter!r}, "
            f"strict={strict}, optimization={optimization!r}"
        )
        print(f"[call_torch_export_export] args={string_type(args, with_shape=True)}")
        print(f"[call_torch_export_export] kwargs={string_type(kwargs, with_shape=True)}")
        print(f"[call_torch_export_export] dynamic_shapes={string_type(ds)}")
        print(f"[call_torch_export_export] dynamic_shapes_export_export={string_type(dse)}")
        print("[call_torch_export_export] export...")

    model = data["model"]
    ep = _quiet_or_not_quiet(
        quiet,
        "export_export",
        summary,
        data,
        (
            lambda m=model, args=args, kws=kwargs, dse=dse, s=strict: (
                torch.export.export(m, args, kwargs=kws, dynamic_shapes=dse, strict=s)
            )
        ),
    )
    if "ERR_export_export" in summary:
        return summary, data

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

        expected = _quiet_or_not_quiet(
            quiet,
            "run_exported",
            summary,
            data,
            (lambda m=model, inputs=inputs: (model(**inputs))),
        )
        if "ERR_export_export" in summary:
            return summary, data

        disc = max_diff(data["run_expected"], expected)
        for k, v in disc.items():
            summary[f"disc_exported_{k}"] = str(v)
        if verbose:
            print("[validate_model] done (exported run)")
            print(f"[validate_model] exported discrepancies={string_diff(disc)}")
        assert hash_inputs == string_type(data["inputs_export"], with_shape=True), (
            f"The exported model execution did modified the inputs:\n"
            f"before: {hash_inputs}\n"
            f" after: {string_type(data['inputs_export'], with_shape=True)}"
        )
    return summary, data


def validate_onnx_model(
    data: Dict[str, Any],
    quiet: bool = False,
    verbose: int = 0,
    flavour: Optional[str] = None,
    runtime: str = "onnxruntime",
    repeat: int = 1,
    warmup: int = 0,
    second_input_keys: Optional[List[str]] = None,
    ort_logs: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Verifies that an onnx model produces the same
    expected outputs. It uses ``data["onnx_filename]`` as the input
    onnx filename or ``data["onnx_filename_{flavour}]`` if *flavour*
    is specified.

    :param data: dictionary with all the necessary inputs, the dictionary must
        contains keys ``model`` and ``inputs_export``
    :param quiet: catch exception or not
    :param verbose: verbosity
    :param flavour: use a different version of the inputs
    :param runtime: onnx runtime to use, onnxruntime, torch, orteval, ref
    :param repeat: run that number of times the model
    :param warmup: warmup the model
    :param second_input_keys: to validate the model on other input sets
        to make sure the exported model supports dynamism, the value is
        used as an increment added to the first set of inputs (added to dimensions)
    :param ort_logs: triggers the logs for onnxruntime
    :return: two dictionaries, one with some metrics,
        another one with whatever the function produces
    """
    import onnxruntime

    def _mk(key, flavour=flavour):
        return f"{key}_{flavour}" if flavour else key

    summary: Dict[str, Any] = {}
    flat_inputs = flatten_object(data["inputs"], drop_keys=True)
    d = flat_inputs[0].get_device()
    providers = (
        ["CPUExecutionProvider"]
        if d < 0
        else ["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    input_data_key = f"onnx_filename_{flavour}" if flavour else "onnx_filename"

    if input_data_key in data:
        source = data[input_data_key]
        if not os.path.exists(source):
            if verbose:
                print(f"[validate_onnx_model] missing {source!r}")
            summary[_mk("ERR_onnx_missing")] = f"FileNotFoundError({source!r})"
            return summary, data
        summary[input_data_key] = source
        summary[_mk("onnx_size")] = os.stat(source).st_size
    else:
        assert not flavour, f"flavour={flavour!r}, the filename must be saved."
        assert (
            "onnx_program" in data
        ), f"onnx_program is missing from data which has {sorted(data)}"
        source = data["onnx_program"].model_proto.SerializeToString()
        assert len(source) < 2**31, f"The model is highger than 2Gb: {len(source) / 2**30} Gb"
        summary[_mk("onnx_size")] = len(source)
    if verbose:
        print(
            f"[validate_onnx_model] verify onnx model with providers "
            f"{providers}..., flavour={flavour!r}"
        )

    if runtime == "onnxruntime":
        if os.environ.get("DUMPORTOPT", "") in ("1", "true", "True"):
            opts = onnxruntime.SessionOptions()
            opts.optimized_model_filepath = f"{data['onnx_filename']}.rtopt.onnx"
            if verbose:
                print(
                    f"[validate_onnx_model] saved optimized onnxruntime "
                    f"in {opts.optimized_model_filepath!r}"
                )
            onnxruntime.InferenceSession(data["onnx_filename"], opts, providers=providers)
            if verbose:
                print("[validate_onnx_model] -- done")

        if verbose:
            print("[validate_onnx_model] runtime is onnxruntime")
        sess_opts = onnxruntime.SessionOptions()
        if ort_logs:
            sess_opts.log_severity_level = 0
            sess_opts.log_verbosity_level = 4
        cls_runtime = lambda model, providers, _o=sess_opts: onnxruntime.InferenceSession(
            (model.SerializeToString() if isinstance(model, onnx.ModelProto) else model),
            _o,
            providers=providers,
        )
    elif runtime == "torch":
        from ..reference import TorchOnnxEvaluator

        if verbose:
            print("[validate_onnx_model] runtime is TorchOnnxEvaluator")
        cls_runtime = (
            lambda model, providers, _cls_=TorchOnnxEvaluator: _cls_(  # type: ignore[misc]
                model, providers=providers, verbose=max(verbose - 1, 0)
            )
        )
    elif runtime == "orteval":
        from ..reference import OnnxruntimeEvaluator

        if verbose:
            print("[validate_onnx_model] runtime is OnnxruntimeEvaluator")
        cls_runtime = (
            lambda model, providers, _cls_=OnnxruntimeEvaluator: _cls_(  # type: ignore[misc]
                model, providers=providers, verbose=max(verbose - 1, 0)
            )
        )
    elif runtime == "orteval10":
        from ..reference import OnnxruntimeEvaluator

        if verbose:
            print("[validate_onnx_model] runtime is OnnxruntimeEvaluator(verbose=10)")
        cls_runtime = (
            lambda model, providers, _cls_=OnnxruntimeEvaluator: _cls_(  # type: ignore[misc]
                model, providers=providers, verbose=10
            )
        )
    elif runtime == "ref":
        from ..reference import ExtendedReferenceEvaluator

        if verbose:
            print("[validate_onnx_model] runtime is ExtendedReferenceEvaluator")
        cls_runtime = lambda model, providers, _cls_=ExtendedReferenceEvaluator: _cls_(  # type: ignore[misc]
            model, verbose=max(verbose - 1, 0)
        )
    else:
        raise ValueError(f"Unexpecteed runtime={runtime!r}")

    sess = _quiet_or_not_quiet(
        quiet,
        _mk("create_onnx_ort"),
        summary,
        data,
        (lambda source=source, providers=providers: cls_runtime(source, providers)),
    )
    if f"ERR_{_mk('onnx_ort_create')}" in summary:
        return summary, data

    data[_mk("onnx_ort_sess")] = sess
    if verbose:
        print(f"[validate_onnx_model] done (ort_session) flavour={flavour!r}")

    keys = [("inputs", "run_expected", "")]
    if second_input_keys:
        keys.extend([(k, f"run_expected2{k[6:]}", f"2{k[6:]}") for k in second_input_keys])
    for k_input, k_expected, suffix in keys:
        # make_feeds
        assert k_input in data, f"Unable to find {k_input!r} in {sorted(data)}"
        assert k_expected in data, f"Unable to find {k_expected!r} in {sorted(data)}"
        if verbose:
            print(f"[validate_onnx_model] -- make_feeds for {k_input!r}...")
            print(
                f"[validate_onnx_model] inputs={string_type(data[k_input], with_shape=True)}"
            )
        feeds = make_feeds(
            sess,
            data[k_input],
            use_numpy=True,
            check_flatten=False,
            is_modelbuilder=data["exporter"] == "modelbuilder",
        )
        if verbose:
            print(f"[validate_onnx_model] ort inputs={string_type(feeds, with_shape=True)}")
        summary[_mk(f"onnx_ort_inputs{suffix}")] = string_type(feeds, with_shape=True)
        if verbose:
            print("[validate_onnx_model] done (make_feeds)")

        # run ort
        if verbose:
            print("[validate_onnx_model] run session...")

        got = _quiet_or_not_quiet(
            quiet,
            _mk(f"run_onnx_ort{suffix}"),
            summary,
            data,
            (lambda sess=sess, feeds=feeds: sess.run(None, feeds)),
            repeat=repeat,
            warmup=warmup,
        )
        # NOTE: modelbuilder has different order on past_kv outputs
        if data["exporter"] == "modelbuilder":
            logits = got[:1]
            past_key_values = got[1:]
            reorder_past_key_values = reorder_modelbuilder_cache_to_torch(past_key_values)
            got = logits + reorder_past_key_values

        if f"ERR_{_mk(f'time_onnx_ort_run{suffix}')}" in summary:
            return summary, data

        summary[f"run_feeds_{k_input}"] = string_type(feeds, with_shape=True, with_device=True)
        summary[f"run_output_{k_input}"] = string_type(got, with_shape=True, with_device=True)
        if verbose:
            print("[validate_onnx_model] done (run)")
            print(f"[validate_onnx_model] got={string_type(got, with_shape=True)}")

        # compute discrepancies
        disc = max_diff(data[k_expected], got, flatten=True)
        if verbose:
            print(f"[validate_onnx_model] discrepancies={string_diff(disc)}")
        for k, v in disc.items():
            summary[_mk(f"disc_onnx_ort_run{suffix}_{k}")] = v
    return summary, data


def call_torch_export_onnx(
    data: Dict[str, Any],
    exporter: str,
    quiet: bool = False,
    verbose: int = 0,
    optimization: Optional[str] = None,
    output_names: Optional[List[str]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Exports a model into onnx.
    If a patch must be applied, it should be before this functions.

    :param data: dictionary with all the necessary inputs, the dictionary must
        contains keys ``model`` and ``inputs_export``
    :param exporter: exporter to call
    :param quiet: catch exception or not
    :param verbose: verbosity
    :param optimization: optimization to do
    :param output_names: output names to use
    :return: two dictionaries, one with some metrics,
        another one with whatever the function produces
    """
    available = {None, "", "ir", "os_ort", "ir+default"}
    assert (
        optimization in available
    ), f"unexpected value for optimization={optimization}, available={available}"
    assert exporter in {
        "onnx-dynamo",
        "onnx-script",
    }, f"Unexpected value for exporter={exporter!r}"
    assert "model" in data, f"model is missing from data: {sorted(data)}"
    assert "inputs_export" in data, f"inputs_export is missing from data: {sorted(data)}"
    summary: Dict[str, Union[str, int, float]] = {}
    dynamo = "dynamo" in exporter
    args, kwargs = split_args_kwargs(data["inputs_export"])
    ds = data.get("dynamic_shapes", None)
    if verbose:
        print(
            f"[call_torch_export_onnx] exporter={exporter!r}, "
            f"optimization={optimization!r}"
        )
        print(f"[call_torch_export_onnx] args={string_type(args, with_shape=True)}")
        print(f"[call_torch_export_onnx] kwargs={string_type(kwargs, with_shape=True)}")
        print(f"[call_torch_export_onnx] dynamic_shapes={string_type(ds)}")
        print("[call_torch_export_onnx] export...")
    summary["export_exporter"] = exporter
    summary["export_optimization"] = optimization or ""
    summary["export_dynamo"] = dynamo
    summary["export_args"] = string_type(args, with_shape=True)
    summary["export_kwargs"] = string_type(kwargs, with_shape=True)
    opset = data.get("model_opset", None)
    if opset:
        summary["export_opset"] = opset

    if dynamo:
        export_export_kwargs = dict(dynamo=True, dynamic_shapes=ds)
    else:
        export_export_kwargs = dict(
            dynamo=False,
            dynamic_axes={
                k: v
                for k, v in CoupleInputsDynamicShapes(args, kwargs, ds)  # type: ignore[arg-type]
                .replace_by_string()
                .items()
                if isinstance(v, dict)
            },
        )
        args = tuple(flatten_unflatten_for_dynamic_shapes(a) for a in args)
        kwargs = {k: flatten_unflatten_for_dynamic_shapes(v) for k, v in kwargs.items()}
        if verbose:
            print("[call_torch_export_onnx] dynamo=False so...")
            print(f"[call_torch_export_onnx] args={string_type(args, with_shape=True)}")
            print(f"[call_torch_export_onnx] kwargs={string_type(kwargs, with_shape=True)}")
    if output_names:
        export_export_kwargs["output_names"] = output_names
    if opset:
        export_export_kwargs["opset_version"] = opset
    if verbose:
        print(
            f"[call_torch_export_onnx] export_export_kwargs="
            f"{string_type(export_export_kwargs, with_shape=True)}"
        )
    model = data["model"]

    epo = _quiet_or_not_quiet(
        quiet,
        "export_onnx",
        summary,
        data,
        (
            lambda m=model, args=args, kws=kwargs, ekws=export_export_kwargs: (
                torch.onnx.export(
                    m,
                    args,
                    kwargs=kws,
                    **ekws,
                )
            )
        ),
    )
    if "ERR_export_onnx" in summary:
        return summary, data

    assert epo is not None, "no onnx export was found"
    if verbose:
        print("[call_torch_export_onnx] done (export)")
    data["onnx_program"] = epo
    if verbose > 5:
        print("[call_torch_export_onnx] -- ONNXProgram")
        print(epo)
        print("[call_torch_export_onnx] -- End of ONNXProgram")

    if optimization in {"ir", "os_ort", "ir+default"}:
        if verbose:
            print(f"[call_torch_export_onnx] starts optimization={optimization!r}...")
        if optimization == "ir":
            label, f_optim = "export_onnx_opt_ir", (lambda epo=epo: epo.optimize())
        elif optimization == "ir+default":
            import onnxscript
            from experimental_experiment.xbuilder import GraphBuilder, OptimizationOptions

            def _ir_default_opt(epo):
                onnxscript.optimizer.optimize_ir(epo.model)
                onx = epo.model_proto
                # not very efficient
                gr = GraphBuilder(
                    onx,
                    infer_shapes_options=True,
                    optimization_options=OptimizationOptions(patterns="default"),
                )
                cont = gr.to_onnx(large_model=True)
                epo.model = cont.to_ir()

            label, f_optim = "export_onnx_opt_ir_default", (
                lambda epo=epo: _ir_default_opt(epo)
            )

        else:
            import onnxscript
            import onnxscript.rewriter.ort_fusions as ort_fusions

            def _os_ort_optim(epo):
                onnxscript.optimizer.optimize_ir(epo.model)
                optimized = ort_fusions.optimize_for_ort(epo.model)
                if isinstance(optimized, tuple):
                    for k, v in optimized[1].items():
                        summary[f"op_opt_fused_{k}"] = v
                    epo.model = optimized[0]
                else:
                    epo.model = optimized

            label, f_optim = "export_onnx_opt_os_ort", (lambda epo=epo: _os_ort_optim(epo))
        _quiet_or_not_quiet(quiet, label, summary, data, f_optim)
        if "ERR_export_onnx_opt_ir" in summary:
            return summary, data
        if verbose:
            print("[call_torch_export_onnx] done (optimization)")

    return summary, data


def call_torch_export_model_builder(
    data: Dict[str, Any],
    exporter: str,
    quiet: bool = False,
    verbose: int = 0,
    optimization: Optional[str] = None,
    output_names: Optional[List[str]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Exports a model into onnx with :epkg:`ModelBuilder`.

    :param data: dictionary with all the necessary inputs, the dictionary must
        contains keys ``model`` and ``inputs_export``
    :param exporter: exporter to call
    :param quiet: catch exception or not
    :param verbose: verbosity
    :param optimization: optimization to do
    :param output_names: list of output names to use
    :return: two dictionaries, one with some metrics,
        another one with whatever the function produces
    """
    from ..helpers.model_builder_helper import create_model_builder, save_model_builder

    assert optimization in (
        None,
        "",
    ), f"unexpected value for optimization={optimization}, none is available"
    precision = data.get("model_dtype", "fp32")
    provider = data.get("model_device", "cpu")
    dump_folder = data.get("model_dump_folder", "")
    assert dump_folder, "dump_folder cannot be empty with ModelBuilder"
    assert (
        not output_names
    ), f"output_names not empty, not supported yet, output_names={output_names}"
    cache_dir = os.path.join(dump_folder, "cache_mb")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    summary: Dict[str, Any] = {}

    epo = _quiet_or_not_quiet(
        quiet,
        "export_model_builder",
        summary,
        data,
        (
            lambda m=data["model"], c=data[
                "configuration"
            ], p=precision, pr=provider, cd=cache_dir: (
                save_model_builder(
                    create_model_builder(
                        c, m, precision=p, execution_provider=pr, cache_dir=cd
                    )
                )
            )
        ),
    )
    if "ERR_export_model_builder" in summary:
        return summary, data

    assert epo is not None, "no onnx export was found"
    if verbose:
        print("[call_torch_export_model_builder] done (export)")
    data["onnx_program"] = epo
    return summary, data


def process_statistics(data: Sequence[Dict[str, float]]) -> Dict[str, Any]:
    """
    Processes statistics coming from the exporters.
    It takes a sequence of dictionaries (like a data frame)
    and extracts some metrics.
    """

    def _simplify(p):
        for s in [
            "remove_unused",
            "constant_folding",
            "remove_identity",
            "remove_duplicated_initializer",
            "remove_duplicated_shape",
            "dynamic_dimension_naming",
            "inline",
            "check",
            "build_graph_for_pattern",
            "pattern_optimization",
            "topological_sort",
        ]:
            if s in p or s.replace("_", "-") in p:
                return s
        if p.startswith(("apply_", "match_")):
            return p
        return "other"

    def _add(d, a, v, use_max=False):
        if v:
            if a not in d:
                d[a] = v
            elif use_max:
                d[a] = max(d[a], v)
            else:
                d[a] += v

    counts: Dict[str, Any] = {}
    applied_pattern_time: Dict[str, Any] = {}
    applied_pattern_n: Dict[str, Any] = {}
    matching_pattern_time: Dict[str, Any] = {}
    matching_pattern_n: Dict[str, Any] = {}

    for obs in data:
        pattern = _simplify(obs["pattern"])
        _add(counts, "opt_nodes_added", obs.get("added", 0))
        _add(counts, "opt_nodes_removed", obs.get("removed", 0))
        _add(counts, "opt_time_steps", obs.get("time_in", 0))
        _add(counts, "opt_n_steps", 1)
        _add(
            counts,
            "opt_n_iteration",
            max(counts.get("opt_n_iteration", 0), obs.get("iteration", 0)),
            use_max=True,
        )

        if pattern.startswith("apply_"):
            _add(counts, "opt_n_applied_patterns", 1)
            _add(counts, "opt_time_applied_patterns", obs.get("time_in", 0))
            _add(applied_pattern_time, pattern, obs.get("time_in", 0))
            _add(applied_pattern_n, pattern, 1)
        elif pattern.startswith("match_"):
            _add(counts, "opt_n_matching_patterns", 1)
            _add(counts, "opt_time_matching_patterns", obs.get("time_in", 0))
            _add(matching_pattern_time, pattern, obs.get("time_in", 0))
            _add(matching_pattern_n, pattern, 1)
        else:
            _add(counts, f"opt_time_{pattern}", obs.get("time_in", 0))
            _add(counts, f"opt_n_{pattern}", 1)
            _add(counts, f"opt_nodes_added_{pattern}", obs.get("added", 0))
            _add(counts, f"opt_nodes_removed_{pattern}", obs.get("removed", 0))

    if applied_pattern_time:
        longest = max((v, k) for k, v in applied_pattern_time.items())
        counts["opt_top_time_applied_pattern"], counts["opt_top_time_applied_pattern_arg"] = (
            longest
        )
        longest = max((v, k) for k, v in applied_pattern_n.items())
        counts["opt_top_n_applied_pattern"], counts["opt_top_n_applied_pattern_arg"] = longest

    if matching_pattern_time:
        longest = max((v, k) for k, v in matching_pattern_time.items())
        (
            counts["opt_top_time_matching_pattern"],
            counts["opt_top_time_matching_pattern_arg"],
        ) = longest
        longest = max((v, k) for k, v in matching_pattern_n.items())
        counts["opt_top_n_matching_pattern"], counts["opt_top_n_matching_pattern_arg"] = (
            longest
        )
        counts["onnx_opt_optimized"] = 1
    return counts


def call_torch_export_custom(
    data: Dict[str, Any],
    exporter: str,
    quiet: bool = False,
    verbose: int = 0,
    optimization: Optional[str] = None,
    dump_folder: Optional[str] = None,
    output_names: Optional[List[str]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Exports a model into onnx.
    If a patch must be applied, it should be before this functions.

    :param data: dictionary with all the necessary inputs, the dictionary must
        contains keys ``model`` and ``inputs_export``
    :param exporter: exporter to call
    :param quiet: catch exception or not
    :param verbose: verbosity
    :param optimization: optimization to do
    :param dump_folder: to store additional information
    :param output_names: list of output names to use
    :return: two dictionaries, one with some metrics,
        another one with whatever the function produces
    """
    available = {
        "",
        "default",
        "default+onnxruntime",
        "default+os_ort",
        "default+onnxruntime+os_ort",
        None,
    }
    if optimization == "none":
        optimization = ""
    assert (
        optimization in available
    ), f"unexpected value for optimization={optimization}, available={available}"
    available = {
        "custom",
        "custom-strict",
        "custom-strict-default",
        "custom-strict-all",
        "custom-nostrict",
        "custom-nostrict-default",
        "custom-nostrict-all",
        "custom-noinline",
        "custom-strict-noinline",
        "custom-strict-default-noinline",
        "custom-strict-all-noinline",
        "custom-nostrict-noinline",
        "custom-nostrict-default-noinline",
        "custom-nostrict-all-noinline",
        "custom-dec",
        "custom-decall",
    }
    assert exporter in available, f"Unexpected value for exporter={exporter!r} in {available}"
    assert "model" in data, f"model is missing from data: {sorted(data)}"
    assert "inputs_export" in data, f"inputs_export is missing from data: {sorted(data)}"
    summary: Dict[str, Union[str, int, float]] = {}
    strict = "-strict" in exporter
    args, kwargs = split_args_kwargs(data["inputs_export"])
    ds = data.get("dynamic_shapes", None)
    opset = data.get("model_opset", None)
    if opset:
        summary["export_opset"] = opset
    if verbose:
        print(
            f"[call_torch_export_custom] exporter={exporter!r}, "
            f"optimization={optimization!r}"
        )
        print(f"[call_torch_export_custom] args={string_type(args, with_shape=True)}")
        print(f"[call_torch_export_custom] kwargs={string_type(kwargs, with_shape=True)}")
        print(f"[call_torch_export_custom] dynamic_shapes={string_type(ds)}")
        print("[call_torch_export_custom] export...")
    summary["export_exporter"] = exporter
    summary["export_optimization"] = optimization or ""
    summary["export_strict"] = strict
    summary["export_args"] = string_type(args, with_shape=True)
    summary["export_kwargs"] = string_type(kwargs, with_shape=True)

    from experimental_experiment.torch_interpreter import to_onnx, ExportOptions
    from experimental_experiment.xbuilder import OptimizationOptions

    spl = optimization.split("+") if optimization else []
    os_ort = "os_ort" in spl
    optimization = "+".join(_ for _ in spl if _ != "os_ort")

    export_options = ExportOptions(
        strict=strict,
        decomposition_table=(
            "default"
            if ("-default" in exporter or "-dec" in exporter)
            else ("all" if ("-all" in exporter or "-decall" in exporter) else None)
        ),
        save_ep=(os.path.join(dump_folder, f"{exporter}.ep") if dump_folder else None),
    )
    inline = "-noinline" not in exporter
    options = OptimizationOptions(patterns=optimization) if optimization else None
    model = data["model"]
    kws = dict(
        dynamic_shapes=ds,
        export_options=export_options,
        options=options,
        optimize=bool(optimization),
        large_model=True,
        return_optimize_report=True,
        verbose=max(verbose - 2, 0),
        inline=inline,
    )
    if opset:
        kws["target_opset"] = opset
    if output_names:
        kws["output_names"] = output_names

    epo, opt_stats = _quiet_or_not_quiet(
        quiet,
        "export_export_onnx_c",
        summary,
        data,
        (
            lambda m=model, args=args, kwargs=kwargs, kws=kws: (
                to_onnx(
                    model,
                    args,
                    kwargs=kwargs,
                    **kws,
                )
            )
        ),
    )
    if "ERR_export_onnx_c" in summary:
        return summary, data

    new_stat: Dict[str, Any] = {k: v for k, v in opt_stats.items() if k.startswith("time_")}
    new_stat.update({k[5:]: v for k, v in opt_stats.items() if k.startswith("stat_time_")})
    if "optimization" in opt_stats:
        new_stat.update(process_statistics(opt_stats["optimization"]))

    summary.update(new_stat)
    assert epo is not None, "no onnx export was found"
    if verbose:
        print("[call_torch_export_custom] done (export)")

    if os_ort:
        import onnxscript
        import onnxscript.rewriter.ort_fusions as ort_fusions

        if verbose:
            print("[call_torch_export_custom] conversion to IR...")
        begin = time.perf_counter()
        ir_model = epo.to_ir()
        duration = time.perf_counter() - begin
        summary["time_optim_to_ir"] = duration
        if verbose:
            print(f"[call_torch_export_custom] done in {duration}")
            print("[call_torch_export_custom] start optimization...")
        begin = time.perf_counter()
        onnxscript.optimizer.optimize_ir(ir_model)
        ir_optimized = ort_fusions.optimize_for_ort(ir_model)
        if isinstance(ir_optimized, tuple):
            report = ir_optimized[1]
            for k, v in report.items():
                summary[f"op_opt_fused_{k}"] = v
            ir_optimized = ir_optimized[0]
            epo.model = ir_optimized
        duration = time.perf_counter() - begin
        summary["time_optim_os_ort"] = duration
        if verbose:
            print(f"[call_torch_export_custom] done in {duration}")

    data["onnx_program"] = epo
    return summary, data


def run_ort_fusion(
    model_or_path: Union[str, onnx.ModelProto],
    output_path: str,
    num_attention_heads: int,
    hidden_size: int,
    model_type: str = "bert",
    verbose: int = 0,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Runs :epkg:`onnxruntime` fusion optimizer.

    :param model_or_path: path to the ModelProto or the ModelProto itself
    :param output_path: the model to save
    :param num_attention_heads: number of heads, usually ``config.num_attention_heads``
    :param hidden_size: hidden size, usually ``config.hidden_size``
    :param model_type: type of optimization, see below
    :param verbose: verbosity
    :return: two dictionaries, summary and data

    Supported values for ``model_type``:

    .. runpython::
        :showcode:

        import pprint
        from onnxruntime.transformers.optimizer import MODEL_TYPES

        pprint.pprint(sorted(MODEL_TYPES))
    """
    from onnxruntime.transformers.optimizer import optimize_by_fusion
    from onnxruntime.transformers.fusion_options import FusionOptions

    opts = FusionOptions(model_type)

    if isinstance(model_or_path, str):
        if verbose:
            print(f"[run_ort_fusion] loads {model_or_path!r}")
        onx = onnx.load(model_or_path)
    else:
        onx = model_or_path
    begin = time.perf_counter()
    n_nodes = len(onx.graph.node)
    if verbose:
        print(
            f"[run_ort_fusion] starts optimization for "
            f"model_type={model_type!r} with {n_nodes} nodes"
        )
    try:
        new_onx = optimize_by_fusion(
            onx,
            model_type=model_type,
            num_heads=num_attention_heads,
            hidden_size=hidden_size,
            optimization_options=opts,
        )
    except Exception as e:
        duration = time.perf_counter() - begin
        if verbose:
            print(f"[run_ort_fusion] failed in {duration} for model_type={model_type!r}")
        return {
            f"ERR_opt_ort_{model_type}": str(e),
            f"opt_ort_{model_type}_duration": duration,
        }, {}

    duration = time.perf_counter() - begin
    delta = len(new_onx.model.graph.node)
    if verbose:
        print(f"[run_ort_fusion] done in {duration} with {delta} nodes")
        print(f"[run_ort_fusion] save to {output_path!r}")
    begin = time.perf_counter()
    new_onx.save_model_to_file(output_path, use_external_data_format=True)
    d = time.perf_counter() - begin
    if verbose:
        print(f"[run_ort_fusion] done in {d}")
    return {
        f"opt_ort_{model_type}_n_nodes1": n_nodes,
        f"opt_ort_{model_type}_n_nodes2": delta,
        f"opt_ort_{model_type}_delta_node": delta - n_nodes,
        f"opt_ort_{model_type}_duration": duration,
        f"opt_ort_{model_type}_duration_save": d,
    }, {f"opt_ort_{model_type}": output_path}


def _compute_final_statistics(summary: Dict[str, Any]):
    """
    Updates inline the list of statistics. It adds:

    - speedup
    """
    stats = {}
    if (
        "time_run_latency" in summary
        and "time_run_onnx_ort_latency" in summary
        and summary["time_run_onnx_ort_latency"] > 0
    ):
        stats["stat_estimated_speedup_ort"] = (
            summary["time_run_latency"] / summary["time_run_onnx_ort_latency"]
        )
        stats["stat_estimated_speedup_ort_m98"] = (
            summary["time_run_latency_m98"] / summary["time_run_onnx_ort_latency_m98"]
        )
    summary.update(stats)
