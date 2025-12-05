import os
import textwrap
import torch
from typing import Any, Dict, List, Optional, Tuple, Union
from ..helpers import flatten_object
from ..helpers.torch_helper import to_any
from .hghub.model_inputs import _preprocess_model_id
from .hghub import get_untrained_model_with_inputs
from .validate import filter_inputs, make_patch_kwargs


CODE_SAMPLES = {
    "imports": "from typing import Any\nimport torch",
    "get_model_with_inputs": textwrap.dedent(
        """
    def get_model_with_inputs(
        model_id:str,
        subfolder: str | None = None,
        dtype: str | torch.dtype | None = None,
        device: str | torch.device | None = None,
        same_as_pretrained: bool = False,
        use_pretrained: bool = False,
        input_options: dict[str, Any] | None = None,
        model_options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if use_pretrained:
            import transformers
            assert same_as_pretrained, (
                "same_as_pretrained must be True if use_pretrained is True"
            )
            # tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = transformers.AutoModel.from_pretrained(
                model_id,
                trust_remote_code=True,
                subfolder=subfolder,
                dtype=dtype,
                device=device,
            )
            data = {"model": model}
            assert not input_options, "Not implemented yet with input_options{input_options}"
            assert not model_options, "Not implemented yet with input_options{model_options}"
        else:
            from onnx_diagnostic.torch_models.hghub import get_untrained_model_with_inputs
            data = get_untrained_model_with_inputs(
                model_id,
                use_pretrained=use_pretrained,
                same_as_pretrained=same_as_pretrained,
                inputs_kwargs=input_options,
                model_kwargs=model_options,
                subfolder=subfolder,
                add_second_input=False,
            )
            if dtype:
                data["model"] = data["model"].to(
                    getattr(torch, dtype) if isinstance(dtype, str) else dtype
                )
            if device:
                data["model"] = data["model"].to(device)
        return data["model"]
    """
    ),
}


def make_code_for_inputs(inputs: Dict[str, torch.Tensor]) -> str:
    """
    Creates a code to generate random inputs.

    :param inputs: dictionary
    :return: code
    """
    codes = []
    for k, v in inputs.items():
        if isinstance(v, (int, bool, float)):
            code = f"{k}={v}"
        elif isinstance(v, torch.Tensor):
            shape = tuple(map(int, v.shape))
            if v.dtype in (torch.int32, torch.int64):
                code = f"{k}=torch.randint({v.max()}, size={shape}, dtype={v.dtype})"
            elif v.dtype in (torch.float32, torch.float16, torch.bfloat16):
                code = f"{k}=torch.rand({shape}, dtype={v.dtype})"
            else:
                raise ValueError(f"Unexpected dtype = {v.dtype} for k={k!r}")
        elif v.__class__.__name__ == "DynamicCache":
            obj = flatten_object(v)
            cc = [f"torch.rand({tuple(map(int,_.shape))}, dtype={_.dtype})" for _ in obj]
            va = [f"({a},{b})" for a, b in zip(cc[: len(cc) // 2], cc[len(cc) // 2 :])]
            va2 = ", ".join(va)
            code = f"{k}=make_dynamic_cache([{va2}])"
        else:
            raise ValueError(f"Unexpected type {type(v)} for k={k!r}")
        codes.append(code)
    st = ", ".join(codes)
    return f"dict({st})"


def make_export_code(
    exporter: str,
    optimization: Optional[str] = None,
    patch_kwargs: Optional[Dict[str, Any]] = None,
    stop_if_static: int = 0,
    dump_folder: Optional[str] = None,
    opset: Optional[int] = None,
    dynamic_shapes: Optional[Dict[str, Any]] = None,
    output_names: Optional[List[str]] = None,
    verbose: int = 0,
) -> Tuple[str, str]:
    args = [f"dynamic_shapes={dynamic_shapes}"]
    if output_names:
        args.append(f"output_names={output_names}")
    code = []
    imports = []
    if dump_folder:
        code.append(f"os.makedirs({dump_folder!r})")
        imports.append("import os")
        filename = os.path.join(dump_folder, "model.onnx")
    if exporter == "custom":
        if opset:
            args.append(f"target_opset={opset}")
        if optimization:
            args.append(f"options=OptimizationOptions(patterns={optimization!r})")
            args.append(f"large_model=True, filename={filename!r}")
        sargs = ", ".join(args)
        imports.extend(
            [
                "from experimental_experiment.torch_interpreter import to_onnx",
                "from experimental_experiment.xbuilder import OptimizationOptions",
            ]
        )
        code.extend([f"onx = to_onnx(model, inputs, {sargs})"])
    elif exporter == "onnx-dynamo":
        if opset:
            args.append(f"opset_version={opset}")
        sargs = ", ".join(args)
        code.extend([f"epo = torch.onnx.export(model, args=(), kwargs=inputs, {sargs})"])
        if optimization:
            imports.append("import onnxscript")
            code.extend(["onnxscript.optimizer.optimize_ir(epo.model)"])
            if "os_ort" in optimization:
                imports.append("import onnxscript.rewriter.ort_fusions as ort_fusions")
                code.extend(["ort_fusions.optimize_for_ort(epo.model)"])
        if dump_folder:
            code.extend([f"epo.save({filename!r})"])
    else:
        raise ValueError(f"Unexpected exporter {exporter!r}")
    if not patch_kwargs:
        return "\n".join(imports), "\n".join(code)

    imports.append("from onnx_diagnostic.torch_export_patches import torch_export_patches")
    if stop_if_static:
        patch_kwargs["stop_if_static"] = stop_if_static
    sargs = ", ".join(f"{k}={v}" for k, v in patch_kwargs.items())
    code = [f"with torch_export_patches({sargs}):", *["    " + _ for _ in code]]
    return "\n".join(imports), "\n".join(code)


def code_sample(
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
    input_options: Optional[Dict[str, Any]] = None,
    model_options: Optional[Dict[str, Any]] = None,
    subfolder: Optional[str] = None,
    opset: Optional[int] = None,
    runtime: str = "onnxruntime",
    output_names: Optional[List[str]] = None,
) -> str:
    """
    This generates a code to export a model with the proper settings.

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
    :param input_options: additional options to define the dummy inputs
        used to export
    :param model_options: additional options when creating the model such as
        ``num_hidden_layers`` or ``attn_implementation``
    :param subfolder: version or subfolders to uses when retrieving a model id
    :param opset: onnx opset to use for the conversion
    :param runtime: onnx runtime to use to check about discrepancies,
        possible values ``onnxruntime``, ``torch``, ``orteval``,
        ``orteval10``, ``ref`` only if `do_run` is true
    :param output_names: output names the onnx exporter should use
    :return: a code

    .. runpython::
        :showcode:

        from onnx_diagnostic.torch_models.code_sample import code_sample

        print(
            code_sample(
                "arnir0/Tiny-LLM",
                exporter="onnx-dynamo",
                optimization="ir",
                patch=True,
            )
        )
    """
    model_id, subfolder, same_as_pretrained, use_pretrained, submodule = _preprocess_model_id(
        model_id,
        subfolder,
        same_as_pretrained=same_as_pretrained,
        use_pretrained=use_pretrained,
    )
    patch_kwargs = make_patch_kwargs(patch=patch, rewrite=rewrite)

    iop = input_options or {}
    mop = model_options or {}
    data = get_untrained_model_with_inputs(
        model_id,
        verbose=verbose,
        task=task,
        use_pretrained=use_pretrained,
        same_as_pretrained=same_as_pretrained,
        inputs_kwargs=iop,
        model_kwargs=mop,
        subfolder=subfolder,
        add_second_input=False,
        submodule=submodule,
    )
    if drop_inputs:
        update = {}
        for k in data:
            if k.startswith("inputs"):
                update[k], ds = filter_inputs(
                    data[k],
                    drop_names=drop_inputs,
                    model=data["model"],
                    dynamic_shapes=data["dynamic_shapes"],
                )
        update["dynamic_shapes"] = ds
        data.update(update)

    update = {}
    for k in data:
        if k.startswith("inputs"):
            v = data[k]
            if dtype:
                update[k] = v = to_any(
                    v, getattr(torch, dtype) if isinstance(dtype, str) else dtype
                )
            if device:
                update[k] = v = to_any(v, device)
    if update:
        data.update(update)

    args = [f"{model_id!r}"]
    if subfolder:
        args.append(f"subfolder={subfolder!r}")
    if dtype:
        args.append(f"dtype={dtype!r}")
    if device:
        args.append(f"device={device!r}")
    if same_as_pretrained:
        args.append(f"same_as_pretrained={same_as_pretrained!r}")
    if use_pretrained:
        args.append(f"use_pretrained={use_pretrained!r}")
    if input_options:
        args.append(f"input_options={input_options!r}")
    if model_options:
        args.append(f"model_options={model_options!r}")
    model_args = ", ".join(args)
    imports, exporter_code = (
        make_export_code(
            exporter=exporter,
            patch_kwargs=patch_kwargs,
            verbose=verbose,
            optimization=optimization,
            stop_if_static=stop_if_static,
            dump_folder=dump_folder,
            opset=opset,
            dynamic_shapes=data["dynamic_shapes"],
        )
        if exporter is not None
        else ([], [])
    )
    input_code = make_code_for_inputs(data["inputs"])
    cache_import = (
        "from onnx_diagnostic.helpers.cache_helper import make_dynamic_cache"
        if "dynamic_cache" in input_code
        else ""
    )

    pieces = [
        CODE_SAMPLES["imports"],
        imports,
        cache_import,
        CODE_SAMPLES["get_model_with_inputs"],
        textwrap.dedent(
            f"""
            model = get_model_with_inputs({model_args})
                        """
        ),
        f"inputs = {input_code}",
        exporter_code,
    ]
    code = "\n".join(pieces)  # type: ignore[arg-type]
    try:
        import black
    except ImportError:
        # No black formatting.
        return code

    return black.format_str(code, mode=black.Mode())
