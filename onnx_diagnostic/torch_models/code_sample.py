import textwrap
import torch
from typing import Any, Dict, List, Optional, Union


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

        print(code_sample("arnir0/Tiny-LLM"))
    """
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
    pieces = [
        CODE_SAMPLES["imports"],
        CODE_SAMPLES["get_model_with_inputs"],
        textwrap.dedent(
            f"""
            model = get_model_with_inputs({model_args})
                        """
        ),
    ]
    code = "\n".join(pieces)
    try:
        import black
    except ImportError:
        # No black formatting.
        return code

    return black.format_str(code, mode=black.Mode())
