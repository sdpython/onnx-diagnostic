from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import torch


def to_onnx(
    mod: Union["torch.nn.Module", "torch.fx.GraphModule"],  # noqa: F821
    args: Optional[Sequence["torch.Tensor"]] = None,  # noqa: F821
    kwargs: Optional[Dict[str, "torch.Tensor"]] = None,  # noqa: F821
    input_names: Optional[Sequence[str]] = None,
    target_opset: Optional[Union[int, Dict[str, int]]] = None,
    verbose: int = 0,
    dynamic_shapes: Optional[Union[Dict[str, Any], Tuple[Any]]] = None,
    filename: Optional[str] = None,
    output_names: Optional[List[str]] = None,
    output_dynamic_shapes: Optional[Union[Dict[str, Any], Tuple[Any]]] = None,
    exporter: str = "onnx-dynamo",
    exporter_kwargs: Optional[Dict[str, Any]] = None,
    save_ep: Optional[str] = None,
    optimize: bool = True,
    use_control_flow_dispatcher: bool = False,
) -> Any:
    """
    Common API for exporters. By default, the models are optimized to use the
    most efficient kernels implemented in :epkg:`onnxruntime`.

    :param mod: torch model
    :param args: unnamed arguments
    :param kwargs: named arguments
    :param input_names: input names for the onnx model (optional)
    :param target_opset: opset to target, if not specified, each converter
        keeps its default value
    :param verbose: verbosity level
    :param dynamic_shapes: dynamic shapes, usually a nested structure
        included a dictionary for each tensor
    :param filename: output filename
    :param output_names: to change the output of the onnx model
    :param output_dynamic_shapes: to overwrite the dynamic shapes names
    :param exporter: exporter to use (``onnx-dynamo``, ``modelbuilder``, ``custom``)
    :param exporter_kwargs: additional parameters sent to the exporter
    :param save_ep: saves the exported program
    :param optimize: optimizes the model
    :param use_control_flow_dispatcher: use the dispatcher created to supported
        custom loops (see :func:`onnx_diagnostic.export.control_flow.loop_for`)
    :return: the output of the selected exporter, usually a structure including
        an onnx model

    A simple example:

    .. code-block:: python

        to_onnx(
            model,
            kwargs=inputs,
            dynamic_shapes=ds,
            exporter=exporter,
            filename=filename,
        )
    """
    if exporter == "custom":
        from experimental_experiment.torch_interpreter import (
            to_onnx as _to_onnx,
            ExportOptions,
        )
        from experimental_experiment.xbuilder import OptimizationOptions

        if use_control_flow_dispatcher:
            from .control_flow import create_global_dispatcher

            dispatcher = create_global_dispatcher()

        options = None
        if exporter_kwargs is not None:
            options = exporter_kwargs.pop("options", None)
        if options is None:
            options = OptimizationOptions(patterns="default+onnxruntime")

        return _to_onnx(
            mod,
            args=args,
            kwargs=kwargs,
            input_names=input_names,
            output_names=output_names,
            target_opset=target_opset,
            verbose=verbose,
            filename=filename,
            dynamic_shapes=dynamic_shapes,
            large_model=True,
            output_dynamic_shapes=output_dynamic_shapes,
            export_options=ExportOptions(save_ep=save_ep),
            options=options,
            **(exporter_kwargs or {}),
            dispatcher=dispatcher if use_control_flow_dispatcher else None,
        )
    if exporter in ("dynamo", "onnx-dynamo"):
        import onnxscript.rewriter.ort_fusions as ort_fusions

        assert (
            not output_dynamic_shapes
        ), f"output_dynamic_shapes not supported for exporter={exporter!r}"
        epo = torch.onnx.export(
            mod,
            args=args or tuple(),
            kwargs=kwargs,
            input_names=input_names,
            output_names=output_names,
            opset_version=target_opset,
            dynamic_shapes=dynamic_shapes,
            dynamo=True,
            **(exporter_kwargs or {}),
        )
        if optimize:
            ort_fusions.optimize_for_ort(epo.model)
        if filename:
            epo.save(filename, external_data=True)
        if save_ep:
            if isinstance(save_ep, tuple):
                save_ep = save_ep[0]
            torch.export.save(epo.exported_program, f"{save_ep}.pt2")
        return epo

    if exporter == "modelbuilder":
        import os
        from ..helpers import flatten_object, string_type
        from ..helpers.model_builder_helper import create_model_builder, save_model_builder

        assert filename, f"filename must be specified for exporter={exporter!r}"
        assert (
            not output_dynamic_shapes
        ), f"output_dynamic_shapes not supported for exporter={exporter!r}"
        assert hasattr(mod, "config"), f"configuration is missing in model class {type(mod)}"
        assert not args, f"only kwargs can be defined with exporter={exporter!r}"
        assert list(kwargs) == ["input_ids", "attention_mask", "past_key_values"], (  # type: ignore[arg-type]
            f"Only a specified set of inputs is supported for exporter={exporter!r}, "
            f"but it is {list(kwargs)}"  # type: ignore[arg-type]
        )
        flat_inputs = flatten_object(kwargs, drop_keys=True)
        first = flat_inputs[0]
        first_float = [
            t
            for t in flat_inputs
            if t.dtype in {torch.float32, torch.double, torch.float16, torch.bfloat16}
        ]
        assert first_float, (
            f"Unable to find a float tensor in the inputs "
            f"{string_type(kwargs, with_shape=True)}"
        )
        onx = create_model_builder(
            mod.config,
            mod,
            precision=str(first_float[0].dtype).split(".")[-1],
            execution_provider="cuda" if first.is_cuda else "cpu",
            cache_dir=os.path.dirname(filename),
            **(exporter_kwargs or {}),
        )
        save_model_builder(onx, os.path.dirname(filename))
        return onx

    raise ValueError(f"Unknown exporter={exporter!r}")
