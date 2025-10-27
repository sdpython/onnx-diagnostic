from typing import Any, Dict, List, Sequence, Optional, Tuple, Union
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
) -> Any:
    """Common API for exporters."""
    if exporter == "custom":
        from experimental_experiment.torch_interpreter import to_onnx as _to_onnx
        from experimental_experiment.xbuilder import OptimizationOptions

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
            options=OptimizationOptions(patterns="default+onnxruntime"),
        )
    if exporter == "onnx-dynamo":
        import onnxscript.rewriter.ort_fusions as ort_fusions

        assert (
            not output_dynamic_shapes
        ), f"output_dynamic_shapes not supported for exporter={exporter!r}"
        epo = torch.onnx.export(
            mod,
            args=args,
            kwargs=kwargs,
            input_names=input_names,
            output_names=output_names,
            opset_version=target_opset,
            dynamic_shapes=dynamic_shapes,
        )
        ort_fusions.optimize_for_ort(epo.model)
        epo.save(filename)
        return epo

    raise ValueError(f"Unknown exporter={exporter!r}")
