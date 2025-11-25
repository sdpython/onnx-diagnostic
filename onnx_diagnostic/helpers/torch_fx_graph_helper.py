from python import Any, Dict, Optional, Tuple
import torch
from .helper import string_type


def validate_fx_tensor(
    node: torch.fx.Node, tensor: torch.Tensor, expected_shape: Tuple[Any, ...]
) -> None:
    """
    Validates the shape of tensor is expected.

    :param node: node
    :param tensor: tensor
    :param expected_shape: expected shape
    """
    assert len(tensor.shape) == len(expected_shape), (
        f"Shape mismatch, got {tensor.shape} expected {expected_shape}, "
        f"node.name={node.name!r}, node.target={getattr(node, 'target', None)}, "
        f"node.args={node.args}, node.kwargs={node.kwargs}, "
        f"node.meta={node.meta}"
    )
    for a, b in zip(tensor.shape, expected_shape):
        assert not isinstance(b, int) or a == b or {a, b} == {0, 1}, (
            f"Dimension mismatch, got {tensor.shape} expected {expected_shape}, "
            f"node.name={node.name!r}, node.target={getattr(node, 'target', None)}, "
            f"node.args={node.args}, node.kwargs={node.kwargs}, "
            f"node.meta={node.meta}"
        )


def validate_fx_outputs(node: torch.fx.Node, outputs: Tuple[Any, ...]) -> None:
    """
    Validates the outputs of a node using metadata stored in the node.

    :param node: node
    :param outputs: outputs
    """
    if "val" not in node.meta:
        return
    if isinstance(outputs, torch.Tensor):
        validate_fx_tensor(node, outputs, node.meta["val"].shape)
        return
    if isinstance(outputs, (tuple, list)):
        assert isinstance(node.meta["val"], (list, tuple)), (
            f"Unexpected type {string_type(node.meta['val'])} for node.meta['val'], "
            f"node.name={node.name!r}, node.target={getattr(node, 'target', None)}, "
            f"node.args={node.args}, node.kwargs={node.kwargs}, "
            f"node.meta={node.meta}"
        )
        assert len(outputs) == len(node.meta["val"]), (
            f"Length mismatch, got {len(outputs)} expected {len(node.meta['val'])}, "
            f"node.name={node.name!r}, node.target={getattr(node, 'target', None)}, "
            f"node.args={node.args}, node.kwargs={node.kwargs}, "
            f"node.meta={node.meta}"
        )
        for a, b in zip(outputs, node.meta["val"]):
            validate_fx_tensor(node, a, b.shape)
        return
    if isinstance(outputs, int):
        assert (
            isinstance(node.meta["val"], (torch.SymInt, torch.SymBool, torch.SymFloat))
            or outputs == node.meta["val"]
        ), (
            f"Int mismatch, got {outputs} expected {node.meta['val']}, "
            f"node.name={node.name!r}, node.target={getattr(node, 'target', None)}, "
            f"node.args={node.args}, node.kwargs={node.kwargs}, "
            f"node.meta={node.meta}"
        )
        return
    if outputs is None:
        assert node.meta["val"] is None, (
            f"None mismatch, got {outputs} expected {node.meta['val']}, "
            f"node.name={node.name!r}, node.target={getattr(node, 'target', None)}, "
            f"node.args={node.args}, node.kwargs={node.kwargs}, "
            f"node.meta={node.meta}"
        )
        return
    raise NotImplementedError(
        f"Validation for output type {type(outputs)} is not implemented, "
        f"node.name={node.name!r}, node.target={getattr(node, 'target', None)}, "
        f"node.args={node.args}, node.kwargs={node.kwargs}, "
        f"node.meta={node.meta}"
    )


def run_fx_node(
    node: torch.fx.Node, args: Tuple[Any, ...], kwargs: Optional[Dict[str, Any]] = None
) -> Tuple[Any, ...]:
    """
    Executes a node

    :param node: runs a node
    :param args: unnamed inputs to the node
    :param kwargs: named inputs to the node
    :return: results
    """
    if node.op == "output":
        assert len(args) == 1 and not kwargs, (
            f"Unexpected inputs: args={string_type(args, limit=20)} "
            f"kwargs={string_type(kwargs, limit=20)}"
        )
        return args
    if node.op == "call_function":
        assert callable(node.target), f"{node.target!r} not callable in node {node!r}"
        for a, ea in zip(args, node.args):
            if isinstance(a, torch.Tensor) and hasattr(ea, "meta") and "val" in ea.meta:
                ta = ea.meta["val"]
                assert (
                    isinstance(ta, torch.Tensor)
                    and len(a.shape) == len(ta.shape)
                    and a.dtype == ta.dtype
                ), (
                    f"Unable to run node {node!r}, target={node.target!r}, "
                    f"node.args={node.args!r}, node.kwargs={node.kwargs!r}, "
                    f"args={string_type(args, with_shape=True, with_device=True)}, "
                    f"kwargs={string_type(kwargs, with_shape=True, with_device=True)}"
                )
        try:
            outputs = node.target(*args, **(kwargs or {}))
        except RuntimeError as e:
            raise RuntimeError(
                f"Unable to run node {node!r}, target={node.target!r}, "
                f"args={string_type(args, with_shape=True, with_device=True)}, "
                f"kwargs={string_type(kwargs, with_shape=True, with_device=True)}"
            ) from e
        validate_fx_outputs(node, outputs)
        return outputs
    raise NotImplementedError(
        f"node.op={node.op!r} is not implemented, node.name={node.name!r}"
    )


def _pick_result(torch_results: Dict[str, Any], ref: Any) -> Any:
    "See :func:`prepare_args_kwargs`."
    if isinstance(ref, torch.fx.Node):
        return torch_results[ref.name]
    if isinstance(ref, list):
        return [_pick_result(torch_results, n) for n in ref]
    if isinstance(ref, tuple):
        return tuple(_pick_result(torch_results, n) for n in ref)
    if isinstance(ref, dict):
        return {k: _pick_result(torch_results, v) for k, v in ref.items()}
    if isinstance(ref, (bool, int, float, str, torch.device, torch.dtype)):
        return ref
    if ref is None:
        return None
    if isinstance(ref, torch.layout):
        return ref
    raise NotImplementedError(f"Unable to process args type {type(ref)}")


def prepare_args_kwargs(
    torch_results: Dict[str, Any], node: torch.fx.Node
) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    """
    Prepares args and kwargs before executing a fx node.

    :param torch_results: existing results
    :param node: node to execute
    :return: new args and kwargs
    """
    new_args = _pick_result(torch_results, node.args)
    new_kwargs = _pick_result(torch_results, node.kwargs)
    return new_args, new_kwargs
