import contextlib
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
import torch
from torch._higher_order_ops.utils import (
    materialize_as_graph,
    check_input_alias_and_mutation_return_outputs,
    # _maybe_reenter_make_fx,
)

_TEST_EXPORT = False


@contextlib.contextmanager
def enable_code_export_control_flow():
    """Enables the code meant to be exported."""
    global _TEST_EXPORT
    old = _TEST_EXPORT
    _TEST_EXPORT = True
    try:
        yield
    finally:
        _TEST_EXPORT = old


def is_exporting() -> bool:
    """
    Returns :func:`torch.compiler.is_exporting` or
    :func:`torch.compiler.is_compiling`.
    Changes ``_TEST_EXPORT`` to make it trigger.
    """
    return _TEST_EXPORT or torch.compiler.is_exporting() or torch.compiler.is_compiling()


def _loop_for_fn(n_iter, body_fn, reduction_dim, args):
    """
    Python implementation of the loop.

    :param n_iter: number of iteration
    :param body_fn: function implementing the body
    :param reduction_dim: dimension used to reduce the list produced by the loop
    :param args: arguments to the loop body
    :return: results
    """
    res = []
    for i in torch.arange(n_iter, dtype=n_iter.dtype):
        r = body_fn(i, *args)
        if isinstance(r, tuple):
            assert not res or len(r) == len(res[-1]), (
                f"Unexpected number of results {len(r)} for function {body_fn}, "
                f"expected {len(res[-1])}"
            )
            res.append(r)
        else:
            assert isinstance(r, torch.Tensor), (
                f"Unexpected type {r} for function {body_fn}, "
                f"it must be a tuple or a Tensor."
            )
            assert not res or len(res[-1]) == 1, (
                f"Unexpected number of results {len(r)} for function {body_fn}, "
                f"expected {len(res[-1])}"
            )
            res.append((r,))

    if not res:
        return torch.empty(tuple(), dtype=torch.float32, device=args[0].device)
    if len(res) == 1:
        final = res[0]
    else:
        n_res = len(res[0])
        final = [
            torch.cat(
                [r[i] for r in res],
                dim=(
                    0 if reduction_dim is None or i >= len(reduction_dim) else reduction_dim[i]
                ),
            )
            for i in range(n_res)
        ]
    return tuple(final) if len(final) > 1 else final[0]


def make_custom_loop_for(
    n_iter: torch.Tensor,
    body_fn: Callable,
    reduction_dim: Optional[Sequence[int]],
    args: Sequence[torch.Tensor],
    body_gm: Optional[torch.fx.GraphModule] = None,
    body_mutated_inputs: Optional[List[Any]] = None,
    body_outputs: Optional[List[Any]] = None,
) -> Tuple[str, torch.library.CustomOpDef]:
    """
    Defines a custom operator for a loop in order to avoid
    :func:`torch.export.export` digging into it.
    It registers the custom op and a custom conversion
    to ONNX.

    :param n_iter: number of iterations defined by a tensor of no dimension
    :param body_fn: the loop body defined as a function
    :param reduction_dim: dimension used to concatenated the results
    :param args: list of tensors, input to the body
    :param body_gm: torch.fx.GraphModule equivalent to *body_gm*
    :param body_mutated_inputs: inputs to *body_gm*
    :param body_outputs: outputs to *body_gm*
    :return: a name and the custom op definition, the name
        is used to cache the custom op
    """
    assert body_gm is not None, "body_gm cannot be None"
    assert body_mutated_inputs is not None, "body_mutated_inputs cannot be None"
    assert body_outputs is not None, "body_outputs cannot be None"

    srank = "_".join("x".join(map(str, s.shape)) for s in body_outputs)
    sred = "x".join(map(str, reduction_dim)) if reduction_dim else ""
    full_name = (
        body_fn.__qualname__.replace("<locals>", "L")
        .replace("<lambda>", "l")
        .replace(".", "_")
    )
    name = f"loop_for_onnx_{full_name}_{srank}_{sred}"

    schema = "(str body_fn, Tensor n_iter, Tensor[] body_inputs) -> Tensor"
    if len(body_outputs) > 1:
        schema += "[]"
    custom_def = torch.library.CustomOpDef("onnx_higher_ops", "loop_for", schema, body_fn)
    custom_def.register_kernel("cpu")(body_fn)

    custom_def._abstract_fn = lambda _fn_id, *_args, _o=body_outputs: (
        tuple([torch.empty_like(s) for s in _o]) if len(_o) > 1 else torch.empty_like(_o[0])
    )
    return name, custom_def


def loop_for(
    n_iter: Union[torch.SymInt, torch.Tensor],
    body_fn: Callable[..., Tuple[torch.Tensor]],
    args: Sequence[torch.Tensor],
    reduction_dim: Optional[Sequence[int]] = None,
) -> Tuple[torch.Tensor, ...]:
    """
    High operators used to easily export a loop in ONNX.
    Does not fully work with :func:`torch.export.export`,
    it does replaces a custom op with a loop operator afterwards.
    Every iteration produces tensors, all of them are gathered
    into lists, all these lists are concatenated into tensors.

    :param n_iter: number of iterations, it can be fixed on
        variable, in that case it should a tensor with no dimension
    :param body_fn: function body, takes only tensors and returns
        only tensors, the first argument is the iteration number
        in a tensor with no dimension, all the others
        are not changed during the loop
    :param args: the available tensors at every loop
    :param reduction_dim: the loop aggregated the results into list,
        one of each output, each of them is concatenated into one
        tensor along one dimension, by default, it is the first
        dimension, but it can be defined otherwise
    """
    assert args, "The function should have at least one arg."
    assert (
        isinstance(n_iter, torch.Tensor)
        and n_iter.dtype == torch.int64
        and len(n_iter.shape) == 0
    ), f"Only a tensor for one int64 is allowed for n_iter but it equal to {n_iter}."
    if is_exporting():
        from torch.fx.experimental.proxy_tensor import _CURRENT_MAKE_FX_TRACER

        # tracer = _CURRENT_MAKE_FX_TRACER.fx_tracer
        root = _CURRENT_MAKE_FX_TRACER.fx_tracer.root
        # graph = _CURRENT_MAKE_FX_TRACER.fx_tracer.graph

        body_gm: torch.fx.GraphModule = materialize_as_graph(
            body_fn, (torch.tensor(0, dtype=torch.int64), *args)
        )
        (
            _1,
            _2,
            _3,
            body_mutated_inputs,
            body_outputs,
        ) = check_input_alias_and_mutation_return_outputs(body_gm)
        name, _custom_ops = make_custom_loop_for(
            n_iter,
            body_fn,
            reduction_dim,
            args,
            body_gm=body_gm,
            body_mutated_inputs=body_mutated_inputs,
            body_outputs=body_outputs,
        )
        root.register_module(name, body_gm)
        # body_graph = _maybe_reenter_make_fx(body_fn)(n_iter, *args)
        return torch.ops.onnx_higher_ops.loop_for(name, n_iter, args)

    return _loop_for_fn(n_iter, body_fn, reduction_dim, args)


"""
        proxy_mode.tracer.root.register_module(cond_graph_name, cond_graph)
        proxy_mode.tracer.root.register_module(body_graph_name, body_graph)

        args = (cond_graph, body_graph, carried_inputs, additional_inputs)

        proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, args)

        out_proxy = proxy_mode.tracer.create_proxy(
            "call_function", op, proxy_args, {}, name=op._name
        )

        out = op(
            cond_graph, body_graph, unspecialized_carried_inputs, additional_inputs
        )
        return track_tensor_tree(
            out, out_proxy, constant=None, tracer=proxy_mode.tracer
        )
"""
