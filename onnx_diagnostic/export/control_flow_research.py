from typing import Any, Callable, Union
import torch
from torch._C import DispatchKey

# from torch._higher_order_ops import BaseHOP
from torch._ops import HigherOrderOperator
from torch._functorch.utils import exposed_in
import torch.utils._pytree as pytree
from torch._higher_order_ops.utils import (
    check_input_alias_and_mutation_return_outputs,
    reenter_make_fx,
    unique_graph_id,
    validate_subgraph_args_types,
)
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode, track_tensor_tree
from torch.utils._python_dispatch import _get_current_dispatch_mode
from .control_flow_onnx import _loop_for_onnx_fn


class SimpleLoopForOp(HigherOrderOperator):
    def __init__(self):
        super().__init__("simple_loop_for")

    def __call__(self, n_iter, body_fn, operands):
        validate_subgraph_args_types(operands)
        return super().__call__(n_iter, body_fn, operands)

    def gen_schema(self, n_iter, body_fn, operands):
        from torch._higher_order_ops.schema import HopSchemaGenerator
        from torch._higher_order_ops.utils import materialize_as_graph

        body_gm: torch.fx.GraphModule = materialize_as_graph(  # type: ignore[annotation-unchecked]
            body_fn, (torch.tensor(0, dtype=torch.int64), *operands)
        )
        (
            _,
            _,
            _,
            body_mutated_inputs,
            body_outputs,
        ) = check_input_alias_and_mutation_return_outputs(body_gm)
        mutated_inputs = body_mutated_inputs

        schema_gen = HopSchemaGenerator(self)
        schema_gen.add_arg("n_iter", n_iter)
        schema_gen.add_arg("body_fn", body_gm)
        for idx, arg in enumerate(operands):
            schema_gen.add_arg(f"operand{idx}", arg, is_mutated=idx in mutated_inputs)

        for out in body_outputs:
            schema_gen.add_output(out)
        schema_gen.add_schema_tree_spec(n_iter, body_fn, operands)
        return schema_gen.gen_schema()


simple_loop_for_op = SimpleLoopForOp()


@exposed_in("torch")
def simple_loop_for(
    n_iter: Union[int, torch.Tensor],
    body_fn: Callable,
    operands: Union[tuple, list] = (),
) -> Any:
    if torch.compiler.is_dynamo_compiling():
        return simple_loop_for_op(n_iter, body_fn, (n_iter, *operands))

    if isinstance(n_iter, (bool, int, float)):
        return _loop_for_onnx_fn(body_fn, n_iter, None, *operands)

    def _validate_input(n_iter, body_fn, operands):
        assert isinstance(
            n_iter, (int, torch.Tensor, torch.SymInt)
        ), f"Expected pred to be bool or tensor, but got {n_iter}."
        assert (
            not isinstance(n_iter, torch.Tensor) or n_iter.numel() == 1
        ), f"Expected pred to be bool or single-element tensor, but got {n_iter}."
        assert callable(body_fn), "Expect both branches to be callable."
        assert isinstance(operands, (tuple, list)) and pytree.tree_all(
            lambda t: isinstance(t, torch.Tensor), operands
        ), (
            "Expect operands to be a tuple of possibly nested dict/list/tuple that only "
            f"consists of tensor leaves, but got {operands}."
        )

    _validate_input(n_iter, body_fn, operands)

    assert torch._dynamo.is_dynamo_supported(), "torch.cond requires dynamo support."

    def _loop_for_op_wrapper(*args, **kwargs):
        return simple_loop_for_op(*args, **kwargs)

    from torch._higher_order_ops.utils import setup_compilation_env

    with setup_compilation_env() as _backend:
        return _loop_for_op_wrapper(n_iter, body_fn, *operands)
        # return torch.compile(_loop_for_op_wrapper, backend=backend, fullgraph=True)(
        #    n_iter, body_fn, operands
        # )


def trace_loop_for(proxy_mode, func_overload, n_iter, body_fn, operands):
    assert isinstance(
        operands, (list, tuple)
    ), f"Cond operands must be a list or tuple of tensors and SymInts {operands}"

    body_graph = reenter_make_fx(body_fn)(n_iter, *operands)

    body_outs = []
    for node in body_graph.graph.nodes:
        if node.op == "output":
            body_outs.extend(node.args)

    # flat_body_outs = pytree.arg_tree_leaves(*body_outs)
    _i, body_name = unique_graph_id(proxy_mode, prefix="body_graph")
    proxy_mode.tracer.root.register_module(body_name, body_graph)
    args = (n_iter, body_graph, body_graph, operands)
    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, args)
    out_proxy = proxy_mode.tracer.create_proxy("call_function", func_overload, proxy_args, {})
    out = func_overload(n_iter, body_graph, operands)
    return track_tensor_tree(out, out_proxy, constant=None, tracer=proxy_mode.tracer)


@simple_loop_for_op.py_impl(DispatchKey.CompositeExplicitAutograd)
def loop_for_op_dense(n_iter, body_fn, operands):
    assert all(
        isinstance(o, (torch.Tensor, int)) for o in operands
    ), f"Dense implementation operands must be a list of tensors and ints {operands}"
    mode = _get_current_dispatch_mode()
    assert mode is None, "Mode should never be enabled for CPU/CUDA key"
    return _loop_for_onnx_fn(body_fn, n_iter, None, operands)


@simple_loop_for_op.py_impl(ProxyTorchDispatchMode)
def inner(mode, n_iter, body_fn, operands):
    return trace_loop_for(mode, simple_loop_for_op, n_iter, body_fn, operands)


simple_loop_for_op.fallthrough(torch._C.DispatchKey.AutogradCPU)
simple_loop_for_op.fallthrough(torch._C.DispatchKey.AutogradCUDA)
