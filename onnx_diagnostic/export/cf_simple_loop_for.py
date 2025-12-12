import contextlib
from typing import Callable, List, Optional, Sequence, Tuple, Union
import torch
from torch._C import DispatchKey
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
import torch.utils._pytree as pytree
from torch._higher_order_ops.utils import (
    check_input_alias_and_mutation_return_outputs,
    reenter_make_fx,
    unique_graph_id,
    validate_subgraph_args_types,
)
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode, track_tensor_tree
from torch.utils._python_dispatch import _get_current_dispatch_mode


class SimpleLoopForOp(HigherOrderOperator):
    """Higher order op for :func:`simple_loop_for`."""

    def __init__(self):
        super().__init__("simple_loop_for")

    def __call__(self, n_iter, body_fn, operands, concatenation_dims=None):
        validate_subgraph_args_types(operands)
        return super().__call__(n_iter, body_fn, operands, concatenation_dims)

    def gen_schema(self, n_iter, body_fn, operands, concatenation_dims):
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
        assert concatenation_dims is None or len(concatenation_dims) == len(body_outputs), (
            f"concatenation_dims={concatenation_dims} but its length should be equal to "
            f"the number of outputs ({len(body_outputs)})"
        )
        schema_gen.add_schema_tree_spec(n_iter, body_fn, operands, concatenation_dims)
        return schema_gen.gen_schema()


simple_loop_for_op = SimpleLoopForOp()


def _simple_loop_for_fn(
    n_iter: torch.Tensor,
    body_fn: Callable,
    operands: Tuple[torch.Tensor, ...] = (),
    concatenation_dims: Optional[Sequence[int]] = None,
) -> Tuple[torch.Tensor, ...]:
    """
    Python implementation of the loop.

    :param n_iter: number of iteration
    :param body_fn: function implementing the body
    :param concatenation_dims: dimension used to reduce the list produced by the loop
    :param operands: arguments to the loop body
    :return: results
    """
    torch._check(
        isinstance(n_iter, (int, torch.Tensor)),
        lambda: f"Unexpected type {type(n_iter)} for n_iter",
    )
    torch._check(callable(body_fn), lambda: f"Unexpected type {type(body_fn)} for body_fn")
    torch._check(
        concatenation_dims is None or isinstance(concatenation_dims, (list, tuple)),
        lambda: f"Unexpected type {type(concatenation_dims)} for concatenation_dims",
    )
    torch._check(
        isinstance(operands, tuple), lambda: f"Unexpected type {type(operands)} for operands"
    )
    res: List[Union[torch.Tensor, Tuple[torch.Tensor, ...]]] = []
    for i in torch.arange(
        n_iter, dtype=torch.int64 if isinstance(n_iter, int) else n_iter.dtype
    ):
        r = body_fn(i, *operands)
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
        return torch.empty(tuple(), dtype=torch.float32, device=operands[0].device)

    n_res = len(res[0])
    return tuple(
        torch.cat(
            [r[i] for r in res],
            dim=(
                0
                if concatenation_dims is None or i >= len(concatenation_dims)
                else concatenation_dims[i]
            ),
        )
        for i in range(n_res)
    )


# from torch._functorch.utils import exposed_in
# @exposed_in("torch")
def _simple_loop_for(
    n_iter: Union[int, torch.Tensor],
    body_fn: Callable,
    operands: Tuple[torch.Tensor, ...] = (),
    concatenation_dims: Optional[Tuple[int, ...]] = None,
) -> Tuple[torch.Tensor, ...]:
    def _validate_input(n_iter, body_fn, operands, concatenation_dims):
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
        assert concatenation_dims is None or (
            isinstance(concatenation_dims, (list, tuple))
            and all(isinstance(i, int) for i in concatenation_dims)
        ), (
            f"concatenation_dims should be None or a list of integers but it is "
            f"{concatenation_dims}. Its length should be equal to the number of outputs."
        )
        assert torch._dynamo.is_dynamo_supported(), "simple_loop_for requires dynamo support."

    if torch.compiler.is_dynamo_compiling():
        return simple_loop_for_op(
            n_iter, body_fn, (n_iter, *operands), concatenation_dims=concatenation_dims
        )

    if isinstance(n_iter, (bool, int, float)):
        torch._check(
            isinstance(n_iter, int),
            lambda: f"n_iter must be an integer or a tensor not {type(n_iter)}",
        )
        return _simple_loop_for_fn(
            n_iter, body_fn, operands, concatenation_dims=concatenation_dims
        )

    def _loop_for_op_wrapper(n_iter, body_fn, operands, concatenation_dims):
        return simple_loop_for_op(n_iter, body_fn, operands, concatenation_dims)

    _validate_input(n_iter, body_fn, operands, concatenation_dims)

    # This requires torch>=2.10.
    from torch._higher_order_ops.utils import setup_compilation_env

    with setup_compilation_env() as _backend:
        return _loop_for_op_wrapper(n_iter, body_fn, operands, concatenation_dims)
        # return torch.compile(_loop_for_op_wrapper, backend=backend, fullgraph=True)(
        #    n_iter, body_fn, operands, concatenation_dims)


def trace_simple_loop_for(
    proxy_mode, func_overload, n_iter, body_fn, operands, concatenation_dims
):
    """See function ``simple_loop_for``."""
    assert isinstance(operands, (list, tuple)) and (
        concatenation_dims is None
        or (
            isinstance(concatenation_dims, (list, tuple))
            and all(isinstance(i, int) for i in concatenation_dims)
        )
    ), (
        f"simple_loop_for operands must be a list or tuple of tensors and SymInts and "
        f"concatenation_dims must be None or a list of integer, "
        f"operands={[type(o) for o in operands]}, "
        f"concatenation_dims={concatenation_dims}"
    )

    body_graph = reenter_make_fx(body_fn)(n_iter, *operands)

    body_outs = []
    for node in body_graph.graph.nodes:
        if node.op == "output":
            body_outs.extend(node.args)

    # flat_body_outs = pytree.arg_tree_leaves(*body_outs)
    _i, body_name = unique_graph_id(proxy_mode, prefix="body_graph")
    proxy_mode.tracer.root.register_module(body_name, body_graph)
    args = (n_iter, body_graph, operands, concatenation_dims)
    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, args)
    out_proxy = proxy_mode.tracer.create_proxy("call_function", func_overload, proxy_args, {})
    out = func_overload(n_iter, body_graph, operands, concatenation_dims)
    return track_tensor_tree(out, out_proxy, constant=None, tracer=proxy_mode.tracer)


@simple_loop_for_op.py_impl(DispatchKey.CompositeExplicitAutograd)
def loop_for_op_dense(n_iter, body_fn, operands, concatenation_dims=None):
    """Registered eager mode implementation."""
    assert all(isinstance(o, torch.Tensor) for o in operands) and (
        concatenation_dims is None
        or (
            isinstance(concatenation_dims, (list, tuple))
            and all(isinstance(i, int) for i in concatenation_dims)
        )
    ), (
        f"simple_loop_for operands must be a list or tuple of tensors and SymInts and "
        f"concatenation_dims must be None or a list of integer, "
        f"operands={[type(o) for o in operands]}, "
        f"concatenation_dims={concatenation_dims}"
    )
    mode = _get_current_dispatch_mode()
    assert mode is None, "Mode should never be enabled for CPU/CUDA key"
    return _simple_loop_for_fn(
        n_iter, body_fn, operands, concatenation_dims=concatenation_dims
    )


@simple_loop_for_op.py_impl(ProxyTorchDispatchMode)
def inner(mode, n_iter, body_fn, operands, concatenation_dims=None):
    """Registered tracing implementation."""
    return trace_simple_loop_for(
        mode, simple_loop_for_op, n_iter, body_fn, operands, concatenation_dims
    )


@simple_loop_for_op.py_impl(FakeTensorMode)
def simple_loop_for_fake_tensor_mode(mode, n_iter, body_fn, operands, concatenation_dims=None):
    """Registered FakeMode implementation."""
    ignore_fresh_unbacked = contextlib.nullcontext()
    if mode.shape_env:
        ignore_fresh_unbacked = mode.shape_env.ignore_fresh_unbacked_symbols()

    with mode, ignore_fresh_unbacked:
        flat_body_outs, true_body_spec = pytree.tree_flatten(body_fn(n_iter, *operands))

    return pytree.tree_unflatten(flat_body_outs, true_body_spec)


# Registration for autograd.
simple_loop_for_op.fallthrough(torch._C.DispatchKey.AutogradCPU)
simple_loop_for_op.fallthrough(torch._C.DispatchKey.AutogradCUDA)


def simple_loop_for(
    n_iter: Union[int, torch.Tensor],
    body_fn: Callable,
    operands: Tuple[torch.Tensor, ...] = (),
    concatenation_dims: Optional[Union[int, Sequence[int]]] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
    """
    Implements a simple loop for, the body is defined by a function which takes the
    iteration number stored in a tensor, and other tensors.
    It results one or several tensors in a tuple. All of them
    are finally concatenated along the first dimension.

    :param n_iter: iteration number
    :param body: function
    :param operands: bidy  arguments
    :param concatenation_dims: dimension or dimensions used to concatenate the output sequences
    :return: contenated outputs, the output is a Tensor

    An example with one output:

    .. runpython::
        :showcode:

        import torch
        from onnx_diagnostic.export.cf_simple_loop_for import simple_loop_for


        class Model(torch.nn.Module):
            def forward(self, n_iter, x):
                def body(i, x):
                    return (x[: i.item() + 1].unsqueeze(1),)

                return simple_loop_for(n_iter, body, (x,))


        model = Model()
        n_iter = torch.tensor(4, dtype=torch.int64)
        x = torch.arange(10, dtype=torch.float32)
        ep = torch.export.export(
            model, (n_iter, x), dynamic_shapes=({}, ({0: torch.export.Dim.DYNAMIC}))
        )
        print(ep)

    Another example with two outputs and a final concatenation on different axes.

    .. runpython::
        :showcode:

        import torch
        from onnx_diagnostic.export.cf_simple_loop_for import simple_loop_for


        class Model(torch.nn.Module):
            def forward(self, n_iter, x):
                def body(i: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor]:
                    return (x[: i.item() + 1].unsqueeze(1), x[i.item() + 1 :].unsqueeze(0))

                return simple_loop_for(n_iter, body, (x,), (0, 1))


        model = Model()
        n_iter = torch.tensor(4, dtype=torch.int64)
        x = torch.arange(10, dtype=torch.float32)
        ep = torch.export.export(
            model, (n_iter, x), dynamic_shapes=({}, ({0: torch.export.Dim.DYNAMIC}))
        )
        print(ep)
    """
    res = _simple_loop_for(
        n_iter,
        body_fn,
        operands,
        concatenation_dims=(
            (concatenation_dims,)
            if isinstance(concatenation_dims, int)
            else concatenation_dims
        ),
    )
    torch._check(
        isinstance(res, tuple), f"Output of the loop should be a tuple not {type(res)}."
    )
    return res[0] if len(res) == 1 else res
