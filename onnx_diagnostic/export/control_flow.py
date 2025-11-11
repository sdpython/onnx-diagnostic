import contextlib
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import onnx
import onnx.helper as oh
import torch
from torch._higher_order_ops.utils import materialize_as_graph
from torch._higher_order_ops.utils import check_input_alias_and_mutation_return_outputs
from ..helpers.onnx_helper import pretty_onnx
from .api import to_onnx

_TEST_EXPORT = False
_REGISTERED_SCHEMA = {}  # type: ignore[var-annotated]
_DISPATCHER = None


def create_global_dispatcher():
    global _DISPATCHER

    if not _DISPATCHER:
        from experimental_experiment.torch_interpreter import Dispatcher

        class ControlFlowDispatcher(Dispatcher):
            def __init__(self):
                super().__init__({})

            def register(self, aten_name: str, converter: Callable):
                assert aten_name not in self.registered_functions, (
                    f"Name {aten_name!r} is already registered in "
                    f"{sorted(self.registered_functions)}"
                )
                self.registered_functions[aten_name] = converter

        _DISPATCHER = ControlFlowDispatcher()
    return _DISPATCHER


@contextlib.contextmanager
def enable_code_export_control_flow():
    """Enables the code means to be exported."""
    global _TEST_EXPORT
    old = _TEST_EXPORT
    _TEST_EXPORT = True
    try:
        yield
    finally:
        _TEST_EXPORT = old


def is_exporting():
    """
    Returns :func:`torch.compiler.is_exporting` or
    :func:`torch.compiler.is_compiling`
    """
    return _TEST_EXPORT or torch.compiler.is_exporting() or torch.compiler.is_compiling()


def _loop_for_fn(n_iter, body_fn, reduction_dim, args):
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
    n_iter,
    body_fn,
    reduction_dim,
    args,
    body_gm=None,
    body_mutated_inputs=None,
    body_outputs=None,
):
    global _DISPATCHER
    srank = "_".join("x".join(map(str, s.shape)) for s in body_outputs)
    sred = "x".join(map(str, reduction_dim)) if reduction_dim else ""
    name = f"loop_for_{body_fn.__name__}_{id(body_fn)}_{srank}_{sred}"
    if name in _REGISTERED_SCHEMA:
        return name, _REGISTERED_SCHEMA[name][0]
    sig = inspect.signature(body_fn)
    inputs = ", ".join([f"Tensor {p}" for p in sig.parameters])
    schema = f"({inputs}) -> Tensor"
    if len(body_outputs) > 1:
        schema += "[]"
    custom_def = torch.library.CustomOpDef("onnx_higher_ops", name, schema, body_fn)
    custom_def.register_kernel("cpu")(body_fn)

    custom_def._abstract_fn = lambda *_args, _o=body_outputs: (
        tuple([torch.empty_like(s) for s in _o]) if len(_o) > 1 else torch.empty_like(_o[0])
    )
    onx = convert_into_onnx(body_gm, args, body_mutated_inputs, body_outputs)
    to_register = (
        custom_def,
        onx,
        (
            lambda g, sts, outputs, *args, body=onx, reduction_dim=reduction_dim, name=name: (
                convert_custom_loop_into_onnx(
                    g, sts, outputs, *args, body=body, reduction_dim=reduction_dim, name=name
                )
            )
        ),
    )
    if _DISPATCHER is None:
        create_global_dispatcher()
    assert _DISPATCHER
    _DISPATCHER.register(f"onnx_higher_ops::{name}", to_register[-1])
    _REGISTERED_SCHEMA[name] = to_register
    return name, custom_def


def convert_custom_loop_into_onnx(
    g: Any,  # "GreaphBuilder"
    sts: Dict[str, Any],
    outputs: List[str],
    *args: str,
    body: onnx.GraphProto,
    reduction_dim: Optional[Tuple[int, ...]] = None,
    name: str = "loop_for",
) -> Union[str, Tuple[str, ...]]:
    graph = body.graph if isinstance(body, onnx.ModelProto) else body
    assert isinstance(
        graph, onnx.GraphProto
    ), f"Unexpected type {type(body)} for body{g.get_debug_msg()}"
    assert len(outputs) == len(graph.output), (
        f"Length mismatch, expecting {len(outputs)} but got "
        f"{len(graph.output)}, \n--\n{pretty_onnx(body)}"
        f"{g.get_debug_msg()}"
    )
    input_names = [i.name for i in graph.input]
    inputs = [
        *graph.input[:1],
        oh.make_tensor_value_info("cond_unused", onnx.TensorProto.BOOL, []),
        *[
            oh.make_tensor_sequence_value_info(
                f"loop_in{i}", graph.output[i].type.tensor_type.elem_type, None
            )
            for i in range(len(graph.output))
        ],
        # hidden inputs are not added
    ]
    nodes = [
        oh.make_node("Identity", ["cond_unused"], ["cond_out"]),
        *[oh.make_node("Identity", [a], [r]) for a, r in zip(args[1:], input_names[1:])],
        *graph.node,
        *[
            oh.make_node(
                "SequenceInsert",
                [f"loop_in{i}", graph.output[i].name],
                [f"loop_out{i}"],
            )
            for i in range(len(graph.output))
        ],
    ]
    graph_outputs = [
        oh.make_tensor_value_info("cond_out", onnx.TensorProto.BOOL, []),
        *[
            oh.make_tensor_sequence_value_info(
                f"loop_out{i}", graph.output[i].type.tensor_type.elem_type, None
            )
            for i in range(len(graph.output))
        ],
    ]
    graph = oh.make_graph(
        nodes, graph.name, inputs, graph_outputs, graph.initializer, graph.sparse_initializer
    )

    sequences = [g.op.SequenceEmpty() for _ in outputs]

    outloop = [g.unique_name(f"loop_for{i}") for i in range(len(sequences))]

    for i, s in enumerate(sequences):
        g.set_sequence(s, graph.output[i].type.tensor_type.elem_type)
    g.make_node("Loop", [args[0], "", *sequences], outloop, name=name, body=graph)
    for i, o in enumerate(outloop):
        g.set_sequence(o, graph.output[i].type.tensor_type.elem_type)
    _res = [
        g.op.ConcatFromSequence(
            out,
            outputs=[o],
            name=name,
            axis=0 if not reduction_dim or i >= len(reduction_dim) else reduction_dim[i],
        )
        for i, (out, o) in enumerate(zip(outloop, outputs))
    ]
    if not sts:
        for i, o in enumerate(outputs):
            g.set_type(o, graph.output[i].type.tensor_type.elem_type)
            g.set_rank(o, len(graph.output[i].type.tensor_type.shape.dims))
    return tuple(outputs) if len(outputs) > 1 else outputs[0]


def convert_into_onnx(body_gm, args, body_mutated_inputs, body_outputs):
    """Converts into ONNX."""
    # This does not work with onnx-dynamo.
    # opset still needs to be defined
    container = to_onnx(body_gm, args, exporter="custom")
    return container.model_proto


def loop_for(
    n_iter: Union[torch.SymInt, torch.Tensor],
    body_fn: Callable[..., Tuple[torch.Tensor]],
    args: Tuple[torch.Tensor, ...],
    reduction_dim: Optional[Tuple[int]] = None,
) -> Tuple[torch.Tensor, ...]:
    """
    High operators used to easily export a loop in ONNX.
    Does not fully work with :func:`torch.export.export`,
    it does replaces a custom op with a loop operator afterwards.

    .. runpython::
        :showcode:

        import torch
        import onnxruntime
        from onnx_diagnostic.export.api import to_onnx
        from onnx_diagnostic.export.control_flow import loop_for


        class Model(torch.nn.Module):
            def forward(self, n_iter, x):
                def body(i, x):
                    return x[: i.item() + 1].unsqueeze(1)

                return loop_for(n_iter, body, (x,))


        model = Model()
        n_iter = torch.tensor(4, dtype=torch.int64)
        x = torch.arange(10, dtype=torch.float32)
        expected = model(n_iter, x)
        print("expected:", expected)

        onx = to_onnx(
            model,
            (n_iter, x),
            dynamic_shapes=({}, ({0: torch.export.Dim.DYNAMIC})),
            exporter="custom",
            use_control_flow_dispatcher=True,
        ).model_proto


        sess = onnxruntime.InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, dict(zip(["n_iter", "x"], [n_iter.numpy(), x.numpy()])))
        print("got:", got)


        # The loop is exported as a custom ops.
        ep = torch.export.export(
            model, (n_iter, x), dynamic_shapes=({}, ({0: torch.export.Dim.DYNAMIC}))
        )
        print(ep)

    """
    assert args, "The function should have at least one arg."
    assert (
        isinstance(n_iter, torch.Tensor)
        and n_iter.dtype == torch.int64
        and len(n_iter.shape) == 0
    ), f"Only a tensor for one int64 is allowed for n_iter but it equal to {n_iter}."
    if is_exporting():
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
        fct = getattr(torch.ops.onnx_higher_ops, name)
        return fct(n_iter, *args)

    return _loop_for_fn(n_iter, body_fn, reduction_dim, args)
