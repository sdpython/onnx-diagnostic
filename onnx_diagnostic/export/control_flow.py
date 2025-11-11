import contextlib
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import onnx
import onnx.helper as oh
import torch
from torch._higher_order_ops.utils import materialize_as_graph
from torch._higher_order_ops.utils import check_input_alias_and_mutation_return_outputs
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
    :param body_fn: function implementating the body
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
    reduction_dim: Optional[List[int]],
    args: List[torch.Tensor],
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
    onx = convert_into_onnx(body_gm, args)
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
    """
    Converts a custom op ``higher_ops::loop_for...`` into e sequence of node.

    :param g: GreaphBuilder
    :param sts: if not defined, torch does not know the output shapes
    :param outputs: output names
    :param args: input argument known at export time
    :param body: GraphProto, the loop body
    :param reduction_dim: the dimension to follow when aggregating the
        list of tensors after the loop ran
    :param name: to give the onnx nodes a name
    :return: output names
    """
    graph = body.graph if isinstance(body, onnx.ModelProto) else body
    assert isinstance(
        graph, onnx.GraphProto
    ), f"Unexpected type {type(body)} for body{g.get_debug_msg()}"
    assert len(outputs) == 1, f"Only one outputs is expected but outputs={outputs!r}"
    if len(graph.output) != 1:
        outputs = [f"{outputs[0]}#{i}" for i in range(len(graph.output))]
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
    return outputs if len(outputs) > 1 else outputs[0]


def convert_into_onnx(
    body_gm: torch.fx.GraphModule, args: List[torch.Tensor]
) -> onnx.ModelProto:
    """
    Converts a torch.fx.GraphModule into ONNX.
    It returns a ModelProto.

    :param body_gm: a torch.fx.GraphModule
    :param args: arguments known at export time
    :return: a ModelProto
    """
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

    Another example with two outputs:

    .. runpython::
        :showcode:

        import torch
        import onnxruntime
        from onnx_diagnostic.export.api import to_onnx
        from onnx_diagnostic.export.control_flow import loop_for


        class Model(torch.nn.Module):
            def forward(self, n_iter, x):
                def body(i, x):
                    return x[: i.item() + 1].unsqueeze(1), x[: i.item() + 1].unsqueeze(1) + 1

                two = loop_for(n_iter, body, (x,))
                return two[0] + two[1]


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

    A last example with ``reduction_dim``:

    .. runpython::
        :showcode:

        import torch
        import onnxruntime
        from onnx_diagnostic.export.api import to_onnx
        from onnx_diagnostic.export.control_flow import loop_for


        class Model(torch.nn.Module):
            def forward(self, n_iter, x):
                def body(i, x):
                    return x[: i.item() + 1].unsqueeze(1), x[: i.item() + 1].unsqueeze(0) + 1

                two = loop_for(n_iter, body, (x,), reduction_dim=[0, 1])
                return two[0] + two[1].T


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
