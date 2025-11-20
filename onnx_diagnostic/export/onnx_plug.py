import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import onnx
import torch
from ..helpers import max_diff
from ..helpers.torch_helper import torch_dtype_to_onnx_dtype
from ..reference import OnnxruntimeEvaluator

TUPLE_TENSORS = Tuple[torch.Tensor, ...]


def is_exporting() -> bool:
    """
    Returns :func:`torch.compiler.is_exporting` or
    :func:`torch.compiler.is_compiling`.
    Changes ``_TEST_EXPORT`` to make it trigger.
    """
    return torch.compiler.is_exporting() or torch.compiler.is_compiling()


@dataclass
class VerifyResult:
    """
    Outputs of method :meth:`verify
    <onnx_diagnostic.export.onnx_plug.EagerDirectReplacementWithOnnx.verify>`.
    """

    eager_outputs: TUPLE_TENSORS
    onnx_output: TUPLE_TENSORS
    diffs: Tuple[Dict[str, float], ...]


class EagerDirectReplacementWithOnnx:
    """
    Replaces a piece of code by another one written in ONNX
    at export time. The function inserts a custom operator
    and links it to the eager_fn

    :param eager_fn: the code it replaces, it must be given in order to be able
        to execute the torch.fx.Graph the exporter produces
    :param shape_fn: the function produces dummy outputs with the shapes
        the exporter can use for the next operators in the graph
    :param function_proto: instances of ``onnx.FunctionProto``,
        its domain must be ``onnx_plug``
    :param n_inputs: number of inputs of the function, if not given,
        the class will infer it from eager_fn signature,
        only tensors must be counted
    :param n_outputs: same for the number of outputs,
        only tensors must be counted
    :param name: the name of the custom op, the function name if not specified
    :param kwargs: constants parameters with their default values
    :param verbose: verbose level

    Here is an example:

    .. runpython::
        :showcode:

        import onnx.helper as oh
        import torch
        from onnx_diagnostic.helpers.onnx_helper import pretty_onnx
        from onnx_diagnostic.export.onnx_plug import EagerDirectReplacementWithOnnx
        from onnx_diagnostic.export.api import to_onnx
        from onnx_diagnostic.torch_export_patches.patch_inputs import use_dyn_not_str


        def demo_customsub(x, y):
            return x - y


        def demo_customsub_shape(x, y):
            return torch.empty(torch.broadcast_shapes(x.shape, y.shape), dtype=x.dtype)


        def make_function_proto():
            return oh.make_function(
                "onnx_plug",
                "demo_customsub",
                ["x", "y"],
                ["z"],
                [oh.make_node("Sub", ["x", "y"], ["z"])],
                opset_imports=[oh.make_opsetid("", 22)],
            )


        class Model(torch.nn.Module):
            def forward(self, x):
                y = x.sum(axis=1, keepdim=True)
                d = torch.ops.onnx_plug.demo_customsub(x, y)
                return torch.abs(d)


        replacements = [
            EagerDirectReplacementWithOnnx(
                demo_customsub, demo_customsub_shape, make_function_proto(), 2, 1
            )
        ]

        x = torch.randn((3, 4), dtype=torch.float32)
        model = Model()
        ds = ({0: "d1", 1: "d2"},)

        # The exported program shows a custom op.
        ep = torch.export.export(model, (x,), dynamic_shapes=use_dyn_not_str(ds))
        print("ep")

        # As the exporter knows how the replace this custom op.
        # Let's export.

        onx = to_onnx(
            model,
            (x,),
            dynamic_shapes=ds,
            exporter="custom",
            onnx_plugs=replacements,
            target_opset=22,
            inline=False,
        ).model_proto

        print(pretty_onnx(onx))

        # And with :func:`torch.onnx.export`:

        onx = to_onnx(
            model,
            (x,),
            dynamic_shapes=ds,
            exporter="onnx-dynamo",
            onnx_plugs=replacements,
            target_opset=22,
            inline=False,
        ).model_proto

        print(pretty_onnx(onx))
    """

    def __init__(
        self,
        eager_fn: Callable[[TUPLE_TENSORS], TUPLE_TENSORS],
        shape_fn: Callable[[TUPLE_TENSORS], TUPLE_TENSORS],
        function_proto: onnx.FunctionProto,
        n_inputs: Optional[int] = None,
        n_outputs: Optional[int] = None,
        name: Optional[str] = None,
        kwargs: Optional[Dict[str, Union[int, float]]] = None,
        verbose: int = 0,
    ):
        assert isinstance(
            function_proto, onnx.FunctionProto
        ), f"Unexpected type {type(function_proto)} for function_proto"
        assert isinstance(n_inputs, int), f"not implemented yet when n_inputs={n_inputs}"
        assert isinstance(n_outputs, int), f"not implemented yet when n_inputs={n_outputs}"
        self.eager_fn = eager_fn
        self.shape_fn = shape_fn
        self.function_proto = function_proto
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.name = name or (
            eager_fn.__name__
            if "<" not in eager_fn.__name__
            else eager_fn.__qualname__.replace("<locals>", "L")
            .replace("<lambda>", "l")
            .replace(".", "_")
        )
        self.kwargs = kwargs or {}
        assert all(isinstance(v, (int, float)) for v in self.kwargs.values()), (
            f"Only int or floats are allowed for kwargs={kwargs}, one of them "
            f"does not respect that constraint."
        )
        sig = inspect.signature(self.eager_fn)
        params = list(sig.parameters)
        assert (
            len(params) >= n_inputs
        ), f"{self.eager_fn} accepts {params} as parameters < n_inputs={n_inputs}"
        assert n_inputs == len(function_proto.input), (
            f"Input mismatch n_inputs={n_inputs} but "
            f"function_proto.input={function_proto.input}"
        )
        assert n_outputs == len(function_proto.output), (
            f"Output mismatch n_outputs={n_outputs} but "
            f"function_proto.output={function_proto.output}"
        )
        assert (
            function_proto.domain == self.domain
        ), f"Function domain must be {self.domain!r} but it is {function_proto.domain!r}"
        self.args_name = [p for p in params if p not in self.kwargs]
        self.kwargs_name = [p for p in params if p in self.kwargs]
        self.verbose = verbose
        self.custom_op = self._register()

    @property
    def domain(self) -> str:
        "Returns the onnx domain."
        return "onnx_plug"

    @property
    def target_name(self) -> str:
        "Returns the target name (see in the exported program)."
        return f"{self.domain}::{self.name}"

    @property
    def torch_op(self) -> Callable:
        "Returns ``torch.ops.onny_plug.<name>``."
        return getattr(getattr(torch.ops, self.domain), self.name).default

    def __call__(self, *args, **kwargs):
        """Calls eager_fn or shape_fn if the model is being exported."""
        if is_exporting():
            return self.torch_op(*args)
        return self.eager_fn(*args, **kwargs)

    def _register(self):
        """Registers the custom op."""
        input_args = [f"Tensor {p}" for p in self.args_name]
        for p in self.kwargs_name:
            val = self.kwargs[p]
            if isinstance(val, int):
                input_args.append(f"int {p}={val}")
            elif isinstance(val, float):
                input_args.append(f"float {p}={val}")
            else:
                raise NotImplementedError(
                    f"kwargs {p!r} has a default value of unsupported type {type(val)}"
                )

        inputs = ", ".join(input_args)
        schema = f"({inputs}) -> Tensor"
        if self.n_outputs > 1:
            schema += "[]"
        if self.verbose:
            print(
                f"[EagerDirectReplacementWithOnnx._register] "
                f"'torch.ops.{self.domain}.{self.name}"
            )
            print(f"[EagerDirectReplacementWithOnnx._register] schema={schema}")
        custom_def = torch.library.CustomOpDef(self.domain, self.name, schema, self.eager_fn)
        custom_def.register_kernel(None)(self.eager_fn)
        custom_def._abstract_fn = self.shape_fn

    def verify(self, *args, engine: Optional[Callable] = None) -> VerifyResult:
        """
        Verifies that the eager mode is equivalent to the onnx function given
        as a replacements. This function evaluates `eager_fn`, checks that the shapes
        are equivalent to the ones given by `shape_fn`, and finally evaluates the
        onnx translation if the previous did not fail.

        :param args: function inputs
        :param engine: by default an instance of
            :class:`onnx_diagnostic.reference.OnnxruntimeEvaluator`.
        :return: outputs of :func:`onnx_diagnostic.helpers.max_diff`
        """
        expected = self.eager_fn(*args)
        shapes = self.shape_fn(*args)
        if isinstance(expected, torch.Tensor):
            expected = (expected,)
            assert isinstance(shapes, torch.Tensor), (
                f"eager_fn={self.eager_fn} returns a Tensor but shape_fn={self.shape_fn} "
                f"returns a {type(shapes)}"
            )
            shapes = (shapes,)
        assert isinstance(expected, tuple) and isinstance(shapes, tuple), (
            f"eager_fn={self.eager_fn} returns a {type(expected)} "
            f"and shape_fn={self.shape_fn} returns a {type(shapes)}"
        )
        assert len(expected) and len(shapes), (
            f"eager_fn={self.eager_fn} and shape_fn={self.shape_fn} "
            f"do not return the same number of tensors."
        )
        for i, (e, s) in enumerate(zip(expected, shapes)):
            assert e.dtype == s.dtype, (
                f"Type mismatch {e.dtype} != {s.dtype} for output {i}, "
                f"eager_fn={self.eager_fn} and shape_fn={self.shape_fn}"
            )
            assert e.shape == s.shape, (
                f"Type mismatch {e.shape} != {s.shape} for output {i}, "
                f"eager_fn={self.eager_fn} and shape_fn={self.shape_fn}"
            )

        # Now the ONNX execution.
        assert engine is None, f"Not implemented yet with engine={engine!r}"
        sess = OnnxruntimeEvaluator(self.function_proto)
        feeds = dict(zip(sess.input_names, args))
        got = sess.run(None, feeds)
        diffs = tuple(max_diff(e, g) for e, g in zip(expected, got))
        return VerifyResult(eager_outputs=expected, onnx_output=tuple(got), diffs=diffs)  # type: ignore[arg-type]

    def custom_converter(
        self,
    ) -> Callable:
        """
        Returns a function which
        converts a custom ops found in the fx graph into ONNX
        following the API of the custom exporter.
        The converter adds a custom op and registers the local function.
        """

        def converter(
            g: Any,  # GraphBuilder
            sts: Optional[Dict[str, Any]],
            outputs: List[str],
            *args,
            **kwargs,
        ) -> Any:
            if not g.has_local_function(
                self.function_proto.name, domain=self.function_proto.domain
            ):
                g.add_function(self.function_proto)
            ags = args[: len(self.args_name)]
            kws = dict(zip(self.kwargs_name, args[len(self.args_name) :]))
            kws.update(kwargs)
            res = g.make_node(
                self.function_proto.name,
                ags,
                outputs,
                domain=self.function_proto.domain,
                name=self.target_name,
                **kws,
            )
            if not sts:
                new_shapes = self.shape_fn(*args)
                if not isinstance(new_shapes, tuple):
                    new_shapes = (new_shapes,)
                for sh, o in zip(new_shapes, outputs):
                    g.set_type(o, torch_dtype_to_onnx_dtype(sh.dtype))
                    g.set_shape(o, sh.shape)
            return res

        return converter

    def onnx_dynamo_converter(self) -> Callable:
        """
        Returns a function which
        which converts a custom ops found in the fx graph into ONNX
        following the API of :func:`torch.onnx.export`.
        """
        import onnxscript

        onnx_plug_op = onnxscript.values.Opset(domain=self.function_proto.domain, version=1)
        schema = onnx_plug_op[self.function_proto.name]
        if schema is None:
            all_types = [
                "tensor(float)",
                "tensor(float16)",
                "tensor(bfloat16)",
                "tensor(double)",
                "tensor(int64)",
                "tensor(int32)",
            ]
            type_constraints = []
            for i in range(self.n_inputs):
                type_constraints.append((f"T{i}", all_types, ""))
            for i in range(self.n_outputs):
                type_constraints.append((f"U{i}", all_types, ""))
            schema = onnx.defs.OpSchema(
                self.function_proto.name,
                self.function_proto.domain,
                1,
                inputs=[
                    onnx.defs.OpSchema.FormalParameter(f"arg_{i}", f"T{i}")
                    for i in range(self.n_inputs)
                ],
                outputs=[
                    onnx.defs.OpSchema.FormalParameter(f"res_{i}", f"U{i}")
                    for i in range(self.n_outputs)
                ],
                type_constraints=type_constraints,
            )
            onnx.defs.register_schema(schema)
        op = onnxscript.values.Op(onnx_plug_op, self.function_proto.name, schema)

        def converter(*cargs):
            return op(*cargs, n_outputs=self.n_outputs)

        return onnxscript.values.TracedOnnxFunction(onnx_plug_op, converter)
