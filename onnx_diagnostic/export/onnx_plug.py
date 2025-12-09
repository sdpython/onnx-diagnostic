import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import onnx
import torch
from ..helpers import max_diff, string_type
from ..helpers.torch_helper import (
    torch_dtype_to_onnx_dtype,
    onnx_dtype_to_torch_dtype,
    int_device_to_torch_device,
)
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
    onnx_outputs: TUPLE_TENSORS
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
    :param version_selector: selects the version based on the arguments,
        see below for an example, this allows the user to define different
        onnx version depending on the inputs
    :param default_opset: opset to use by default
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

    We do the same with :func:`torch.onnx.export`:

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
            exporter="onnx-dynamo",
            onnx_plugs=replacements,
            target_opset=22,
            inline=False,
        ).model_proto

        print(pretty_onnx(onx))

    This shows how to define multiple versions depending on the device,
    the type or the targeted onnx opset.

    .. code-block:: python

        def qwen_version_selector(opset: int, *args: torch.Tensor) -> Tuple[str, torch.dtype]:
            first_tensor = next(a for a in args if a is not None)
            dtype = first_tensor.dtype
            itype = torch_dtype_to_onnx_dtype(dtype)
            if dtype == torch.float32:
                if opset >= 23:
                    return "LOOPA23", itype
                return "LOOPMHA", itype
            if dtype == torch.float16:
                if first_tensor.is_cuda:
                    return "PACKED", itype
                return "LOOPMHA", itype
            raise AssertionError(
                f"Unable to handle type {torch.dtype} (itype={itype}) "
                f"on device {torch.device} with opset={opset}"
            )

        qwen_sdpa_attention_versatile = EagerDirectReplacementWithOnnx(
            qwen_sdpa_attention,
            lambda qs, *args, **kwargs: torch.empty(
                (qs.shape[0], qs.shape[2], qs.shape[1], qs.shape[3]),
                dtype=qs.dtype,
                device=qs.device,
            ),
            {
                ("PACKED", onnx.TensorProto.FLOAT16): _add_com_microsoft_opset(
                    PackedAttention.to_function_proto()
                ),
                ("LOOPA23", onnx.TensorProto.FLOAT): LoopAttention23.to_function_proto(),
                ("LOOPA23", onnx.TensorProto.FLOAT16): _update_sequence_type(
                    onnx.TensorProto.FLOAT16, LoopAttention23.to_function_proto()
                ),
                ("LOOPMHA", onnx.TensorProto.FLOAT): _add_com_microsoft_opset(
                    LoopMHAAttention.to_function_proto()
                ),
                ("LOOPMHA", onnx.TensorProto.FLOAT16): _update_sequence_type(
                    onnx.TensorProto.FLOAT16,
                    _add_com_microsoft_opset(LoopMHAAttention.to_function_proto()),
                ),
            },
            n_inputs=4,
            n_outputs=1,
            kwargs=dict(scaling=0.11180339887498948, num_heads=16),
            name="qwen_sdpa_attention_versatile",
            version_selector=qwen_version_selector,
        )
    """

    def __init__(
        self,
        eager_fn: Callable[[TUPLE_TENSORS], TUPLE_TENSORS],
        shape_fn: Callable[[TUPLE_TENSORS], TUPLE_TENSORS],
        function_proto: Union[onnx.FunctionProto, Dict[Any, onnx.FunctionProto]],
        n_inputs: Optional[int] = None,
        n_outputs: Optional[int] = None,
        name: Optional[str] = None,
        kwargs: Optional[Dict[str, Union[int, float]]] = None,
        verbose: int = 0,
        version_selector: Optional[Callable[..., Tuple[Any, ...]]] = None,
        default_opset: int = 22,
    ):
        assert isinstance(function_proto, onnx.FunctionProto) or (
            isinstance(function_proto, dict)
            or all(isinstance(v, onnx.FunctionProto) for v in function_proto.values())
        ), f"Unexpected type {type(function_proto)} for function_proto"
        assert isinstance(n_inputs, int), f"not implemented yet when n_inputs={n_inputs}"
        assert isinstance(n_outputs, int), f"not implemented yet when n_outputs={n_outputs}"
        self.eager_fn = eager_fn
        self.shape_fn = shape_fn
        self._function_proto = (
            function_proto if isinstance(function_proto, onnx.FunctionProto) else None
        )
        self._function_proto_versioned = (
            function_proto if isinstance(function_proto, dict) else {}
        )
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
        self.args_name = [p for p in params if p not in self.kwargs]
        self.kwargs_name = [p for p in params if p in self.kwargs]
        self.verbose = verbose
        self.custom_op = self._register()
        self.version_selector = version_selector
        self.default_opset = default_opset
        self._check_protos(params)

    def _check_protos(self, params):
        assert (
            len(params) >= self.n_inputs
        ), f"{self.eager_fn} accepts {params} as parameters < n_inputs={self.n_inputs}"

        # one proto
        assert self._function_proto is None or self.n_inputs == len(
            self._function_proto.input
        ), (
            f"Input mismatch n_inputs={self.n_inputs} but "
            f"function_proto.input={self._function_proto.input}"
        )
        assert self._function_proto is None or self.n_outputs == len(
            self._function_proto.output
        ), (
            f"Output mismatch n_outputs={self.n_outputs} but "
            f"function_proto.output={self._function_proto.output}"
        )
        assert self._function_proto is None or (
            self._function_proto.domain == self.domain
        ), f"Function domain must be {self.domain!r} but it is {self._function_proto.domain!r}"

        # multiple protos
        assert all(
            self.n_inputs == len(v.input) for v in self._function_proto_versioned.values()
        ), f"Output mismatch n_inputs={self.n_inputs} but one version is wrong"
        assert all(
            self.n_outputs == len(v.output) for v in self._function_proto_versioned.values()
        ), f"Output mismatch n_outputs={self.n_outputs} but one version is wrong"
        assert all(
            v.domain == self.domain for v in self._function_proto_versioned.values()
        ), f"Function domain must be {self.domain!r} but it is different in one version"
        assert (
            not self._function_proto_versioned or self.version_selector
        ), "version_selector is needed when multiple protos are given."

    def get_function_proto(self, opset: int, *args) -> onnx.FunctionProto:
        """Returns the correct version based on the inputs."""
        if self._function_proto:
            return self._function_proto
        assert isinstance(
            opset, int
        ), f"The first argument must be an integer for the onnx opset but it is {type(opset)}"
        assert any(
            a is not None for a in args
        ), f"Unexpected args={string_type(args, with_shape=True)}"
        try:
            key = self.version_selector(opset, *args)  # type: ignore[misc]
        except (ValueError, AttributeError) as e:
            raise AssertionError(
                f"Unable to select a version, fails to get a key, available="
                f"{set(self._function_proto_versioned)}, "
                f"args={string_type(args,with_shape=True)}"
            ) from e
        assert key in self._function_proto_versioned, (
            f"Unable to select a version, key={key}, available="
            f"{set(self._function_proto_versioned)}, args={string_type(args,with_shape=True)}"
        )
        return self._function_proto_versioned[key]

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
            elif isinstance(val, str):
                input_args.append(f"str {p}={val}")
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

    def verify(
        self,
        *args,
        engine: Optional[Callable] = None,
        dump_onnx_model: Optional[str] = None,
        opset: int = 22,
        **kwargs,
    ) -> VerifyResult:
        """
        Verifies that the eager mode is equivalent to the onnx function given
        as a replacements. This function evaluates `eager_fn`, checks that the shapes
        are equivalent to the ones given by `shape_fn`, and finally evaluates the
        onnx translation if the previous did not fail.

        :param args: function inputs
        :param kwargs: arguments for eager_fn
        :param engine: by default an instance of
            :class:`onnx_diagnostic.reference.OnnxruntimeEvaluator`.
        :param dump_onnx_model: to dump the onnx model used to verify
            eager and onnx produce the same results
        :param opset: onnx opset to use
        :param kwargs: additional arguments to the function
        :return: outputs of :func:`onnx_diagnostic.helpers.max_diff`
        """
        expected = self.eager_fn(*args, **kwargs)
        shapes = self.shape_fn(*args, **kwargs)
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
        ags, kws = self._make_args_kwargs(*args, **kwargs)
        sess = OnnxruntimeEvaluator(
            self.get_function_proto(opset, *args),
            whole=True,
            dump_onnx_model=dump_onnx_model,
            function_kwargs=kws,
        )
        feeds = dict(zip(sess.input_names, ags))
        got = sess.run(None, feeds)
        diffs = tuple(max_diff(e, g, hist=[0.1, 0.01]) for e, g in zip(expected, got))
        return VerifyResult(eager_outputs=expected, onnx_outputs=tuple(got), diffs=diffs)  # type: ignore[arg-type]

    def _make_args_kwargs(self, *args, **kwargs):
        ags = args[: len(self.args_name)]
        kws = dict(zip(self.kwargs_name, args[len(self.args_name) :]))
        kws.update(kwargs)
        return ags, kws

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
            has_devices = [a for a in args if isinstance(a, str) and g.has_device(a)]
            assert (
                has_devices
            ), f"Missing device for any of the inputs {args}{g.get_debug_msg()}"
            arg_device = has_devices[0]
            fake_tensor = torch.empty(
                tuple([(_ if isinstance(_, int) else 2) for _ in g.get_shape(args[0])]),
                dtype=onnx_dtype_to_torch_dtype(g.get_type(args[0])),
                device=int_device_to_torch_device(g.get_device(arg_device)),
            )
            function_proto = self.get_function_proto(g.main_opset, fake_tensor)
            if not g.has_local_function(function_proto.name, domain=function_proto.domain):
                g.add_function(function_proto)
            ags, kws = self._make_args_kwargs(*args, **kwargs)
            res = g.make_node(
                function_proto.name,
                ags,
                outputs,
                domain=function_proto.domain,
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

        onnx_plug_op = onnxscript.values.Opset(domain=self.domain, version=1)

        def get_proto(*args):
            function_proto = self.get_function_proto(self.default_opset, *args)
            schema = onnx_plug_op[function_proto.name]
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
                    function_proto.name,
                    function_proto.domain,
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
            op = onnxscript.values.Op(onnx_plug_op, function_proto.name, schema)
            return op

        def converter(*cargs, **ckwargs):
            ags, kws = self._make_args_kwargs(*cargs, **ckwargs)
            op = get_proto(*cargs)
            return op(*ags, n_outputs=self.n_outputs, **kws)

        return onnxscript.values.TracedOnnxFunction(onnx_plug_op, converter)
