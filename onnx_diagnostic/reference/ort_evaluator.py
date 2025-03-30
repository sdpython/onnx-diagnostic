from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from onnx import (
    GraphProto,
    FunctionProto,
    ModelProto,
    NodeProto,
    TypeProto,
    ValueInfoProto,
    helper as oh,
    load,
)
from onnx.defs import onnx_opset_version
import onnxruntime
from ..helpers import string_type
from ..helpers.onnx_helper import pretty_onnx, dtype_to_tensor_dtype, to_array_extended
from ..helpers.ort_session import (
    InferenceSessionForTorch,
    InferenceSessionForNumpy,
    _InferenceSession,
)

PROTO = (FunctionProto, ModelProto, GraphProto, NodeProto)
Proto = Union[FunctionProto, ModelProto, GraphProto, NodeProto]


class OnnxruntimeEvaluator:
    """
    This class loads an onnx model and the executes one by one the nodes
    with onnxruntime. This class is mostly meant for debugging.

    :param proto: proto or filename
    :param session_options: options
    :param providers: providers
    :param nvtx: enable nvidia events
    :param providers: `None`, `"CPU"`, `"CUDA"` or a list of providers
    :param graph_optimization_level: see :class:`onnxruntime.SessionOptions`
    :param log_severity_level: see :class:`onnxruntime.SessionOptions`
    :param log_verbosity_level: see :class:`onnxruntime.SessionOptions`
    :param optimized_model_filepath:  see :class:`onnxruntime.SessionOptions`
    :param disable_aot_function_inlining:  see :class:`onnxruntime.SessionOptions`
    :param use_training_api: use onnxruntime-traning API
    :param verbose: verbosity
    :param local_functions: additional local function
    :param ir_version: ir version to use when unknown
    :param opsets: opsets to use when unknown
    """

    def __init__(
        self,
        proto: Union[str, Proto, "OnnxruntimeEvaluator"],
        session_options: Optional[onnxruntime.SessionOptions] = None,
        providers: Optional[Union[str, List[str]]] = None,
        nvtx: bool = False,
        enable_profiling: bool = False,
        graph_optimization_level: Union[onnxruntime.GraphOptimizationLevel, bool] = None,
        log_severity_level: Optional[int] = None,
        log_verbosity_level: Optional[int] = None,
        optimized_model_filepath: Optional[str] = None,
        disable_aot_function_inlining: Optional[bool] = None,
        use_training_api: bool = False,
        verbose: int = 0,
        local_functions: Optional[
            Dict[Tuple[str, str], Union[Proto, "OnnxruntimeEvaluator"]]
        ] = None,
        ir_version: int = 10,
        opsets: Optional[Union[int, Dict[str, int]]] = None,
    ):
        if isinstance(proto, str):
            self.proto: Proto = load(proto)
        elif isinstance(proto, OnnxruntimeEvaluator):
            assert isinstance(
                proto.proto, PROTO
            ), f"Unexpected type for proto.proto {type(proto.proto)}"
            self.proto = proto.proto
        else:
            self.proto = proto
        assert isinstance(
            self.proto, PROTO
        ), f"Unexpected type for self.proto {type(self.proto)}"

        self._cache: Dict[
            Any, Tuple[Proto, Union["OnnxruntimeEvaluator", _InferenceSession]]  # noqa: UP037
        ] = {}
        self.ir_version = ir_version
        self.opsets = opsets
        self.session_kwargs: Dict[str, Any] = dict(
            session_options=session_options,
            providers=providers,
            nvtx=nvtx,
            enable_profiling=enable_profiling,
            graph_optimization_level=graph_optimization_level,
            log_severity_level=log_severity_level,
            log_verbosity_level=log_verbosity_level,
            optimized_model_filepath=optimized_model_filepath,
            disable_aot_function_inlining=disable_aot_function_inlining,
            use_training_api=use_training_api,
        )

        self.nodes = (
            [self.proto]
            if isinstance(self.proto, NodeProto)
            else (
                list(
                    self.proto.graph.node if hasattr(self.proto, "graph") else self.proto.node
                )
            )
        )
        self.rt_inits_ = (
            {init.name: to_array_extended(init) for init in self.proto.graph.initializer}
            if hasattr(self.proto, "graph")
            else {}
        )
        self.rt_nodes_ = self.nodes.copy()
        self.verbose = verbose
        self.local_functions: Dict[Tuple[str, str], "OnnxruntimeEvaluator"] = (  # noqa: UP037
            {(f.domain, f.name): self.__class__(f) for f in self.proto.functions}
            if hasattr(self.proto, "functions")
            else {}
        )
        if local_functions:
            self.local_functions.update(local_functions)

    @property
    def input_names(self) -> List[str]:
        "Returns input names."
        if isinstance(self.proto, NodeProto):
            return self.nodes[0].input
        return [
            getattr(o, "name", o)
            for o in (
                self.proto.graph.input if hasattr(self.proto, "graph") else self.proto.input
            )
        ]

    @property
    def output_names(self) -> List[str]:
        "Returns output names."
        if isinstance(self.proto, NodeProto):
            return self.nodes[0].output
        return [
            getattr(o, "name", o)
            for o in (
                self.proto.graph.output if hasattr(self.proto, "graph") else self.proto.output
            )
        ]

    @property
    def input_types(self) -> List[TypeProto]:
        "Returns input types."
        if not isinstance(self.proto, (ModelProto, GraphProto)):
            raise ValueError(f"Cannot guess input types for type {type(self.proto)}")
        g = self.proto.graph if hasattr(self.proto, "graph") else self.proto
        return [i.type for i in g.input]

    @property
    def output_types(self) -> List[TypeProto]:
        "Returns output types."
        if not isinstance(self.proto, (ModelProto, GraphProto)):
            raise ValueError(f"Cannot guess output types for type {type(self.proto)}")
        g = self.proto.graph if hasattr(self.proto, "graph") else self.proto
        return [i.type for i in g.output]

    def _log_arg(self, a: Any) -> Any:
        if isinstance(a, (str, int, float)):
            return a
        device = f"D{a.get_device()}:" if hasattr(a, "detach") else ""
        if hasattr(a, "shape"):
            if self.verbose < 4:  # noqa: PLR2004
                return f"{device}{a.dtype}:{a.shape} in [{a.min()}, {a.max()}]"
            elements = a.ravel().tolist()
            if len(elements) > 10:  # noqa: PLR2004
                elements = elements[:10]
                return f"{device}{a.dtype}:{a.shape}:{','.join(map(str, elements))}..."
            return f"{device}{a.dtype}:{a.shape}:{elements}"
        if hasattr(a, "append"):
            return ", ".join(map(self._log_arg, a))
        return a

    def _log(self, level: int, pattern: str, *args: Any) -> None:
        if level < self.verbose:
            new_args = [self._log_arg(a) for a in args]
            print(pattern % tuple(new_args))

    def _is_local_function(self, node: NodeProto) -> bool:
        return (node.domain, node.op_type) in self.local_functions

    def run(
        self,
        outputs: Optional[List[str]],
        feed_inputs: Dict[str, Any],
        intermediate: bool = False,
    ) -> Union[Dict[str, Any], List[Any]]:
        """
        Runs the model.
        It only works with numpy arrays.

        :param outputs: required outputs or None for all
        :param feed_inputs: inputs
        :param intermediate: returns all output instead of the last ones
        :return: outputs, as a list if return_all is False,
            as a dictionary if return_all is True
        """
        if outputs is None:
            outputs = self.output_names
        results: Dict[str, Any] = self.rt_inits_.copy()

        for k, v in self.rt_inits_.items():
            self._log(2, " +C %s: %s", k, v)
        for k, v in feed_inputs.items():
            self._log(2, " +I %s: %s", k, v)
            results[k] = v

        for node in self.rt_nodes_:
            self._log(1, "%s(%s) -> %s", node.op_type, node.input, node.output)
            for i in node.input:
                if i != "" and i not in results:
                    raise RuntimeError(
                        f"Unable to find input {i!r} in known results {sorted(results)}, "
                        f"self.rt_inits_ has {sorted(self.rt_inits_)}, "
                        f"feed_inputs has {sorted(feed_inputs)}."
                    )
            inputs = [(results[i] if i != "" else None) for i in node.input]
            if node.op_type == "If" and node.domain == "":
                outputs = self._run_if(node, inputs, results)
            elif self._is_local_function(node):
                outputs = self._run_local(node, inputs, results)
            else:
                outputs = self._run(node, inputs, results)
            for name, value in zip(node.output, outputs):
                if name == "":
                    continue
                self._log(2, " + %s: %s", name, value)  # type: ignore[arg-type]
                assert isinstance(name, str), f"unexpected type for name {type(name)}"
                results[name] = value

        if intermediate:
            return results
        output_names = self.output_names
        for name in output_names:
            if name == "":
                continue
            if name not in results:
                raise RuntimeError(
                    f"Unable to find output name {name!r} "
                    f"in {sorted(results)}, proto is\n{pretty_onnx(self.proto)}"
                )
        return [results[name] for name in output_names if name != ""]

    def _make_model_proto(
        self,
        nodes: Sequence[NodeProto],
        vinputs: Sequence[ValueInfoProto],
        voutputs: Sequence[ValueInfoProto],
    ) -> ModelProto:
        onx = oh.make_model(
            oh.make_graph(nodes, "-", vinputs, voutputs),
            ir_version=getattr(self.proto, "ir_version", self.ir_version),
            functions=getattr(self.proto, "functions", None),
        )
        del onx.opset_import[:]
        if hasattr(self.proto, "opset_import"):
            onx.opset_import.extend(self.proto.opset_import)
        elif self.opsets:
            if isinstance(self.opsets, int):
                onx.opset_import.append(oh.make_opsetid("", self.opsets))
            else:
                onx.opset_import.extend(
                    [oh.make_opsetid(k, v) for k, v in self.opsets.items()]
                )
        else:
            onx.opset_import.append(oh.make_opsetid("", onnx_opset_version()))

        return onx

    def _get_sess(
        self, node: NodeProto, inputs: List[Any]
    ) -> Tuple[ModelProto, _InferenceSession]:
        unique_names = set()
        vinputs = []
        for i, it in zip(node.input, inputs):
            if i == "" or i in unique_names:
                continue
            unique_names.add(i)
            value = oh.make_tensor_value_info(i, dtype_to_tensor_dtype(it.dtype), it.shape)
            vinputs.append(value)

        # no need to run shape inference
        voutputs = [oh.make_value_info(o, TypeProto()) for o in node.output]
        onx = self._make_model_proto([node], vinputs, voutputs)

        cls = (
            InferenceSessionForNumpy
            if any(isinstance(i, np.ndarray) for i in inputs)
            else InferenceSessionForTorch
        )
        try:
            sess = cls(onx, **self.session_kwargs)
        except (
            onnxruntime.capi.onnxruntime_pybind11_state.Fail,
            onnxruntime.capi.onnxruntime_pybind11_state.InvalidGraph,
            onnxruntime.capi.onnxruntime_pybind11_state.InvalidArgument,
        ) as e:
            raise RuntimeError(
                f"Unable to infer a session with inputs\n{string_type(inputs)}"
                f"\ndue to {e}\n{pretty_onnx(onx)}"
            ) from e
        return onx, sess

    def _get_sess_if(
        self, node: NodeProto, branch: str, inputs: List[Any], context: Dict[str, Any]
    ) -> Tuple[ModelProto, "OnnxruntimeEvaluator"]:
        unique_names = set()
        vinputs = []
        for i, it in zip(node.input, inputs):
            if i == "" or i in unique_names:
                continue
            unique_names.add(i)
            value = oh.make_tensor_value_info(i, dtype_to_tensor_dtype(it.dtype), it.shape)
            vinputs.append(value)

        for i, v in context.items():
            if i not in unique_names:
                unique_names.add(i)
                value = oh.make_tensor_value_info(i, dtype_to_tensor_dtype(v.dtype), v.shape)
                vinputs.append(value)

        for att in node.attribute:
            if att.name == branch:
                g = att.g

        voutputs = g.output

        onx = self._make_model_proto(g.node, vinputs, voutputs)
        sess = OnnxruntimeEvaluator(
            onx,
            local_functions=self.local_functions,
            verbose=self.verbose,
            ir_version=self.ir_version,
            opsets=self.opsets,
            **self.session_kwargs,
        )
        return onx, sess

    def _get_sess_local(
        self, node: NodeProto, inputs: List[Any]
    ) -> Tuple[FunctionProto, "OnnxruntimeEvaluator"]:
        ev = self.local_functions[node.domain, node.op_type]
        sess = OnnxruntimeEvaluator(
            ev,
            local_functions=self.local_functions,
            verbose=self.verbose,
            ir_version=self.ir_version,
            opsets=self.opsets,
            **self.session_kwargs,
        )
        return ev.proto, sess

    def _run(self, node: NodeProto, inputs: List[Any], results: Dict[str, Any]) -> List[Any]:
        """Runs a node."""
        types = [(None if a is None else (a.dtype, a.shape)) for a in inputs]
        key = (id(node), *types)
        if key in self._cache:
            sess = self._cache[key][1]
        else:
            onx, sess = self._get_sess(node, inputs)
            self._cache[key] = onx, sess

        feeds = dict(zip(node.input, inputs))
        if "" in feeds:
            feeds[""] = np.array([0], dtype=np.float32)

        assert hasattr(sess, "run"), f"Missing method run for type {type(sess)}"
        outputs = list(sess.run(None, feeds))
        assert isinstance(outputs, list), f"Unexpected type for outputs {type(outputs)}"
        return outputs

    def _run_if(
        self, node: NodeProto, inputs: List[Any], results: Dict[str, Any]
    ) -> List[Any]:
        """Runs a node if."""
        feeds = dict(zip(node.input, inputs))
        feeds.update(results)
        if feeds[node.input[0]]:
            name = "then_branch"
        else:
            name = "else_branch"

        key = (id(node), name)
        if key in self._cache:
            sess = self._cache[key][1]
        else:
            self._cache[key] = onx, sess = self._get_sess_if(node, name, inputs, results)

        assert hasattr(sess, "run"), f"Missing method run for type {type(sess)}"
        outputs = sess.run(None, feeds)
        assert isinstance(outputs, list), f"Unexpected type for outputs {type(outputs)}"
        return outputs

    def _run_local(
        self, node: NodeProto, inputs: List[Any], results: Dict[str, Any]
    ) -> List[Any]:
        """Runs a node."""
        types = [(None if a is None else (a.dtype, a.shape)) for a in inputs]
        key = (id(node), *types)
        if key in self._cache:
            sess = self._cache[key][1]
        else:
            onx, sess = self._get_sess_local(node, inputs)
            self._cache[key] = onx, sess

        replace = dict(zip(node.input, sess.input_names))
        assert len(node.input) == len(sess.input_names), (
            f"Input mismatch: input_names={sess.input_names}, "
            f"replace={replace}, "
            f"type(self.proto)={type(self.proto)}, and node=\n{node}"
        )
        feeds = {replace[i]: v for i, v in zip(node.input, inputs)}
        if "" in feeds:
            feeds[""] = np.array([0], dtype=np.float32)

        assert hasattr(sess, "run"), f"Missing method run for type {type(sess)}"
        outputs = sess.run(None, feeds)
        assert isinstance(outputs, list), f"Unexpected type for outputs {type(outputs)}"
        return outputs
