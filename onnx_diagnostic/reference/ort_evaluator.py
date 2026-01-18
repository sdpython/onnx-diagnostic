from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union
import numpy as np
from onnx import (
    AttributeProto,
    GraphProto,
    FunctionProto,
    ModelProto,
    NodeProto,
    TensorProto,
    TypeProto,
    ValueInfoProto,
    helper as oh,
    load,
    save as onnx_save,
    shape_inference as shi,
)
from onnx.defs import onnx_opset_version
import onnxruntime
from ..helpers import string_type
from ..helpers.onnx_helper import (
    get_hidden_inputs,
    dtype_to_tensor_dtype,
    np_dtype_to_tensor_dtype,
    to_array_extended,
    pretty_onnx,
)
from ..helpers.torch_helper import onnx_dtype_to_torch_dtype, torch_dtype_to_onnx_dtype
from ..helpers.ort_session import (
    InferenceSessionForTorch,
    InferenceSessionForNumpy,
    _InferenceSession,
)
from ..helpers.torch_helper import to_tensor
from .report_results_comparison import ReportResultComparison
from .evaluator import ExtendedReferenceEvaluator

PROTO = (FunctionProto, ModelProto, GraphProto, NodeProto)
Proto = Union[FunctionProto, ModelProto, GraphProto, NodeProto]


class OnnxList(list):
    """Defines a list for the runtime."""

    def __init__(self, itype: Union[list, int]):
        super().__init__()
        if isinstance(itype, int):
            self.itype = itype
            self.dtype = onnx_dtype_to_torch_dtype(itype)
        else:
            assert itype, "The list cannot be created with an empty list."
            self.itype = (
                np_dtype_to_tensor_dtype(itype[0].dtype)
                if isinstance(itype[0], np.ndarray)
                else torch_dtype_to_onnx_dtype(itype[0].dtype)
            )
            self.extend(itype)
            self.dtype = itype[0].dtype
        self.shape = "OnnxList"

    def get_device(self):
        "Returns the device of the first tensor."
        assert len(self) > 0, "Cannot access the device for an empty list."
        return self[0].get_device() if hasattr(self[0], "get_device") else -1

    def numpy(self):
        "Creates a new list with all tensors on numpy or self it is already the case."
        if all(isinstance(v, np.ndarray) for v in self):
            return self
        return OnnxList([v.detach().cpu().numpy() for v in self])

    def to(self, tensor_like) -> "OnnxList":
        "Creates a new list with all tensors on numpy or pytorch depending on `tensor_like`."
        if isinstance(tensor_like, np.ndarray):
            return self
        import torch

        return OnnxList(
            [
                torch.from_numpy(t).to(tensor_like.device) if isinstance(t, np.ndarray) else t
                for t in self
            ]
        )

    def clone(self) -> "OnnxList":
        "Clone (torch)."
        return OnnxList([t.clone() for t in self]) if len(self) > 0 else OnnxList(self.itype)


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
    :param whole: if True, do not split node by node
    :param torch_or_numpy: force the use of one of them, True for torch,
        False for numpy, None to let the class choose
    :param dump_onnx_model: dumps the temporary onnx model created if whole is True
    :param function_kwargs: a FunctionProto may have parameters,
        this contains the values of them
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
        whole: bool = False,
        torch_or_numpy: Optional[bool] = None,
        function_kwargs: Optional[Dict[str, Any]] = None,
        dump_onnx_model: Optional[str] = None,
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
        assert (
            whole or not dump_onnx_model
        ), f"whole must be True for dump_onnx_model={dump_onnx_model!r}"

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
        self.to_tensor_or_array = to_array_extended if not torch_or_numpy else to_tensor
        self.function_kwargs = function_kwargs
        self.dump_onnx_model = dump_onnx_model

        self.verbose = verbose
        self.torch_or_numpy = torch_or_numpy
        self.sess_: Optional[_InferenceSession] = None
        if whole:
            self.nodes: Optional[List[NodeProto]] = None
            self.rt_inits_: Optional[Dict[str, Any]] = None
            self.rt_nodes_: Optional[List[NodeProto]] = None
        else:
            self.nodes = (
                [self.proto]
                if isinstance(self.proto, NodeProto)
                else (
                    list(
                        self.proto.graph.node
                        if hasattr(self.proto, "graph")
                        else self.proto.node
                    )
                )
            )
            self.rt_inits_ = (
                {
                    init.name: self.to_tensor_or_array(init)
                    for init in self.proto.graph.initializer
                }
                if hasattr(self.proto, "graph")
                else {}
            )
            self.rt_nodes_ = self.nodes.copy()

        self.local_functions: Dict[Tuple[str, str], "OnnxruntimeEvaluator"] = (  # noqa: UP037
            {(f.domain, f.name): self.__class__(f) for f in self.proto.functions}
            if hasattr(self.proto, "functions")
            else {}
        )
        if local_functions:
            self.local_functions.update(local_functions)
        self.garbage_collector = self._build_garbage_collector() if self.rt_nodes_ else {}

    @property
    def input_names(self) -> List[str]:
        "Returns input names."
        assert self.proto, "self.proto is empty"
        if isinstance(self.proto, NodeProto):
            assert isinstance(
                self.nodes, list
            ), f"Unexpected type {type(self.nodes)} for self.nodes"
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
        assert self.proto, "self.proto is empty"
        if isinstance(self.proto, NodeProto):
            assert isinstance(
                self.nodes, list
            ), f"Unexpected type {type(self.nodes)} for self.nodes"
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
        if isinstance(a, OnnxList):
            return string_type(a)
        device = f"D{a.get_device()}:" if hasattr(a, "detach") else ""
        if hasattr(a, "shape"):
            prefix = "A:" if hasattr(a, "astype") else "T:"
            if self.verbose < 4:  # noqa: PLR2004
                return f"{prefix}{device}{a.dtype}:{a.shape} in [{a.min()}, {a.max()}]"
            elements = a.ravel().tolist()
            if len(elements) > 10:  # noqa: PLR2004
                elements = elements[:10]
                return f"{prefix}{device}{a.dtype}:{a.shape}:{','.join(map(str, elements))}..."
            return f"{prefix}{device}{a.dtype}:{a.shape}:{elements}"
        if hasattr(a, "append"):
            return ", ".join(map(self._log_arg, a))
        return a

    def _log(self, level: int, pattern: str, *args: Any) -> None:
        if level < self.verbose:
            new_args = [self._log_arg(a) for a in args]
            print(pattern % tuple(new_args))

    def _is_local_function(self, node: NodeProto) -> bool:
        return (node.domain, node.op_type) in self.local_functions

    def _run_init(self, feed_inputs):
        if self.sess_ is None:
            assert self.proto, "self.proto is empty"
            _, self.sess_ = self._get_sess(self.proto, list(feed_inputs.values()))
        return self.sess_

    def run(
        self,
        outputs: Optional[List[str]],
        feed_inputs: Dict[str, Any],
        intermediate: bool = False,
        report_cmp: Optional[ReportResultComparison] = None,
    ) -> Union[Dict[str, Any], List[Any]]:
        """
        Runs the model.
        It only works with numpy arrays.

        :param outputs: required outputs or None for all
        :param feed_inputs: inputs
        :param intermediate: returns all output instead of the last ones
        :param report_cmp: used as a reference,
            every intermediate results is compare to every existing one,
            if not empty, it is an instance of
            :class:`onnx_diagnostic.reference.ReportResultComparison`
        :return: outputs, as a list if return_all is False,
            as a dictionary if return_all is True
        """
        if self.rt_nodes_ is None:
            # runs a whole
            self._run_init(feed_inputs)
            assert self.sess_, "mypy not happy"
            return self.sess_.run(outputs, feed_inputs)
        if outputs is None:
            outputs = self.output_names
        results: Dict[str, Any] = (self.rt_inits_ or {}).copy()

        for k, v in results.items():
            self._log(2, " +C %s: %s", k, v)
        for k, v in feed_inputs.items():
            assert not isinstance(v, str), f"Unexpected type str for {k!r}"
            self._log(2, " +I %s: %s", k, v)
            results[k] = v

        for i_node, node in enumerate(self.rt_nodes_ or []):
            self._log(1, "%s(%s) -> %s", node.op_type, node.input, node.output)
            for i in node.input:
                if i != "" and i not in results:
                    raise RuntimeError(
                        f"Unable to find input {i!r} in known results {sorted(results)}, "
                        f"self.rt_inits_ has {sorted((self.rt_inits_ or {}))}, "
                        f"feed_inputs has {sorted(feed_inputs)}."
                    )
            inputs = [(results[i] if i != "" else None) for i in node.input]
            if node.op_type == "If" and node.domain == "":
                outputs = self._run_if(node, inputs, results)
            elif node.op_type in {"Scan", "Loop"} and node.domain == "":
                outputs = self._run_scan_or_loop(node, inputs, results)
            elif self._is_local_function(node):
                outputs = self._run_local(node, inputs, results)
            else:
                outputs = self._run(node, inputs, results)
            node_output = [o for o in node.output if o]
            assert len(node_output) == len(
                outputs
            ), f"Length mismatch between node output={node.output} and outputs={outputs}"
            for name, value in zip(node_output, outputs):
                self._log(2, " + %s: %s", name, value)  # type: ignore[arg-type]
                assert isinstance(name, str), f"unexpected type for name {type(name)}"
                results[name] = value
            if report_cmp:
                reported = report_cmp.report(dict(zip(node.output, outputs)))
                if self.verbose > 1:
                    print(f"  -- report {len(reported)} comparisons")
            if not intermediate:
                self._clean_unused_inplace(i_node, node, results)

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

    def _build_garbage_collector(self) -> Dict[str, int]:
        """
        Memorizes the results not needed anymore for every node.
        Returns a dictionary with the last node using the results.
        """
        needed = {}
        for i, node in enumerate(self.rt_nodes_ or []):
            for name in node.input:
                needed[name] = i
            if node.op_type in {"Scan", "If", "Loop"}:
                hidden = self._get_hidden_node_inputs(node)
                for name in hidden:
                    needed[name] = i
        if isinstance(self.proto, ModelProto):
            for o in self.proto.graph.output:
                needed[o.name] = len(self.rt_nodes_ or [])
        elif isinstance(self.proto, GraphProto):
            for o in self.proto.output:
                needed[o.name] = len(self.rt_nodes_ or [])
        elif isinstance(self.proto, FunctionProto):
            for o in self.proto.output:
                needed[o] = len(self.rt_nodes_ or [])
        return needed

    def _clean_unused_inplace(self, i_node: int, node: NodeProto, results: Dict[str, Any]):
        """
        Cleans all results not needed anymore. Some models requires to clean the memory
        to be able to run.
        """
        if not self.garbage_collector:
            return
        for name in node.input:
            if self.garbage_collector[name] == i_node and name in results:
                if self.verbose:
                    t = results[name]
                    print(f" - deletes: {name} - {t.dtype}:{t.shape}")
                del results[name]
        if node.op_type in {"Scan", "If", "Loop"}:
            hidden = self._get_hidden_node_inputs(node)
            for name in hidden:
                if self.garbage_collector[name] == i_node and name in results:
                    if self.verbose:
                        t = results[name]
                        print(f" - deletes: {name} - {t.dtype}:{t.shape}")
                    del results[name]

    def _make_model_proto(
        self,
        nodes: Sequence[NodeProto],
        vinputs: Sequence[ValueInfoProto],
        voutputs: Sequence[ValueInfoProto],
        functions: Optional[Sequence[FunctionProto]] = None,
    ) -> ModelProto:
        onx = oh.make_model(
            oh.make_graph(nodes, "-", vinputs, voutputs),
            ir_version=getattr(self.proto, "ir_version", self.ir_version),
            functions=[*getattr(self.proto, "functions", []), *(functions or [])],
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
        opsets = {d.domain: d.version for d in onx.opset_import}
        add = {}
        for node in self.enumerate_nodes(onx.graph.node):
            if node.domain and node.domain not in opsets and node.domain not in add:
                add[node.domain] = 1
        onx.opset_import.extend([oh.make_opsetid(k, v) for k, v in add.items()])

        # That helps fixing bugs.
        onx = shi.infer_shapes(onx)
        return onx

    def _make_model_outputs(
        self, node: NodeProto, inputs: List[ValueInfoProto]
    ) -> Tuple[List[NodeProto], List[ValueInfoProto]]:
        return [], [oh.make_value_info(o, TypeProto()) for o in node.output if o]

    def enumerate_nodes(self, nodes: List[NodeProto]) -> Iterator[NodeProto]:
        "Enumerates nodes recursively."
        for node in nodes:
            if node.op_type in {"Scan", "If", "Loop"}:
                for att in node.attribute:
                    if att.type == AttributeProto.GRAPH:
                        yield from self.enumerate_nodes(att.g.node)
            yield node

    @classmethod
    def _get_hidden_node_inputs(cls, node: NodeProto) -> Set[str]:
        """Calls multiple get_hidden_inputs on every attribute."""
        if node.op_type not in {"Loop", "Scan", "If"}:
            return set()
        hidden = set()
        for att in node.attribute:
            if att.type == AttributeProto.GRAPH:
                hidden |= get_hidden_inputs(att.g)
        return hidden - (hidden & set(node.input))

    def _get_sess(
        self, node: Union[ModelProto, NodeProto], inputs: List[Any]
    ) -> Tuple[ModelProto, _InferenceSession]:
        on_cpu = None
        if isinstance(node, ModelProto):
            onx = node
        else:
            functions = []
            if isinstance(node, FunctionProto):
                functions.append(node)
                node = oh.make_node(
                    node.name,
                    list(node.input),
                    list(node.output),
                    domain=node.domain,
                    **(self.function_kwargs or {}),
                )
            assert isinstance(node, NodeProto), f"Unexpected type {type(node)} for node"
            if node.op_type == "Constant" and node.domain == "":
                # We force the type to be a boolean.
                ref = ExtendedReferenceEvaluator(node)
                cst = ref.run(None, {})[0]
                vinputs: List[ValueInfoProto] = []
                voutputs = [
                    oh.make_tensor_value_info(
                        node.output[0], dtype_to_tensor_dtype(cst.dtype), cst.shape
                    )
                ]
                prenodes = []  # type: ignore[var-annotated]
            elif node.op_type == "ConcatFromSequence" and node.domain == "":
                # We force the type to be a boolean.
                vinputs = [
                    oh.make_value_info(
                        node.input[0],
                        type_proto=oh.make_sequence_type_proto(
                            oh.make_tensor_type_proto(elem_type=inputs[0].itype, shape=None)
                        ),
                    )
                ]
                voutputs = [oh.make_tensor_value_info(node.output[0], inputs[0].itype, None)]
                prenodes = []  # type: ignore[var-annotated]
            else:
                unique_names = set()
                vinputs = []
                for i, it in zip(node.input, inputs):
                    if i == "" or i in unique_names:
                        continue
                    unique_names.add(i)
                    value = oh.make_tensor_value_info(
                        i, dtype_to_tensor_dtype(it.dtype), it.shape
                    )
                    vinputs.append(value)

                # no need to run shape inference
                prenodes, voutputs = self._make_model_outputs(node, vinputs)

            onx = self._make_model_proto(
                [*prenodes, node], vinputs, voutputs, functions=functions
            )
            if node.op_type in {"Shape", "Size"}:
                on_cpu = True

        if self.dump_onnx_model:
            onnx_save(
                onx, self.dump_onnx_model, save_as_external_data=len(onx.graph.node) > 100
            )
        cls = (
            InferenceSessionForNumpy
            if any(isinstance(i, np.ndarray) for i in inputs)
            and (not isinstance(self.torch_or_numpy, bool) or not self.torch_or_numpy)
            else InferenceSessionForTorch
        )
        if (
            "providers" not in self.session_kwargs or not self.session_kwargs["providers"]
        ) and any(hasattr(t, "is_cuda") and t.is_cuda for t in inputs):
            sess_kwargs = self.session_kwargs.copy()
            sess_kwargs["providers"] = ["CUDAExecutionProvider"]
        else:
            sess_kwargs = self.session_kwargs or {}
        if on_cpu and "CUDAExecutionProvider" in (sess_kwargs.get("providers", []) or []):
            sess_kwargs["cpu_outputs"] = True
        try:
            sess = cls(onx, **sess_kwargs)
        except (
            onnxruntime.capi.onnxruntime_pybind11_state.Fail,
            onnxruntime.capi.onnxruntime_pybind11_state.InvalidGraph,
            onnxruntime.capi.onnxruntime_pybind11_state.InvalidArgument,
        ) as e:
            onnx_save(onx, "_debug_OnnxruntimeEvaluator_last_failure.onnx")
            raise RuntimeError(
                f"Unable to infer a session with inputs\n{string_type(inputs)}"
                f"\ndue to {e}\n{pretty_onnx(onx)}"
            ) from e
        return onx, sess

    def _get_sess_init_subgraph(
        self, node: NodeProto, inputs: List[Any], context: Dict[str, Any], g: GraphProto
    ) -> List[Any]:
        unique_names = set()
        vinputs = []
        for i, it in zip(node.input, inputs):
            if i == "" or i in unique_names:
                continue
            unique_names.add(i)
            if isinstance(it, OnnxList):
                value = oh.make_value_info(
                    i,
                    type_proto=oh.make_sequence_type_proto(
                        oh.make_tensor_type_proto(
                            elem_type=dtype_to_tensor_dtype(it.dtype), shape=None
                        )
                    ),
                )
            else:
                value = oh.make_tensor_value_info(i, dtype_to_tensor_dtype(it.dtype), it.shape)
            vinputs.append(value)

        reduced_set = get_hidden_inputs(g)
        for i, v in context.items():
            if i in reduced_set and i not in unique_names:
                unique_names.add(i)
                value = oh.make_tensor_value_info(i, dtype_to_tensor_dtype(v.dtype), v.shape)
                vinputs.append(value)
        assert len(reduced_set & set(context)) == len(reduced_set), (
            f"Missing hidden inputs {sorted(reduced_set)} from context={sorted(context)} "
            f"(len(inputs)={len([i for i in inputs if i])}) for node {pretty_onnx(node)}"
        )
        return vinputs

    def _get_sess_if(
        self, node: NodeProto, branch: str, inputs: List[Any], context: Dict[str, Any]
    ) -> Tuple[ModelProto, "OnnxruntimeEvaluator"]:
        g = None
        for att in node.attribute:
            if att.name == branch:
                g = att.g
        assert g, f"Missing attribute {branch!r}"
        vinputs = self._get_sess_init_subgraph(node, inputs, context, g)

        voutputs = g.output

        identities = [
            oh.make_node("Identity", [iname], [ginput.name])
            for iname, ginput in zip(node.input, g.input)
        ]

        onx = self._make_model_proto([*identities, *g.node], vinputs, voutputs)
        sess = OnnxruntimeEvaluator(
            onx,
            local_functions=self.local_functions,
            verbose=self.verbose,
            ir_version=self.ir_version,
            opsets=self.opsets,
            torch_or_numpy=self.torch_or_numpy,
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
            torch_or_numpy=self.torch_or_numpy,
            **self.session_kwargs,
        )
        return ev.proto, sess

    def _run(self, node: NodeProto, inputs: List[Any], results: Dict[str, Any]) -> List[Any]:
        """Runs a node."""
        if node.op_type[0] == "S":
            if node.op_type == "SequenceEmpty":
                dtype = TensorProto.FLOAT
                for att in node.attribute:
                    if att.name == "dtype":
                        dtype = att.i
                return [OnnxList(itype=dtype)]

        types = [(None if a is None else (a.dtype, a.shape)) for a in inputs]
        key = (id(node), *types)
        if key in self._cache:
            sess = self._cache[key][1]
        else:
            onx, sess = self._get_sess(node, inputs)
            self._cache[key] = onx, sess

        feeds = {}
        for i, val in zip(node.input, inputs):
            if i == "":
                assert (
                    val is None
                ), f"input name={i!r} but val={string_type(val, with_shape=True)}"
                continue
            feeds[i] = val
        assert hasattr(sess, "run"), f"Missing method run for type {type(sess)}"

        if node.op_type[0] == "C":
            if node.op_type == "ConcatFromSequence":
                res = sess.sess.run(None, self.feeds_to_numpy(feeds))  # type: ignore[union-attr]
                if isinstance(inputs[0][0], np.ndarray):
                    return list(res)
                import torch

                return [torch.from_numpy(r).to(inputs[0][0].device) for r in res]

        outputs = list(sess.run(None, feeds))
        assert isinstance(outputs, list), f"Unexpected type for outputs {type(outputs)}"
        assert not any(type(v) is list for v in outputs), (
            f"One output type is a list, this should not be allowed, "
            f"node.op_type={node.op_type}, feeds={string_type(feeds, with_shape=True)}"
        )
        return outputs

    def _run_if(
        self, node: NodeProto, inputs: List[Any], results: Dict[str, Any]
    ) -> List[Any]:
        """Runs a node If."""
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
            self._cache[key] = _onx, sess = self._get_sess_if(node, name, inputs, results)

        assert hasattr(sess, "run"), f"Missing method run for type {type(sess)}"
        feeds = {name: results[name] for name in sess.input_names}
        outputs = sess.run(None, feeds)
        assert isinstance(outputs, list), f"Unexpected type for outputs {type(outputs)}"
        return outputs

    def _get_sess_scan_or_loop(
        self, node: NodeProto, branch: str, inputs: List[Any], context: Dict[str, Any]
    ) -> Tuple[ModelProto, "OnnxruntimeEvaluator"]:
        g = None
        for att in node.attribute:
            if att.name == branch:
                g = att.g
        assert g, f"Missing attribute {branch!r}"
        vinputs = self._get_sess_init_subgraph(node, inputs, context, g)

        begin = 0 if node.op_type == "Scan" else 1
        voutputs = []
        for name, _goutput in zip(node.output, g.output[begin:]):
            v = ValueInfoProto()
            # v.ParseFromString(goutput.SerializeToString())
            v.name = name
            voutputs.append(v)

        # identities = []
        # for iname, ginput in zip(node.input, g.input):
        #    identities.append(oh.make_node("Identity", [iname], [ginput.name]))

        onx = self._make_model_proto([node], vinputs, voutputs)
        sess = OnnxruntimeEvaluator(
            onx,
            local_functions=self.local_functions,
            verbose=self.verbose,
            ir_version=self.ir_version,
            opsets=self.opsets,
            torch_or_numpy=self.torch_or_numpy,
            whole=True,
            **self.session_kwargs,
        )
        return onx, sess

    def feeds_to_numpy(self, feeds):
        new_feeds = {}
        for k, v in feeds.items():
            if hasattr(v, "detach"):
                new_feeds[k] = v.detach().cpu().numpy()
            elif isinstance(v, OnnxList):
                new_feeds[k] = v.numpy()
            else:
                new_feeds[k] = v
        return new_feeds

    def _run_scan_or_loop(
        self, node: NodeProto, inputs: List[Any], results: Dict[str, Any]
    ) -> List[Any]:
        """Runs a node Scan."""
        assert not any(type(i) is list for i in inputs), (
            f"One input is a list but it should an OnnxList, "
            f"node.op_type={node.op_type!r}, node.input={node.input}, "
            f"inputs={string_type(inputs, with_shape=True)}"
        )
        feeds = dict(zip(node.input, inputs))
        feeds.update(results)
        name = "body"
        key = (id(node), name)
        if key in self._cache:
            sess = self._cache[key][1]
        else:
            self._cache[key] = _onx, sess = self._get_sess_scan_or_loop(
                node, name, inputs, results
            )

        assert hasattr(sess, "run"), f"Missing method run for type {type(sess)}"
        feeds = {name: results[name] for name in sess.input_names}
        if node.op_type == "Loop" and any(isinstance(v, OnnxList) for v in feeds.values()):
            # This operator uses sequence. onnxruntime does not play well with sequence.
            sess._run_init(feeds)  # type: ignore[union-attr]
            outputs = sess.sess_.sess.run(None, self.feeds_to_numpy(feeds))  # type: ignore[union-attr]
            return [
                (OnnxList(v).to(feeds[node.input[0]]) if isinstance(v, list) else v)
                for v in outputs
            ]

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
