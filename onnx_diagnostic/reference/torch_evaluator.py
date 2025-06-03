import functools
from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import onnx
import torch
from ..helpers.torch_helper import to_tensor
from ..torch_onnx.runtime_info import first_used_last_used, RuntimeValue
from . import torch_ops


@functools.lru_cache
def get_kernels() -> Dict[Tuple[str, str, int], type[torch_ops.OpRun]]:
    """
    Retrieves all the available kernels class :class:`TorchOnnxEvaluator`
    can use. The full list is the following.

    .. runpython::
        :showcode:

        from onnx_diagnostic.reference.torch_evaluator import get_kernels

        for k, v in sorted(get_kernels().items()):
            domain, name, version = k
            f = f"{name}({version})" if domain == "" else f"{name}[{domain}]({version})"
            add = " " * max(25 - len(f), 0)
            dd = " -- device dependent" if v.device_dependent() else ""
            print(f"{f}{add} -- {v.__name__}{dd}")
    """
    res = {}
    for _k, v in torch_ops.__dict__.items():
        if isinstance(v, type) and issubclass(v, torch_ops.OpRun) and "_" in v.__name__:
            name, version = v.__name__.split("_")
            domain = getattr(v, "domain", "")
            res[domain, name, int(version)] = v
    return res


class TorchOnnxEvaluator:
    """
    Torch evaluator for onnx models.
    The model does not stores the original proto it evaluates to avoid

    :param proto: a proto
    :param providers: where to run the model
    :param opsets: needed if proto is a graph
    :param functions: known local functions
    :param verbose: verbosity level

    The class holds the following attributes:

    * `providers`: providers
    * `default_device`: default torch device
    * `constants`: all initializers or constants
    * `kernels`: kernels
    * `runtime_info`: produced by :func:`first_used_last_used
      <onnx_diagnostic.torch_onnx.runtime_info.first_used_last_used>`
    * `last_used`: contains the list of intermediate results,
       to remove after every node execution,
       this avoid the memory to grow too much
    * `functions`: local functions

    The class is not multithreaded. `runtime_info` gets updated
    by the the class. The list of available kernels is returned by function
    :func:`onnx_diagnostic.reference.torch_evaluator.get_kernels`.
    Example:

    .. runpython::
        :showcode:

        import onnx
        import onnx.helper as oh
        import torch
        from onnx_diagnostic.helpers import string_type
        from onnx_diagnostic.reference import TorchOnnxEvaluator

        TFLOAT = onnx.TensorProto.FLOAT

        proto = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Sigmoid", ["Y"], ["sy"]),
                    oh.make_node("Mul", ["Y", "sy"], ["ysy"]),
                    oh.make_node("Mul", ["X", "ysy"], ["final"]),
                ],
                "-nd-",
                [
                    oh.make_tensor_value_info("X", TFLOAT, [1, "b", "c"]),
                    oh.make_tensor_value_info("Y", TFLOAT, ["a", "b", "c"]),
                ],
                [oh.make_tensor_value_info("final", TFLOAT, ["a", "b", "c"])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )

        sess = TorchOnnxEvaluator(proto)
        feeds = dict(X=torch.rand((4, 5)), Y=torch.rand((4, 5)))
        result = sess.run(None, feeds)
        print(string_type(result, with_shape=True, with_min_max=True))

    Adding ``verbose=1`` shows which kernels is executed:

    .. runpython::
        :showcode:

        import onnx
        import onnx.helper as oh
        import torch
        from onnx_diagnostic.helpers import string_type
        from onnx_diagnostic.reference import TorchOnnxEvaluator

        TFLOAT = onnx.TensorProto.FLOAT

        proto = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Sigmoid", ["Y"], ["sy"]),
                    oh.make_node("Mul", ["Y", "sy"], ["ysy"]),
                    oh.make_node("Mul", ["X", "ysy"], ["final"]),
                ],
                "-nd-",
                [
                    oh.make_tensor_value_info("X", TFLOAT, [1, "b", "c"]),
                    oh.make_tensor_value_info("Y", TFLOAT, ["a", "b", "c"]),
                ],
                [oh.make_tensor_value_info("final", TFLOAT, ["a", "b", "c"])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )

        sess = TorchOnnxEvaluator(proto, verbose=1)
        feeds = dict(X=torch.rand((4, 5)), Y=torch.rand((4, 5)))
        result = sess.run(None, feeds)
        print(string_type(result, with_shape=True, with_min_max=True))

    It also shows when a result is not needed anymore. In that case,
    it is deleted to free the memory it takes.
    The runtime can also execute the kernel the onnx model on CUDA.
    It follows the same logic as :class:`onnxruntime.InferenceSession`:
    ``providers=["CUDAExecutionProvider"]``.
    It is better in that case to move the input on CUDA. The class
    tries to move every weight on CUDA but tries to keep any tensor
    identified as a shape in CPU. Some bugs may remain as torch
    raises an exception when devices are expected to be the same.
    The runtime was validated with model :epkg:`arnir0/Tiny-LLM`.
    """

    class IO:
        "IO"

        def __init__(self, name: str, type: int, shape: Tuple[Union[str, int], ...]):
            self.name = name
            self.type = type
            self.shape = shape

    @classmethod
    def _on_cuda(cls, providers) -> int:
        if not providers:
            return -1
        for p in providers:
            if p == "CUDAExecutionProvider":
                return 0
            if isinstance(p, tuple) and p[0] == "CUDAExecutionProvider":
                return p[1]["device_id"]
        return -1

    def __init__(
        self,
        proto: Union[onnx.FunctionProto, onnx.GraphProto, onnx.ModelProto],
        providers: Tuple[str, ...] = ("CPUExecutionProvider",),
        opsets: Optional[Dict[str, int]] = None,
        local_functions: Optional[Dict[Tuple[str, str], "TorchOnnxEvaluator"]] = None,
        verbose: int = 0,
    ):
        self.providers = providers
        self.constants: Dict[str, torch.Tensor] = {}
        self.kernels: List[Optional[torch_ops.OpRun]] = []
        self.functions = local_functions.copy() if local_functions else {}
        self.CPU = torch.tensor([0]).to("cpu").device
        self.verbose = verbose
        dev = self._on_cuda(providers)
        if dev < 0:
            self.default_device = self.CPU
            self.CUDA = None
        else:
            self.CUDA = torch.tensor([0]).to(f"cuda:{dev}").device
            self.default_device = self.CUDA

        if isinstance(proto, str):
            proto = onnx.load(proto)
        if isinstance(proto, onnx.ModelProto):
            assert opsets is None, "proto is a model, opsets must be None in that case"
            assert not proto.graph.sparse_initializer, "sparse_initializer not support yet"
            self.opsets = {d.domain: d.version for d in proto.opset_import}
            for f in proto.functions:
                self.functions[f.domain, f.name] = self.__class__(
                    f,
                    providers=providers,
                    local_functions=self.functions,
                    verbose=self.verbose,
                )
            self._build_initializers(proto.graph.initializer)
            self._build_initializers(proto.graph.node)
            self._build_kernels(proto.graph.node)
            self.input_names = [i.name for i in proto.graph.input]
            self.output_names = [i.name for i in proto.graph.output]
            self._io_input_names = [
                self.IO(
                    name=i.name,
                    type=i.type.tensor_type.elem_type,
                    shape=tuple(
                        d.dim_param or d.dim_value for d in i.type.tensor_type.shape.dim
                    ),
                )
                for i in proto.graph.input
            ]
            self._io_output_names = [
                self.IO(
                    name=i.name,
                    type=i.type.tensor_type.elem_type,
                    shape=tuple(
                        d.dim_param or d.dim_value for d in i.type.tensor_type.shape.dim
                    ),
                )
                for i in proto.graph.output
            ]
        elif isinstance(proto, onnx.GraphProto):
            assert opsets, "opsets must be specified if proto is a graph"
            assert not proto.sparse_initializer, "sparse_initializer not support yet"
            self.opsets = opsets
            self._build_initializers(proto.initializer)
            self._build_initializers(proto.node)
            self._build_kernels(proto.node)
            self.input_names = [i.name for i in proto.input]
            self.output_names = [i.name for i in proto.output]
        elif isinstance(proto, onnx.FunctionProto):
            assert opsets is None, "proto is a model, opsets must be None in that case"
            self.opsets = {d.domain: d.version for d in proto.opset_import}
            self._build_initializers(proto.node)
            self._build_kernels(proto.node)
            self.input_names = list(proto.input)
            self.output_names = list(proto.output)
        else:
            raise TypeError(f"Unexpected type {type(proto)} for proto")

        self.runtime_info = first_used_last_used(proto, constant_as_initializer=True)
        self.last_used: List[List[str]] = [[] for _ in self.kernels]
        for name, info in self.runtime_info.items():
            assert isinstance(info.last_used, int) or info.is_input, (
                f"Missing field last_used in {info!r}, last_used={info.last_used!r}, "
                f"This may mean the node is unused and it should be removed."
            )
            if info.last_used is None:
                # Not used.
                self.last_used[0].append(name)
            elif not info.is_output and not info.is_initializer:
                self.last_used[info.last_used].append(name)

    def get_inputs(self):
        "Same API than onnxruntime."
        assert hasattr(self, "_io_input_names"), "Missing attribute '_io_input_names'."
        return self._io_input_names

    def get_outputs(self):
        "Same API than onnxruntime."
        assert hasattr(self, "_io_output_names"), "Missing attribute '_io_output_names'."
        return self._io_output_names

    @property
    def on_cuda(self) -> bool:
        "Tells if the default device is CUDA."
        return self.default_device == self.CUDA

    def _build_initializers(self, inits: Sequence[Union[onnx.NodeProto, onnx.TensorProto]]):
        for init in inits:
            if isinstance(init, onnx.TensorProto):
                self.constants[init.name] = to_tensor(init).to(self.default_device)
            elif (
                isinstance(init, onnx.NodeProto)
                and init.op_type == "Constant"
                and init.domain == ""
            ):
                value = None
                for att in init.attribute:
                    if att.name == "value":
                        value = to_tensor(att.t).to(self.default_device)
                    elif att.name == "value_floats":
                        value = torch.tensor(list(att.floats), dtype=torch.float32).to(
                            self.default_device
                        )
                assert value is not None, f"No attribute value in node {init}"
                self.constants[init.output[0]] = value

    def _build_kernels(self, nodes: Sequence[onnx.NodeProto]):
        kernels = get_kernels()
        self.kernels.clear()
        for node in nodes:
            if (node.domain, node.op_type) in self.functions:
                kernel = torch_ops.OpRunFunction(
                    self.functions[node.domain, node.op_type], node, self.opsets[node.domain]
                )
                self.kernels.append(kernel)
                continue

            if node.op_type == "Constant" and node.domain == "":
                # Treated as a constant.
                self.kernels.append(None)
                continue

            opset = self.opsets[node.domain]
            key = node.domain, node.op_type, opset
            while key not in kernels and opset > 0:
                opset -= 1
                key = node.domain, node.op_type, opset
            assert key in kernels, (
                f"Missing kernel for node type {node.op_type!r} from domain {node.domain!r}, "
                f"local functions={sorted(self.functions)}"
            )
            cls = kernels[key]
            ags = [self.default_device] if cls.device_dependent() else []
            kws = dict(parent=self) if cls.has_subgraphs() else {}
            kernel2 = cls(node, opset, *ags, **kws)
            self.kernels.append(kernel2)

    def run(
        self,
        outputs: Optional[List[str]],
        feeds: Union[Dict[str, torch.Tensor], Dict[str, np.ndarray]],
    ) -> Union[List[Optional[torch.Tensor]], List[Optional[np.ndarray]]]:
        """
        Runs the ONNX model.

        :param outputs: outputs required
        :param feeds: inputs
        :return: output tensors.
        """
        use_numpy = any(isinstance(t, np.ndarray) for t in feeds.values())
        if use_numpy:
            feeds = {k: torch.from_numpy(v) for k, v in feeds.items()}
        if outputs is None:
            outputs = self.output_names

        # sets constants
        for k, v in self.constants.items():
            r = self.runtime_info[k]
            if not r.has_value:
                r.set_value(
                    torch_ops.OpRunTensor(
                        v.to(self.CUDA) if not r.is_shape and self.on_cuda else v,
                        is_constant=True,
                        may_cpu=len(v.shape) == 1 and v.numel() < 8 and v.dtype == torch.int64,
                    )
                )
            if self.verbose:
                print(f"+C {r.name}: {r.string_type()}")

        # inputs
        for k, v in feeds.items():
            r = self.runtime_info[k]
            r.set_value(
                torch_ops.OpRunTensor(
                    v.to(self.CUDA) if not r.is_shape and self.on_cuda else v,
                    is_constant=False,
                    may_cpu=len(v.shape) == 1 and v.numel() < 8 and v.dtype == torch.int64,
                )
            )
            if self.verbose:
                print(f"+I {r.name}: {r.string_type()}")

        # node execution
        for it, kernel in enumerate(self.kernels):
            if kernel is not None:
                if self.verbose:
                    print(
                        f"{kernel.__class__.__name__}"
                        f"({', '.join(kernel.input)}) -> "
                        f"{', '.join(kernel.output)}"
                    )
                # kernel execution
                inputs = [(self.runtime_info[i].value if i else None) for i in kernel.input]
                if kernel.has_subgraphs():
                    res = kernel.run(*inputs, context=self.runtime_info)  # type: ignore[call-arg]
                else:
                    res = kernel.run(*inputs)
                if isinstance(res, tuple):
                    # outputs
                    assert all(isinstance(o, torch_ops.OpRunValue) for o in res), (
                        f"Unexpected output type {[type(o) for o in res]} "
                        f"for kernel {type(kernel)}."
                    )
                    for name, t in zip(kernel.output, res):
                        self.runtime_info[name].set_value(t)
                    if self.verbose:
                        for name in kernel.output:
                            print(f"+R {name}: {self.runtime_info[name].string_type()}")
                else:
                    assert isinstance(
                        res, torch_ops.OpRunValue
                    ), f"Unexpected output type {type(res)} for kernel {type(kernel)}."
                    self.runtime_info[kernel.output[0]].set_value(res)
                    if self.verbose:
                        print(
                            f"+R {kernel.output[0]}: "
                            f"{self.runtime_info[kernel.output[0]].string_type()}"
                        )

            # free intermediate results
            for name in self.last_used[it]:
                self.runtime_info[name].clean_value()
                if self.verbose:
                    print(f"- clean {name}")

        assert all(
            self.runtime_info[o].value is not None for o in outputs
        ), "Not implemented yet when one output is None."
        fres = [self.runtime_info[o].value.tensor for o in outputs]  # type: ignore[union-attr]
        if self.verbose:
            print(f"++ outputs {', '.join(outputs)}")

        # clean previous execution
        for k in feeds:
            self.runtime_info[k].clean_value()
            if self.verbose:
                print(f"- clean {k}")
        for o in outputs:
            self.runtime_info[o].clean_value()
            if self.verbose:
                print(f"- clean {o}")

        if use_numpy:
            return [None if a is None else a.detach().cpu().numpy() for a in fres]
        return fres

    def run_with_values(
        self,
        *args: Optional[torch_ops.OpRunTensor],
        context: Optional[Dict[str, RuntimeValue]] = None,
    ) -> Union[torch_ops.OpRunValue, Tuple[torch_ops.OpRunValue, ...]]:
        """
        Runs the ONNX model.

        :param args: inputs
        :param context: local context for the execution of subgraphs
        :return: output OpRunTensor
        """
        assert all(
            isinstance(a, torch_ops.OpRunValue) for a in args
        ), f"Unexpected type in args: {[type(a) for a in args]}"
        outputs = self.output_names
        context = context or {}

        # sets constants
        for k, v in self.constants.items():
            r = self.runtime_info[k]
            if not r.has_value:
                r.set_value(
                    torch_ops.OpRunTensor(
                        v.to(self.CUDA) if r.is_shape is False and self.on_cuda else v,
                        is_constant=True,
                        may_cpu=len(v.shape) == 1 and v.numel() < 8 and v.dtype == torch.int64,
                    )
                )

        # inputs
        for k, v in zip(self.input_names, args):
            r = self.runtime_info[k]
            r.set_value(
                torch_ops.OpRunTensor(None) if v is None else v.__class__(v.tensor_or_sequence)
            )

        # node execution
        for it, kernel in enumerate(self.kernels):
            if kernel is not None:
                # kernel execution
                inputs = [
                    (
                        (
                            self.runtime_info[i].value
                            if i in self.runtime_info
                            else context[i].value
                        )
                        if i
                        else None
                    )
                    for i in kernel.input
                ]
                res = kernel.run(*inputs)
                if isinstance(res, tuple):
                    # outputs
                    assert all(isinstance(o, torch_ops.OpRunTensor) for o in res), (
                        f"Unexpected output type {[type(o) for o in res]} "
                        f"for kernel {type(kernel)}."
                    )
                    for name, t in zip(kernel.output, res):
                        self.runtime_info[name].set_value(t)
                else:
                    assert isinstance(
                        res, torch_ops.OpRunValue
                    ), f"Unexpected output type {type(res)} for kernel {type(kernel)}."
                    self.runtime_info[kernel.output[0]].set_value(res)

            # free intermediate results
            for name in self.last_used[it]:
                self.runtime_info[name].clean_value()

        assert all(
            self.runtime_info[o].value is not None for o in outputs
        ), "Not implemented yet when one output is None."
        res2 = [self.runtime_info[o].value.copy() for o in outputs]  # type: ignore[assignment, union-attr]

        # clean previous execution
        for k in self.input_names:
            self.runtime_info[k].clean_value()
        for o in self.output_names:
            self.runtime_info[o].clean_value()

        return res2[0] if len(res2) == 1 else tuple(res2)  # type: ignore[index, return-value, arg-type]
