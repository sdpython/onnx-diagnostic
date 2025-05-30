import functools
from typing import Dict, List, Optional, Sequence, Tuple, Union
import onnx
import torch
from ..helpers.torch_helper import to_tensor
from ..torch_onnx.runtime_info import first_used_last_used
from . import torch_ops


@functools.lru_cache
def get_kernels() -> Dict[Tuple[str, str, int], type[torch_ops.OpRun]]:
    """Retrieves all the available kernels."""
    res = {}
    for _k, v in torch_ops.__dict__.items():
        if isinstance(v, type) and issubclass(v, torch_ops.OpRun) and "_" in v.__name__:
            name, version = v.__name__.split("_")
            domain = getattr(v, "domain", "")
            res[domain, name, int(version)] = v
    return res


class TorchEvaluator:
    """
    Torch evaluator for onnx models.
    The model does not stores the original proto it evaluates to avoid

    :param proto: a proto
    :param providers: where to run the model
    :param opsets: needed if proto is a graph

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

    The class is not multithreaded. `runtime_info` gets updated
    by the the class.
    """

    def __init__(
        self,
        proto: Union[onnx.FunctionProto, onnx.GraphProto, onnx.ModelProto],
        providers: Tuple[str, ...] = ("CPUExecutionProvider",),
        opsets: Optional[Dict[str, int]] = None,
    ):
        self.providers = providers
        self.constants: Dict[str, torch.Tensor] = {}
        self.kernels: List[torch_ops.OpRun] = []
        self.CPU = torch.tensor([0]).to("cpu").device
        if "CUDAExecutionProvider" in providers:
            self.CUDA = torch.tensor([0]).to("cuda").device
            self.default_device = self.CUDA
        else:
            self.default_device = self.CPU

        if isinstance(proto, onnx.ModelProto):
            assert opsets is None, "proto is a model, opsets must be None in that case"
            assert not proto.graph.sparse_initializer, "sparse_initializer not support yet"
            self.opsets = {d.domain: d.version for d in proto.opset_import}
            self._build_initializers(proto.graph.initializer)
            self._build_initializers(proto.graph.node)
            self._build_kernels(proto.graph.node)
            self.input_names = [i.name for i in proto.graph.input]
            self.output_names = [i.name for i in proto.graph.output]
        elif isinstance(proto, onnx.GraphProto):
            assert opsets, "opsets must be specified if proto is a graph"
            assert not proto.sparse_initializer, "sparse_initializer not support yet"
            self.opsets = opsets
            self._build_initializers(proto)
            self._build_initializers(proto.node)
            self._build_kernels(proto.nodes)
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
            assert isinstance(info.last_used, int), f"Missing field last_used in {info!r}"
            if not info.is_output and not info.is_initializer:
                self.last_used[info.last_used].append(name)

    @property
    def on_cuda(self) -> bool:
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
                assert value is not None, f"No attribute value in node {init}"
                self.constants[init.output[0]] = value

    def _build_kernels(self, nodes: Sequence[onnx.NodeProto]):
        kernels = get_kernels()
        self.kernels.clear()
        for node in nodes:
            if node.op_type == "Constant" and node.domain == "":
                # Treated as a constant.
                self.kernels.append(None)
                continue
            opset = self.opsets[node.domain]
            key = node.domain, node.op_type, opset
            while key not in kernels:
                opset -= 1
                key = node.domain, node.op_type, opset
            assert (
                key in kernels
            ), f"Missing kernel for node type {node.op_type!r} from domain {node.domain!r}"
            self.kernels.append(kernels[key](node, opset))

    def run(
        self, outputs: Optional[List[str]], feeds: Dict[str, torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Runs the ONNX model.

        :param outputs: outputs required:
        :param feeds: inputs
        :return: output tensors.
        """
        if outputs is None:
            outputs = self.output_names

        # sets constants
        for k, v in self.constants.items():
            r = self.runtime_info[k]
            if not r.has_value:
                r.set_value(v.to(self.CUDA) if r.is_shape and self.on_cuda else v)

        # inputs
        for k, v in feeds.items():
            r = self.runtime_info[k]
            r.set_value(v.to(self.CUDA) if r.is_shape and self.on_cuda else v)

        # node execution
        for it, kernel in enumerate(self.kernels):
            if kernel is not None:
                # kernel execution
                inputs = [(self.runtime_info[i].value if i else None) for i in kernel.input]
                res = kernel.run(*inputs)
                if isinstance(res, tuple):
                    for name, t in zip(kernel.output, res):
                        self.runtime_info[name].set_value(t)
                else:
                    self.runtime_info[kernel.output[0]].set_value(res)

            # free intermediate results
            for name in self.last_used[it]:
                self.runtime_info[name].clean_value()

        # outputs
        res = [self.runtime_info[o].value for o in outputs]

        # clean previous execution
        for k in feeds:
            self.runtime_info[k].clean_value()
        for o in outputs:
            self.runtime_info[o].clean_value()

        return res
