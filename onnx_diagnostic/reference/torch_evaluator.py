from typing import Dict, Optional, Tuple, Union
import onnx
import torch


class TorchEvaluator:
    """
    Torch evaluator for onnx models.
    The model does not stores the original proto it evaluates to avoid

    :param proto: a proto
    :param providers: where to run the model
    :param opsets: needed if proto is a graph
    """

    def __init__(
        self,
        proto: Union[onnx.FunctionProto, onnx.GraphProto, onnx.ModelProto],
        providers: Tuple[str, ...] = ("CPUExecutionProvider",),
        opsets: Optional[Dict[str, int]] = None,
    ):
        self.providers = providers
        self.constants = {}

        if isinstance(proto, onnx.ModelProto):
            assert opsets is None, "proto is a model, opsets must be None in that case"
            assert not proto.graph.sparse_initializer, "sparse_initializer not support yet"
            self._build_initializers(proto.graph.initializers)
            self._build_initializers(proto.graph.nodes)
            self._build_kernels(proto.graph.node)
            self.input_names = [i.name for i in proto.graph.input]
            self.output_names = [i.name for i in proto.graph.output]
            self.opsets = {d.domain: d.version for d in proto.opset_import}
        elif isinstance(proto, onnx.GraphProto):
            assert opsets, "opsets must be specified if proto is a graph"
            assert not proto.sparse_initializer, "sparse_initializer not support yet"
            self._build_initializers(proto)
            self._build_initializers(proto.node)
            self._build_kernels(proto.nodes)
            self.input_names = [i.name for i in proto.input]
            self.output_names = [i.name for i in proto.output]
            self.opsets = opsets
        elif isinstance(proto, onnx.FunctionProto):
            assert opsets is None, "proto is a model, opsets must be None in that case"
            self._build_initializers(proto.node)
            self._build_kernels(proto.node)
            self.input_names = [i.name for i in proto.graph.input]
            self.output_names = [i.name for i in proto.graph.output]
            self.opsets = {d.domain: d.version for d in proto.opset_import}
        else:
            raise TypeError(f"Unexpected type {type(proto)} for proto")
