from typing import Optional
import onnx
import torch
from ...helpers.torch_helper import onnx_dtype_to_torch_dtype
from . import OpRun, OpRunValue


class LayerNormalization_17(OpRun):
    "LayerNormalization"

    def __init__(self, node: onnx.NodeProto, version: Optional[int] = None):
        super().__init__(node, version)
        self.axis = self.get_attribute_int(node, "axis", -1)
        self.epsilon = self.get_attribute_float(node, "epsilon", 1e-5)
        self.stash_type = onnx_dtype_to_torch_dtype(
            self.get_attribute_int(node, "stash_type", onnx.TensorProto.FLOAT)
        )
        self.compute_std = len(node.output) > 1

    def run(self, x, scale, bias=None):
        original_dtype = x.dtype
        xt = x.tensor.to(self.stash_type)
        res = torch.nn.functional.layer_norm(
            xt,
            xt.shape[self.axis :],
            weight=scale.tensor,
            bias=None if bias is None else bias.tensor,
            eps=self.epsilon,
        )
        if not self.compute_std:
            return OpRunValue(res.to(original_dtype))
        axes = tuple(range(len(xt.shape)))[self.axis :]
        mean, var = torch.var(xt, dim=axes, keepdim=False)
        x_inv_std_dev = torch.reciprocal(torch.sqrt(var + self.epsilon))
        return OpRunValue(res.to(original_dtype)), OpRunValue(mean), OpRunValue(x_inv_std_dev)


class Softmax_13(OpRun):
    "Softmax"

    def __init__(self, node: onnx.NodeProto, version: Optional[int] = None):
        super().__init__(node, version)
        self.axis = self.get_attribute_int(node, "axis", -1)
        assert isinstance(self.axis, int), f"Unexpected value for attribute axis={self.axis!r}"
        # this is out of spec
        stash_type = self.get_attribute_int(node, "stash_type", None)
        self.stash_type = None if stash_type is None else onnx_dtype_to_torch_dtype(stash_type)

    def run(self, data: OpRunValue) -> OpRunValue:
        return OpRunValue(
            torch.nn.functional.softmax(data.tensor, dim=self.axis, dtype=self.stash_type)
        )


class Tanh_6(OpRun):
    "Tanh"

    def run(self, data: OpRunValue) -> OpRunValue:
        return OpRunValue(torch.nn.functional.tanh(data.tensor))
