from typing import Optional
import onnx
import torch
from ...helpers.torch_helper import onnx_dtype_to_torch_dtype
from . import OpRun, OpRunValue


class Cast_6(OpRun):
    def __init__(self, node: onnx.NodeProto, version: Optional[int] = None):
        super().__init__(node, version)
        to = self.get_attribute_int(node, "to", 0)
        assert isinstance(to, int), f"Unexpected value for attribute to={to!r}"
        self.to = onnx_dtype_to_torch_dtype(to)

    def run(self, data: OpRunValue) -> OpRunValue:
        return OpRunValue(data.tensor.to(self.to))


class Concat_1(OpRun):
    def __init__(self, node: onnx.NodeProto, version: Optional[int] = None):
        super().__init__(node, version)
        axis = self.get_attribute_int(node, "axis", 0)
        assert isinstance(axis, int), f"Unexpected value for attribute axis={axis!r}"
        self.axis = axis

    def run(self, *data: OpRunValue) -> OpRunValue:
        return OpRunValue(torch.cat([t.tensor for t in data], axis=self.axis))


class Transpose_1(OpRun):
    def __init__(self, node: onnx.NodeProto, version: Optional[int] = None):
        super().__init__(node, version)
        self.perm = self.get_attribute_ints(node, "perm", None)

    def run(self, data: OpRunValue) -> OpRunValue:
        return OpRunValue(torch.permute(data.tensor, self.perm))
