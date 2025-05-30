from typing import Optional
import onnx
import torch
from . import OpRun, OpRunValue


class Shape_15(OpRun):
    def __init__(self, node: onnx.NodeProto, version: Optional[int] = None):
        super().__init__(node, version)
        self.start = self.get_attribute_int(node, "start", 0)
        self.end = self.get_attribute_int(node, "end", None)

    def run(self, data: OpRunValue) -> OpRunValue:
        shape = data.shape
        sh = shape[self.start :] if self.end is None else shape[self.start : self.end]
        return OpRunValue(torch.tensor(sh, dtype=torch.int64), is_constant=True)


class Squeeze_13(OpRun):
    "Squeeze"

    def run(self, data: OpRunValue, axes: OpRunValue) -> OpRunValue:
        return OpRunValue(data.tensor.squeeze(axes.as_tuple_int))
