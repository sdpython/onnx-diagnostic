from typing import Optional
import onnx
import torch
from ...helpers.torch_helper import onnx_dtype_to_torch_dtype
from . import OpRun, OpRunValue


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
