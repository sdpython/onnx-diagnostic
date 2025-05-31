from typing import Optional
import onnx
from ...helpers.torch_helper import onnx_dtype_to_torch_dtype
from . import OpRun, OpRunValue


class ReduceOp(OpRun):
    def __init__(self, node: onnx.NodeProto, version: Optional[int] = None):
        super().__init__(node, version)
        self.keepdims = bool(self.get_attribute_int(node, "keepdims", 1))
        self.noop_with_empty_axes = bool(
            self.get_attribute_int(node, "noop_with_empty_axes", 0)
        )
        assert isinstance(
            self.keepdims, bool
        ), f"Unexpected value for attribute keepdims={self.keepdims!r}"
        assert isinstance(self.noop_with_empty_axes, bool), (
            f"Unexpected value for attribute "
            f"noop_with_empty_axes={self.noop_with_empty_axes!r}"
        )
        assert (
            not self.noop_with_empty_axes
        ), f"Not implemented with noop_with_empty_axes={self.noop_with_empty_axes}"
        # this is out of spec
        stash_type = self.get_attribute_int(node, "stash_type", None)
        self.stash_type = None if stash_type is None else onnx_dtype_to_torch_dtype(stash_type)


class ReduceMin_18(ReduceOp):
    """ReduceMin"""

    def run(self, x: OpRunValue, axes: OpRunValue) -> OpRunValue:
        assert self.stash_type is None, f"Not implemented with stash_type={self.stash_type}"
        taxes = axes.as_tuple_int
        if len(taxes) == 1:
            t = x.tensor.min(taxes[0], keepdim=self.keepdims)
            return OpRunValue(t.values)
        t = x.tensor
        for a in reversed(taxes):
            t = t.min(a, keepdim=self.keepdims).values
        return OpRunValue(t)
