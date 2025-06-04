from typing import Optional
import onnx
import torch
from ...helpers.torch_helper import onnx_dtype_to_torch_dtype
from . import OpRun, OpRunTensor


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


class ReduceOpAxes(ReduceOp):
    def __init__(self, node: onnx.NodeProto, version: Optional[int] = None):
        super().__init__(node, version)
        self.axes = self.get_attribute_ints(node, "axes") or []


class ReduceMax_18(ReduceOp):
    """ReduceMax"""

    def run(self, x: OpRunTensor, axes: Optional[OpRunTensor] = None) -> OpRunTensor:
        assert self.stash_type is None, f"Not implemented with stash_type={self.stash_type}"
        if axes is None:
            assert (
                not self.keepdims
            ), f"axes is Empty, keepdims={self.keepdims} for {self.__class__.__name__}"
            return OpRunTensor(torch.max(x.tensor).values)
        taxes = axes.as_tuple_int
        if len(taxes) == 1:
            t = x.tensor.max(taxes[0], keepdim=self.keepdims)
            return OpRunTensor(t.values)
        t = x.tensor
        for a in reversed(taxes):
            t = t.max(a, keepdim=self.keepdims).values
        return OpRunTensor(t)


class ReduceMean_18(ReduceOp):
    """ReduceMean"""

    def run(self, x: OpRunTensor, axes: Optional[OpRunTensor] = None) -> OpRunTensor:
        assert self.stash_type is None, f"Not implemented with stash_type={self.stash_type}"
        if axes is None:
            assert (
                not self.keepdims
            ), f"axes is Empty, keepdims={self.keepdims} for {self.__class__.__name__}"
            return OpRunTensor(torch.mean(x.tensor))
        taxes = axes.as_tuple_int
        if len(taxes) == 1:
            t = x.tensor.mean(taxes[0], keepdim=self.keepdims)
            return OpRunTensor(t)
        t = x.tensor.mean(taxes, keepdim=self.keepdims)
        return OpRunTensor(t)


class ReduceMin_17(ReduceOpAxes):
    """ReduceMin"""

    def run(self, x: OpRunTensor) -> OpRunTensor:
        assert self.stash_type is None, f"Not implemented with stash_type={self.stash_type}"
        axes = self.axes
        if not axes:
            assert (
                not self.keepdims
            ), f"axes is Empty, keepdims={self.keepdims} for {self.__class__.__name__}"
            return OpRunTensor(torch.min(x.tensor).values)
        taxes = tuple(axes)
        if len(taxes) == 1:
            t = x.tensor.min(taxes[0], keepdim=self.keepdims)
            return OpRunTensor(t.values)
        t = x.tensor
        for a in reversed(taxes):
            t = t.min(a, keepdim=self.keepdims).values
        return OpRunTensor(t)


class ReduceMin_18(ReduceOp):
    """ReduceMin"""

    def run(self, x: OpRunTensor, axes: Optional[OpRunTensor] = None) -> OpRunTensor:
        assert self.stash_type is None, f"Not implemented with stash_type={self.stash_type}"
        if axes is None:
            assert (
                not self.keepdims
            ), f"axes is Empty, keepdims={self.keepdims} for {self.__class__.__name__}"
            return OpRunTensor(torch.min(x.tensor).values)
        taxes = axes.as_tuple_int
        if len(taxes) == 1:
            t = x.tensor.min(taxes[0], keepdim=self.keepdims)
            return OpRunTensor(t.values)
        t = x.tensor
        for a in reversed(taxes):
            t = t.min(a, keepdim=self.keepdims).values
        return OpRunTensor(t)


class ReduceSum_13(ReduceOp):
    """ReduceSum"""

    def run(self, x: OpRunTensor, axes: Optional[OpRunTensor] = None) -> OpRunTensor:
        assert self.stash_type is None, f"Not implemented with stash_type={self.stash_type}"
        if axes is None:
            assert (
                not self.keepdims
            ), f"axes is Empty, keepdims={self.keepdims} for {self.__class__.__name__}"
            return OpRunTensor(torch.sum(x.tensor))
        taxes = axes.as_tuple_int
        if len(taxes) == 1:
            t = x.tensor.sum(taxes[0], keepdim=self.keepdims)
            return OpRunTensor(t)
        t = x.tensor.sum(taxes, keepdim=self.keepdims)
        return OpRunTensor(t)
