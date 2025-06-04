from typing import Optional
import onnx
import torch
from ...helpers.torch_helper import onnx_dtype_to_torch_dtype
from . import OpRun, OpRunTensor


class Cast_6(OpRun):
    "Cast"

    def __init__(self, node: onnx.NodeProto, version: Optional[int] = None):
        super().__init__(node, version)
        to = self.get_attribute_int(node, "to", 0)
        assert isinstance(to, int), f"Unexpected value for attribute to={to!r}"
        self.to = onnx_dtype_to_torch_dtype(to)
        self.saturate = self.get_attribute_int(node, "saturate", 1)
        assert self.saturate == 1, f"saturate={self.saturate} not implemented for Cast"

    def run(self, data: OpRunTensor) -> OpRunTensor:
        return OpRunTensor(data.tensor.to(self.to))


class CastLike_15(OpRun):
    "Cast"

    def __init__(self, node: onnx.NodeProto, version: Optional[int] = None):
        super().__init__(node, version)
        self.saturate = self.get_attribute_int(node, "saturate", 1)
        assert self.saturate == 1, f"saturate={self.saturate} not implemented for CastLike"

    def run(self, data: OpRunTensor, like: OpRunTensor) -> OpRunTensor:
        return OpRunTensor(data.tensor.to(like.tensor.dtype))


class Concat_1(OpRun):
    "Concat"

    def __init__(self, node: onnx.NodeProto, version: Optional[int] = None):
        super().__init__(node, version)
        axis = self.get_attribute_int(node, "axis", 0)
        assert isinstance(axis, int), f"Unexpected value for attribute axis={axis!r}"
        self.axis = axis

    def run(self, *data: OpRunTensor) -> OpRunTensor:
        return OpRunTensor(torch.cat([t.tensor for t in data], axis=self.axis))


class NonZero_13(OpRun):
    "NonZero"

    def run(self, x: OpRunTensor) -> OpRunTensor:
        return OpRunTensor(torch.nonzero(x.tensor).T)


class Transpose_1(OpRun):
    "Transpose"

    def __init__(self, node: onnx.NodeProto, version: Optional[int] = None):
        super().__init__(node, version)
        self.perm = self.get_attribute_ints(node, "perm", None)

    def run(self, data: OpRunTensor) -> OpRunTensor:
        return OpRunTensor(torch.permute(data.tensor, self.perm))


class Trilu_14(OpRun):
    "Trilu"

    def __init__(self, node: onnx.NodeProto, version: Optional[int] = None):
        super().__init__(node, version)
        self.upper = self.get_attribute_int(node, "upper", 1)

    def run(self, data: OpRunTensor, k: Optional[OpRunTensor] = None) -> OpRunTensor:
        diagonal = 0 if k is None else k.tensor.item()
        if self.upper:
            return OpRunTensor(torch.triu(data.tensor, diagonal=diagonal))
        return OpRunTensor(torch.tril(data.tensor, diagonal=diagonal))


class Where_9(OpRun):
    "Where"

    def run(self, cond: OpRunTensor, x: OpRunTensor, y: OpRunTensor) -> OpRunTensor:
        return OpRunTensor(torch.where(cond.tensor, x.tensor, y.tensor))
