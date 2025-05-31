from typing import Optional
import onnx
import torch
from . import OpRun, OpRunValue


class Gather_1(OpRun):
    "Gather"

    def __init__(self, node: onnx.NodeProto, version: Optional[int] = None):
        super().__init__(node, version)
        axis = self.get_attribute_int(node, "axis", 0)
        assert isinstance(axis, int), f"Unexpected value for attribute axis={axis!r}"
        self.axis = axis

    def run(self, x, indices):
        if indices.tensor.numel() == 0:
            return torch.empty((0,), dtype=x.tensor.dtype, device=x.tensor.device)
        ind = [slice(0, s) for s in x.shape]
        ind[self.axis] = indices.tensor
        return OpRunValue(x.tensor[*ind])


class Slice_13(OpRun):
    "Slice"

    def run(
        self,
        data: OpRunValue,
        starts: OpRunValue,
        ends: OpRunValue,
        axes: Optional[OpRunValue] = None,
        steps: Optional[OpRunValue] = None,
    ) -> OpRunValue:
        if axes is None:
            if steps is None:
                slices = [slice(s, e) for s, e in zip(starts.tensor, ends.tensor)]
            else:
                slices = [
                    slice(s, e, d) for s, e, d in zip(starts.tensor, ends.tensor, steps.tensor)
                ]
        else:
            if steps is None:
                slices = [slice(0, a) for a in data.shape]
                for s, e, a in zip(starts.tensor, ends.tensor, axes.tensor):
                    slices[a] = slice(s, e)
            else:
                slices = [slice(0, a) for a in data.shape]
                for s, e, a, d in zip(starts.tensor, ends.tensor, axes.tensor, steps.tensor):
                    slices[a] = slice(s, e, d)
        return OpRunValue(data.tensor[tuple(slices)])
