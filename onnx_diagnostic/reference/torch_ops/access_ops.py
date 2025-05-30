from typing import Optional
from . import OpRun, OpRunValue


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
