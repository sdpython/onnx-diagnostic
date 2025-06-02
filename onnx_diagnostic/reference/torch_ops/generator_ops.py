import torch
from . import OpRun, OpRunValue


class Range_11(OpRun):
    """Range"""

    def run(self, starts: OpRunValue, limit: OpRunValue, delta: OpRunValue) -> OpRunValue:
        return OpRunValue(
            torch.arange(starts.tensor, limit.tensor, delta.tensor, dtype=starts.dtype)
        )
