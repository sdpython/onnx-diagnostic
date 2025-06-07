from typing import Optional
import onnx
import torch
from . import OpRunKernel, OpRunTensor


class Range_11(OpRunKernel):
    """Range"""

    @classmethod
    def device_dependent(cls) -> bool:
        """
        Returns True if the kernel needs a device to be efficiently initialized.
        """
        return True

    def __init__(
        self,
        node: onnx.NodeProto,
        version: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__(node, version)
        self.device = device

    def run(self, starts: OpRunTensor, limit: OpRunTensor, delta: OpRunTensor) -> OpRunTensor:
        return OpRunTensor(
            torch.arange(
                starts.tensor,
                limit.tensor,
                delta.tensor,
                dtype=starts.dtype,
                device=self.device,
            )
        )
