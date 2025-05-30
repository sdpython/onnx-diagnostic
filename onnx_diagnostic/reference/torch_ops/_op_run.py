from typing import Optional, Union, Tuple
import onnx
import torch


class OpRun:
    """
    Main class. Every kernel should inherit from it.
    It does not copy the proto.
    """

    def __init__(self, node: onnx.NodeProto, version: Optional[int] = None):
        self.op_type = node.op_type
        self.domain = node.domain
        self.input = node.input
        self.output = node.output
        if version is None:
            name = self.__class__.__name__.split("_")
            assert (
                len(name) == 2
            ), f"Cannot guess version from name={self.__class__.__name__!r}"
            version = int(name[1])
        self.version = version

    def __str__(self) -> str:
        "usual"
        if self.domain:
            return (
                f"{self.op_type}[{self.domain}]({', '.join(self.input)}) "
                f"-> {', '.join(self.output)}"
            )
        return f"{self.op_type}({', '.join(self.input)}) -> {', '.join(self.output)}"

    def run(self, *args) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        "Kernel implementation."
        raise NotImplementedError(
            f"Method run is not implemented for kernel {self.__class__.__name__!r}"
        )
