from typing import Optional
import onnx


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
                f"-> {', '.join(self.outputs)}"
            )
        return f"{self.op_type}({', '.join(self.input)}) -> {', '.join(self.outputs)}"
