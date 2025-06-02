from typing import Optional, Union, Tuple
import onnx
import torch
from ...helpers import string_type
from ...helpers.torch_helper import to_tensor


class OpRunValue:
    """
    Wrapper around a tensor.

    :param tensor: torch.Tensor
    :param is_constant: is it a constant
    """

    __slots__ = ("cached", "is_constant", "tensor")

    def __init__(self, tensor, is_constant: bool = False):
        self.tensor = tensor
        self.is_constant = is_constant
        self.cached: Optional[Tuple[int, ...]] = None

    def __repr__(self) -> str:
        "usual"
        if self.is_constant:
            return (
                f"{self.__class__.__name__}"
                f"({string_type(self.tensor, with_shape=True)}, is_constant=True)"
            )
        return f"{self.__class__.__name__}({string_type(self.tensor, with_shape=True)})"

    @property
    def shape(self):
        "shape (torch shape)"
        return self.tensor.shape

    @property
    def dtype(self):
        "dtype (torch dtype)"
        return self.tensor.dtype

    def _tensor_as_tuple_int(self) -> Tuple[int, ...]:
        return tuple(map(int, self.tensor))

    @property
    def as_tuple_int(self) -> Tuple[int, ...]:
        "value as int"
        if self.is_constant:
            if self.cached is None:
                self.cached = self._tensor_as_tuple_int()
            return self.cached
        return self._tensor_as_tuple_int()


class OpRun:
    """
    Main class. Every kernel should inherit from it.
    It does not copy the proto.
    """

    @classmethod
    def device_dependent(cls) -> bool:
        """
        Returns True if the kernel needs a device to be efficiently initialized.
        """
        return False

    @classmethod
    def has_subgraphs(cls) -> bool:
        """Returns True if the kernel has subgraphs."""
        return False

    def __init__(self, node: onnx.NodeProto, version: Optional[int] = None):
        assert isinstance(
            node, onnx.NodeProto
        ), f"node must be a NodeProto but node is {type(node)}"
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

    def run(
        self, *args: Optional[OpRunValue]
    ) -> Union[OpRunValue, Tuple[Optional[OpRunValue], ...]]:
        "Kernel implementation."
        raise NotImplementedError(
            f"Method run is not implemented for kernel {self.__class__.__name__!r}"
        )

    def _find_attribute(self, node: onnx.NodeProto, name: str):
        for att in node.attribute:
            if att.name == name:
                return att
        return None

    def get_attribute_float(
        self, node: onnx.NodeProto, name: str, default_value: Optional[float] = None
    ) -> Optional[float]:
        """
        Returns an attribute as an int.

        :param node: NodeProto
        :param name: name
        :param default_value: default_value
        :return: value
        """
        att = self._find_attribute(node, name)
        return default_value if att is None else float(att.f)

    def get_attribute_int(
        self, node: onnx.NodeProto, name: str, default_value: Optional[int] = None
    ) -> Optional[int]:
        """
        Returns an attribute as an int.

        :param node: NodeProto
        :param name: name
        :param default_value: default_value
        :return: value
        """
        att = self._find_attribute(node, name)
        return default_value if att is None else int(att.i)

    def get_attribute_ints(
        self, node: onnx.NodeProto, name: str, default_value: Optional[Tuple[int, ...]] = None
    ) -> Optional[Tuple[int, ...]]:
        """
        Returns an attribute as a tuple of ints.

        :param node: NodeProto
        :param name: name
        :param default_value: default_value
        :return: value
        """
        att = self._find_attribute(node, name)
        return default_value if att is None else tuple(att.ints)

    def get_attribute_tensor(self, node: onnx.NodeProto, name: str) -> Optional[torch.Tensor]:
        """
        Returns an attribute as a torch tensor.

        :param node: NodeProto
        :param name: name
        :param default_value: default_value
        :return: value
        """
        att = self._find_attribute(node, name)
        if att is None:
            return None
        return to_tensor(att.t)


class OpRunFunction(OpRun):
    """
    Defines a kernel based on a local functions.
    """

    def __init__(
        self,
        runtime: "onnx_diagnostic.reference.TorchOnnxEvaluator",  # noqa: F821
        node: onnx.NodeProto,
        version: Optional[int] = None,
    ):
        super().__init__(node, version)
        self.runtime = runtime
        self.input_names = runtime.input_names

    def run(
        self, *args: Optional[OpRunValue]
    ) -> Union[OpRunValue, Tuple[Optional[OpRunValue], ...]]:
        return self.runtime.run_with_values(*args)
