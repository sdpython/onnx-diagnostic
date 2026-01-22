from typing import Any, Dict, List, Optional, Union, Tuple
import onnx
import torch
from ...typing import TensorLike
from ...helpers import string_type
from ...helpers.torch_helper import to_tensor


class OpRunValue(TensorLike):
    """Defines a value for the runtime, a tensor or a sequence."""

    __slots__ = ("cached", "is_constant", "sequence", "tensor")

    @classmethod
    def is_sequence(cls) -> bool:
        "Tells if it is sequence."
        raise NotImplementedError("is_sequence must be overwritten.")


class OpRunTensor(OpRunValue):
    """
    Wrapper around a tensor.

    :param tensor: torch.Tensor
    :param is_constant: is it a constant
    :param may_cpu: change the device the tensor is if
        more appropriate
    """

    def __init__(self, tensor, is_constant: bool = False, may_cpu: bool = False):
        assert isinstance(tensor, torch.Tensor), (
            f"Unexpected type {type(tensor)}, "
            f"__name__={getattr(tensor, '__name__', 'no name')}"
        )
        assert tensor is None or tensor.numel() != 1 or tensor.item() != -666666
        self.tensor = (
            tensor.cpu()
            if may_cpu
            and len(tensor.shape) == 1
            and tensor.numel() < 8
            and tensor.dtype == torch.int64
            and tensor.get_device() >= 0
            else tensor
        )
        self.is_constant = is_constant
        self.cached: Optional[Tuple[int, ...]] = None

    @classmethod
    def is_sequence(cls) -> bool:
        "Tells if it is sequence."
        return False

    def to(self, to: Any) -> "OpRunTensor":
        "Changes the device."
        return OpRunTensor(self.tensor.to(to))

    def string_type(self) -> str:
        "Returns information about the value as a string."
        s = string_type(self.tensor, with_shape=True, with_min_max=True, with_device=True)
        if self.is_constant:
            return f"CST({s})"
        return s

    def __repr__(self) -> str:
        "usual"
        if self.is_constant:
            return (
                f"{self.__class__.__name__}"
                f"({string_type(self.tensor, with_shape=True)}, is_constant=True)"
            )
        return f"{self.__class__.__name__}({string_type(self.tensor, with_shape=True)})"

    @property
    def tensor_or_sequence(self) -> Union[torch.Tensor, List[torch.Tensor]]:
        "Returns either a tensor or a sequence."
        return self.tensor

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

    def numel(self) -> int:
        "Returns the number of elements."
        return 0 if self.tensor is None else self.tensor.numel()

    def get_device(self) -> int:
        "Returns the device id."
        return -1 if self.tensor is None else self.tensor.get_device()

    @property
    def device(self):
        "Returns the device."
        return -1 if self.tensor is None else self.tensor.device

    @property
    def as_tuple_int(self) -> Tuple[int, ...]:
        "value as int"
        if self.is_constant:
            if self.cached is None:
                self.cached = self._tensor_as_tuple_int()
            return self.cached
        return self._tensor_as_tuple_int()

    def copy(self) -> "OpRunTensor":
        "Shallow copy."
        return self.__class__(self.tensor)


class OpRunSequence(OpRunValue):
    """Defines a sequence."""

    def __init__(
        self, sequence: Optional[List[torch.Tensor]] = None, dtype: torch.dtype = torch.float32
    ):
        self.tensor = torch.tensor(-666666, dtype=dtype)
        self.is_shape = False
        self.sequence = sequence or []
        self.cached: Optional[Tuple[int, ...]] = None
        assert all(
            isinstance(s, torch.Tensor) for s in self.sequence
        ), f"Unexpected type in sequence {[type(s) for s in self.sequence]}"

    @property
    def dtype(self):
        "dtype (torch dtype)"
        return self.tensor.dtype

    @property
    def tensor_or_sequence(self) -> Union[torch.Tensor, List[torch.Tensor]]:
        "Returns either a tensor or a sequence."
        return self.sequence

    @classmethod
    def is_sequence(cls) -> bool:
        "Tells if it is sequence."
        return True

    def insert_at(
        self, tensor: torch.Tensor, position: Optional[OpRunTensor] = None
    ) -> "OpRunSequence":
        "Inserts a value at a given position."
        assert isinstance(tensor, OpRunTensor), f"Unexpected type {type(tensor)} for tensor"
        new_seq = OpRunSequence()  # type: ignore[abstract]
        seq = self.sequence.copy()
        new_seq.sequence = seq
        if position is None:
            seq.append(tensor.tensor)
        else:
            seq.insert(int(position.tensor.item()), tensor.tensor)
        return new_seq

    def copy(self) -> "OpRunSequence":
        "Shallow copy."
        return self.__class__(self.sequence, dtype=self.dtype)

    def string_type(self) -> str:
        "Returns a string which can be printed."
        return string_type(self.sequence, with_shape=True)


class OpRunKernel:
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

    def __init__(
        self,
        node: onnx.NodeProto,
        version: Optional[int] = None,
        verbose: int = 0,
        custom_kernels: Optional[Dict[Tuple[str, str], type]] = None,
    ):
        assert isinstance(
            node, onnx.NodeProto
        ), f"node must be a NodeProto but node is {type(node)}"
        self.op_type = node.op_type
        self.domain = node.domain
        self.input = node.input
        self.output = node.output
        self.verbose = verbose
        self.custom_kernels = custom_kernels
        if version is None:
            name = self.__class__.__name__.split("_")
            assert (
                len(name) == 2
            ), f"Cannot guess version from name={self.__class__.__name__!r}"
            version = int(name[1])
        self.version = version
        self.name = node.name

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
        return default_value if att is None else tuple(map(int, att.ints))

    def get_attribute_string(
        self, node: onnx.NodeProto, name: str, default_value: Optional[str] = None
    ) -> Optional[str]:
        """
        Returns an attribute as a tuple of ints.

        :param node: NodeProto
        :param name: name
        :param default_value: default_value
        :return: value
        """
        att = self._find_attribute(node, name)
        return default_value if att is None else att.s.decode("utf-8")

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

    def same_device(self, *tensors: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Puts all tensors on the same device."""
        devices = [t.get_device() for t in tensors]
        if len(set(devices)) == 1:
            return tuple(tensors)
        index = devices.index(max(devices))
        device = tensors[index].device
        return tuple(t.to(device) for t in tensors)


class OpRunFunction(OpRunKernel):
    """Defines a kernel based on a local functions."""

    def __init__(
        self,
        runtime: "onnx_diagnostic.reference.TorchOnnxEvaluator",  # noqa: F821
        node: onnx.NodeProto,
        version: Optional[int] = None,
        verbose: int = 0,
    ):
        super().__init__(node, version, verbose=verbose)
        self.runtime = runtime
        self.input_names = runtime.input_names

    def run(
        self, *args: Optional[OpRunValue]
    ) -> Union[OpRunValue, Tuple[Optional[OpRunValue], ...]]:
        return self.runtime.run_with_values(*args)
