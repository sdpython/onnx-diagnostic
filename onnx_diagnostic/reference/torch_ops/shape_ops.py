from typing import Optional, Tuple
import onnx
import torch
from . import OpRun, OpRunTensor


class ConstantOfShape_9(OpRun):
    "ConstantOfShape"

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
        value = self.get_attribute_tensor(node, "value")
        if value is None:
            value = torch.tensor([0], dtype=torch.float32)
        self.dtype = value.dtype
        self.device = device
        self.value = value[0]

    def run(self, shape: OpRunTensor) -> OpRunTensor:
        # The device is unknown as shapes usually take place on CPU.
        return OpRunTensor(
            torch.full(
                shape.as_tuple_int, fill_value=self.value, dtype=self.dtype, device=self.device
            )
        )


class Expand_8(OpRun):
    "Expand"

    def run(self, data: OpRunTensor, shape: OpRunTensor) -> OpRunTensor:
        ishape = tuple(-1 if i == 1 else i for i in shape.as_tuple_int)
        return OpRunTensor(data.tensor.expand(ishape))


class Reshape_14(OpRun):
    "Reshape"

    def __init__(self, node: onnx.NodeProto, version: Optional[int] = None):
        super().__init__(node, version)
        self.allowzero = self.get_attribute_int(node, "allowzero", 0)

    def run(self, data: OpRunTensor, shape: OpRunTensor) -> OpRunTensor:
        ishape = shape.as_tuple_int
        assert ishape is not None, f"Unexpected return for shape={shape!r}"
        if not self.allowzero and 0 in ishape:
            xshape = data.tensor.shape
            new_shape = []
            for i, s in enumerate(ishape):
                new_shape.append(xshape[i] if s == 0 else s)
            return OpRunTensor(data.tensor.reshape(new_shape))
        return OpRunTensor(data.tensor.reshape(ishape))


class Shape_15(OpRun):
    def __init__(self, node: onnx.NodeProto, version: Optional[int] = None):
        super().__init__(node, version)
        self.start = self.get_attribute_int(node, "start", 0)
        self.end = self.get_attribute_int(node, "end", None)

    def run(self, data: OpRunTensor) -> OpRunTensor:
        shape = data.shape
        sh = shape[self.start :] if self.end is None else shape[self.start : self.end]
        return OpRunTensor(torch.tensor(sh, dtype=torch.int64), is_constant=True)


class Split_18(OpRun):
    def __init__(self, node: onnx.NodeProto, version: Optional[int] = None):
        super().__init__(node, version)
        self.axis = self.get_attribute_int(node, "axis", 0)
        self.num_outputs = self.get_attribute_int(node, "num_outputs", None)

    def run(
        self, data: OpRunTensor, split: Optional[OpRunTensor] = None
    ) -> Tuple[OpRunTensor, ...]:
        if split is None:
            assert isinstance(
                self.num_outputs, int
            ), f"Incompatibilities: split is None and num_outputs={self.num_outputs}"
            size = data.tensor.shape[self.axis]
            split_size = (
                size // self.num_outputs
                if size % self.num_outputs == 0
                else size // self.num_outputs + 1
            )
            spl = torch.split(data.tensor, split_size, dim=self.axis)
        else:
            spl = torch.split(data.tensor, split.as_tuple_int, dim=self.axis)
        return tuple(OpRunTensor(t) for t in spl)


class Squeeze_13(OpRun):
    "Squeeze"

    def run(self, data: OpRunTensor, axes: Optional[OpRunTensor] = None) -> OpRunTensor:
        if axes is None:
            return OpRunTensor(data.tensor.squeeze())
        return OpRunTensor(data.tensor.squeeze(axes.as_tuple_int))


class Unsqueeze_13(OpRun):
    "Unsqueeze"

    def run(self, data: OpRunTensor, axes: OpRunTensor) -> OpRunTensor:
        t = data.tensor
        for i in axes.as_tuple_int:
            t = t.unsqueeze(i)
        return OpRunTensor(t)
