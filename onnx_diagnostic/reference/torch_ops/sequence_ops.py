from typing import Optional
import onnx
import torch
from ...helpers.torch_helper import onnx_dtype_to_torch_dtype
from . import OpRun, OpRunValue, OpRunValueSequence


class OpRunSequence(OpRun):
    "Ancestor for kernel using sequences."


class ConcatFromSequence_11(OpRunSequence):
    "ConcatFromSequence"

    def __init__(self, node: onnx.NodeProto, version: Optional[int] = None):
        super().__init__(node, version)
        axis = self.get_attribute_int(node, "axis", None)
        assert isinstance(axis, int), f"Unexpected value for attribute axis={axis!r}"
        self.axis = axis
        self.new_axis = self.get_attribute_int(node, "new_axis", 0)

    def run(self, input_sequence: OpRunValueSequence) -> OpRunValue:
        assert isinstance(
            input_sequence, OpRunValueSequence
        ), f"Unexpected type {type(input_sequence)} for input_sequence"
        seq = input_sequence.sequence
        if self.new_axis == 1:
            if self.axis == -1:
                seq2 = [s.unsqueeze(len(s.shape)) for s in seq]
                res = torch.cat(seq2, axis=-1)
            else:
                seq2 = [s.expand(self.axis) for s in seq]
                res = torch.cat(seq2, axis=self.axis)
        else:
            res = torch.cat(seq, axis=self.axis)
        return OpRunValue(res)


class SequenceEmpty_11(OpRunSequence):
    "SqeuenceEmpty"

    def __init__(self, node: onnx.NodeProto, version: Optional[int] = None):
        super().__init__(node, version)
        self.dtype = onnx_dtype_to_torch_dtype(
            self.get_attribute_int(node, "dtype", onnx.TensorProto.FLOAT)
        )

    def run(self) -> OpRunValueSequence:
        return OpRunValueSequence(dtype=self.dtype)


class SequenceInsert_11(OpRunSequence):
    "SqeuenceInsert"

    def run(
        self,
        input_sequence: OpRunValueSequence,
        tensor: OpRunValue,
        position: Optional[OpRunValue] = None,
    ) -> OpRunValueSequence:
        assert isinstance(
            input_sequence, OpRunValueSequence
        ), f"Unexpected type {type(input_sequence)} for input_sequence"
        return input_sequence.insert_at(tensor, position)
