import unittest
import torch
import transformers
from onnx_diagnostic.ext_test_case import ExtTestCase, hide_stdout, requires_transformers
from onnx_diagnostic.helpers import string_type
from onnx_diagnostic.torch_export_patches.patch_inputs import (
    convert_dynamic_axes_into_dynamic_shapes,
)


class TestPatchInputs(ExtTestCase):
    @hide_stdout()
    @requires_transformers("4.50")
    def test_convert_dynamic_axes_into_dynamic_shapes_1(self):
        args = (
            torch.randint(0, 10, size=(2, 8)).to(torch.int64),
            torch.randint(0, 10, size=(2, 8)).to(torch.int64),
            torch.randint(0, 10, size=(2, 8)).to(torch.int64),
            [(torch.rand((2, 1, 3, 96)), torch.rand((2, 1, 3, 96)))],
        )
        dynamic_axes = {
            "attention_mask": {0: "batch_size", 1: "total_sequence_length"},
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size", 1: "sequence_length"},
            "past_key_values.0.key": {0: "batch_size", 2: "past_sequence_length"},
            "past_key_values.0.value": {0: "batch_size", 2: "past_sequence_length"},
            "position_ids": {0: "batch_size", 1: "sequence_length"},
            "present.0.key": {0: "batch_size", 2: "total_sequence_length"},
            "present.0.value": {0: "batch_size", 2: "total_sequence_length"},
        }

        model_cls = transformers.LlamaModel
        res = convert_dynamic_axes_into_dynamic_shapes(
            model_cls, args=args, dynamic_axes=dynamic_axes, verbose=1
        )
        self.assertEqual((), res[0])
        self.assertEqual(
            (
                "dict(input_ids:T7s2x8,attention_mask:T7s2x8,position_ids:T7s2x8,"
                "past_key_values:DynamicCache[serialized](#2[#1[T1s2x1x3x96],#1[T1s2x1x3x96]]))"
            ),
            string_type(res[1], with_shape=True),
        )
        self.assertEqual(
            {
                "attention_mask": {0: "batch_size", 1: "total_sequence_length"},
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "past_key_values": [
                    [{0: "batch_size", 2: "past_sequence_length"}],
                    [{0: "batch_size", 2: "past_sequence_length"}],
                ],
                "position_ids": {0: "batch_size", 1: "sequence_length"},
            },
            res[2],
        )

    @hide_stdout()
    @requires_transformers("4.50")
    def test_convert_dynamic_axes_into_dynamic_shapes_2(self):
        args = (
            torch.randint(0, 10, size=(2, 8)).to(torch.int64),
            torch.randint(0, 10, size=(2, 8)).to(torch.int64),
            torch.randint(0, 10, size=(2, 8)).to(torch.int64),
            [(torch.rand((2, 1, 3, 96)), torch.rand((2, 1, 3, 96)))],
        )
        dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "position_ids": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size", 1: "sequence_length"},
            "present.0.key": {0: "batch_size", 2: "past_sequence_length"},
            "present.0.value": {0: "batch_size", 2: "past_sequence_length"},
        }

        model_cls = transformers.LlamaModel
        res = convert_dynamic_axes_into_dynamic_shapes(
            model_cls,
            args=args,
            dynamic_axes=dynamic_axes,
            verbose=1,
            prefix_mapping={"present": "past_key_values"},
        )
        self.assertEqual((), res[0])
        self.assertEqual(
            {"attention_mask", "input_ids", "past_key_values", "position_ids"}, set(res[2])
        )
        self.assertEqual(
            [
                [{0: "batch_size", 2: "past_sequence_length"}],
                [{0: "batch_size", 2: "past_sequence_length"}],
            ],
            res[2]["past_key_values"],
        )
        self.assertEqual(
            {
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "past_key_values": [
                    [{0: "batch_size", 2: "past_sequence_length"}],
                    [{0: "batch_size", 2: "past_sequence_length"}],
                ],
                "position_ids": {0: "batch_size", 1: "sequence_length"},
            },
            res[2],
        )
        self.assertEqual(
            (
                "dict(input_ids:T7s2x8,attention_mask:T7s2x8,position_ids:T7s2x8,"
                "past_key_values:DynamicCache[serialized](#2[#1[T1s2x1x3x96],#1[T1s2x1x3x96]]))"
            ),
            string_type(res[1], with_shape=True),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
