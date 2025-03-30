import unittest
import ml_dtypes
import onnx
import torch
from onnx_diagnostic.ext_test_case import ExtTestCase, hide_stdout
from onnx_diagnostic.torch_test_helper import (
    dummy_llm,
    to_numpy,
    is_torchdynamo_exporting,
    steel_forward,
    replace_string_by_dynamic,
)

TFLOAT = onnx.TensorProto.FLOAT


class TestOrtSession(ExtTestCase):

    def test_is_torchdynamo_exporting(self):
        self.assertFalse(is_torchdynamo_exporting())

    def test_dummy_llm(self):
        for cls_name in ["AttentionBlock", "MultiAttentionBlock", "DecoderLayer", "LLM"]:
            model, inputs = dummy_llm(cls_name)
            model(*inputs)

    def test_dummy_llm_ds(self):
        for cls_name in ["AttentionBlock", "MultiAttentionBlock", "DecoderLayer", "LLM"]:
            model, inputs, ds = dummy_llm(cls_name, dynamic_shapes=True)
            model(*inputs)
            self.assertIsInstance(ds, dict)

    def test_dummy_llm_exc(self):
        self.assertRaise(lambda: dummy_llm("LLLLLL"), NotImplementedError)

    def test_to_numpy(self):
        t = torch.tensor([0, 1], dtype=torch.bfloat16)
        a = to_numpy(t)
        self.assertEqual(a.dtype, ml_dtypes.bfloat16)

    @hide_stdout()
    def test_steel_forward(self):
        class Model(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        inputs = torch.rand(3, 4), torch.rand(3, 4)
        model = Model()
        with steel_forward(model):
            model(*inputs)

    def test_replace_string_by_dynamic(self):
        example = {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": ({0: "batch_size", 1: "sequence_length"},),
            "position_ids": [{0: "batch_size", 1: "sequence_length"}],
        }
        proc = replace_string_by_dynamic(example)
        sproc = (
            str(proc)
            .replace("_DimHint(type=<_DimHintType.DYNAMIC: 3>)", "DYN")
            .replace(" ", "")
            .replace("<_DimHint.DYNAMIC:3>", "DYN")
        )
        self.assertEqual(
            "{'input_ids':{0:DYN,1:DYN},'attention_mask':({0:DYN,1:DYN},),'position_ids':[{0:DYN,1:DYN}]}",
            sproc,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
