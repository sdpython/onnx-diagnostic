import unittest
import ml_dtypes
import onnx
import torch
import transformers
from onnx_diagnostic.ext_test_case import ExtTestCase, hide_stdout
from onnx_diagnostic.helpers import max_diff, string_type
from onnx_diagnostic.helpers.torch_test_helper import (
    dummy_llm,
    to_numpy,
    is_torchdynamo_exporting,
    steal_forward,
    replace_string_by_dynamic,
    to_any,
    torch_deepcopy,
)
from onnx_diagnostic.helpers.cache_helper import (
    make_dynamic_cache,
    make_encoder_decoder_cache,
    make_mamba_cache,
    make_sliding_window_cache,
)

TFLOAT = onnx.TensorProto.FLOAT


class TestTorchTestHelper(ExtTestCase):

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
    def test_steal_forward(self):
        class Model(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        inputs = torch.rand(3, 4), torch.rand(3, 4)
        model = Model()
        with steal_forward(model):
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
            .replace(
                "_DimHint(type=<_DimHintType.DYNAMIC:3>,min=None,max=None,_factory=True)",
                "DYN",
            )
        )
        self.assertEqual(
            "{'input_ids':{0:DYN,1:DYN},'attention_mask':({0:DYN,1:DYN},),'position_ids':[{0:DYN,1:DYN}]}",
            sproc,
        )

    def test_to_any(self):
        c1 = make_dynamic_cache([(torch.rand((4, 4, 4)), torch.rand((4, 4, 4)))])
        c2 = make_encoder_decoder_cache(
            make_dynamic_cache([(torch.rand((4, 4, 4)), torch.rand((4, 4, 4)))]),
            make_dynamic_cache([(torch.rand((5, 5, 5)), torch.rand((5, 5, 5)))]),
        )
        a = {"t": [(torch.tensor([1, 2]), c1, c2), {4, 5}]}
        at = to_any(a, torch.float16)
        self.assertIn("T10r", string_type(at))

    def test_torch_deepcopy_cache_dce(self):
        c1 = make_dynamic_cache([(torch.rand((4, 4, 4)), torch.rand((4, 4, 4)))])
        c2 = make_encoder_decoder_cache(
            make_dynamic_cache([(torch.rand((4, 4, 4)), torch.rand((4, 4, 4)))]),
            make_dynamic_cache([(torch.rand((5, 5, 5)), torch.rand((5, 5, 5)))]),
        )
        cc = torch_deepcopy(c2)
        self.assertEqual(type(c2), type(c2))
        self.assertEqual(max_diff(c2, cc)["abs"], 0)
        a = {"t": [(torch.tensor([1, 2]), c1, c2), {4, 5}]}
        at = torch_deepcopy(a)
        hash1 = string_type(at, with_shape=True, with_min_max=True)
        c1.key_cache[0] += 1000
        hash2 = string_type(at, with_shape=True, with_min_max=True)
        self.assertEqual(hash1, hash2)

    def test_torch_deepcopy_mamba_cache(self):
        cache = make_mamba_cache(
            [
                (torch.rand((4, 4, 4)), torch.rand((4, 4, 4))),
                (torch.rand((4, 4, 4)), torch.rand((4, 4, 4))),
                (torch.rand((4, 4, 4)), torch.rand((4, 4, 4))),
            ]
        )
        at = torch_deepcopy(cache)
        self.assertEqual(type(cache), type(at))
        self.assertEqual(max_diff(cache, at)["abs"], 0)
        hash1 = string_type(at, with_shape=True, with_min_max=True)
        cache.conv_states[0] += 1000
        hash2 = string_type(at, with_shape=True, with_min_max=True)
        self.assertEqual(hash1, hash2)

    def test_torch_deepcopy_base_model_outputs(self):
        bo = transformers.modeling_outputs.BaseModelOutput(
            last_hidden_state=torch.rand((4, 4, 4))
        )
        at = torch_deepcopy(bo)
        self.assertEqual(max_diff(bo, at)["abs"], 0)
        self.assertEqual(type(bo), type(at))
        hash1 = string_type(at, with_shape=True, with_min_max=True)
        bo.last_hidden_state[0] += 1000
        hash2 = string_type(at, with_shape=True, with_min_max=True)
        self.assertEqual(hash1, hash2)

    def test_torch_deepcopy_sliding_windon_cache(self):
        cache = make_sliding_window_cache(
            [
                (torch.rand((4, 5, 6, 7)), torch.rand((4, 5, 6, 7))),
                (torch.rand((4, 5, 6, 7)), torch.rand((4, 5, 6, 7))),
                (torch.rand((4, 5, 6, 7)), torch.rand((4, 5, 6, 7))),
            ]
        )
        at = torch_deepcopy(cache)
        self.assertEqual(type(cache), type(at))
        self.assertEqual(max_diff(cache, at)["abs"], 0)
        hash1 = string_type(at, with_shape=True, with_min_max=True)
        cache.key_cache[0] += 1000
        hash2 = string_type(at, with_shape=True, with_min_max=True)
        self.assertEqual(hash1, hash2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
