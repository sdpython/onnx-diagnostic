import unittest
import torch
import transformers
import transformers.integrations.sdpa_attention as sdpa_attention
import onnx_diagnostic.torch_export_patches.patches.patch_transformers as patch_transformers
from onnx_diagnostic.ext_test_case import ExtTestCase, requires_transformers
from onnx_diagnostic.helpers.torch_helper import torch_deepcopy


class TestPatchPatchTransformers(ExtTestCase):
    @requires_transformers("4.55")
    def test_sdpa_mask_recent_torch(self):
        sdpa_mask_recent_torch = transformers.masking_utils.sdpa_mask_recent_torch
        patched_sdpa_mask_recent_torch = patch_transformers.patched_sdpa_mask_recent_torch
        kwargs = {
            "batch_size": 1,
            "cache_position": torch.tensor([3], dtype=torch.int64),
            "kv_length": 4,
            "kv_offset": 0,
            "mask_function": transformers.masking_utils.causal_mask_function,
            "attention_mask": torch.tensor([[True, True, True, True]]),
            "local_size": None,
            "allow_is_causal_skip": True,
            "allow_is_bidirectional_skip": False,
        }
        expected = sdpa_mask_recent_torch(**kwargs)
        got = patched_sdpa_mask_recent_torch(**kwargs)
        self.assertEqual(expected, got)

        kwargs = {
            "batch_size": 1,
            "cache_position": torch.tensor([3], dtype=torch.int64),
            "kv_length": 4,
            "kv_offset": 0,
            "mask_function": transformers.masking_utils.causal_mask_function,
            "attention_mask": torch.tensor([[True, True, True, True]]),
            "local_size": None,
            "allow_is_causal_skip": False,
            "allow_is_bidirectional_skip": False,
        }
        expected = sdpa_mask_recent_torch(**kwargs)
        got = patched_sdpa_mask_recent_torch(**kwargs)
        self.assertEqualArray(expected, got)

    @requires_transformers("4.55")
    def test_sdpa_attention_forward_not_causal(self):
        sdpa_attention_forward = sdpa_attention.sdpa_attention_forward
        patched_sdpa_attention_forward = patch_transformers.patched_sdpa_attention_forward
        kwargs = {
            "module": None,
            "query": torch.rand((1, 2, 1, 96), dtype=torch.float32),
            "key": torch.rand((1, 2, 4, 96), dtype=torch.float32),
            "value": torch.rand((1, 2, 4, 96), dtype=torch.float32),
            "attention_mask": None,
            "attention_dropout": 0,
            "scaling": 0.10206207261596575,
            "is_causal": False,
        }
        expected = sdpa_attention_forward(**torch_deepcopy(kwargs))[0]
        got = patched_sdpa_attention_forward(**torch_deepcopy(kwargs))[0]
        self.assertEqualArray(expected, got)

        kwargs = {
            "module": None,
            "query": torch.rand((1, 2, 1, 96), dtype=torch.float32),
            "key": torch.rand((1, 2, 4, 96), dtype=torch.float32),
            "value": torch.rand((1, 2, 4, 96), dtype=torch.float32),
            "attention_mask": torch.tensor([[[[True, True, True, True]]]]),
            "attention_dropout": 0,
            "scaling": 0.10206207261596575,
            "is_causal": False,
        }
        expected = sdpa_attention_forward(**torch_deepcopy(kwargs))[0]
        got = patched_sdpa_attention_forward(**torch_deepcopy(kwargs))[0]
        self.assertEqualArray(expected, got)

    @requires_transformers("4.55")
    def test_sdpa_attention_forward_causal(self):
        sdpa_attention_forward = sdpa_attention.sdpa_attention_forward
        patched_sdpa_attention_forward = patch_transformers.patched_sdpa_attention_forward
        kwargs = {
            "module": None,
            "query": torch.rand((1, 2, 1, 96), dtype=torch.float32),
            "key": torch.rand((1, 2, 4, 96), dtype=torch.float32),
            "value": torch.rand((1, 2, 4, 96), dtype=torch.float32),
            "attention_mask": torch.tensor([[[[True, True, True, True]]]]),
            "attention_dropout": 0,
            "scaling": 0.10206207261596575,
            "is_causal": True,
        }
        expected = sdpa_attention_forward(**torch_deepcopy(kwargs))[0]
        got = patched_sdpa_attention_forward(**torch_deepcopy(kwargs))[0]
        self.assertEqualArray(expected, got)

        kwargs = {
            "module": None,
            "query": torch.rand((1, 2, 1, 96), dtype=torch.float32),
            "key": torch.rand((1, 2, 4, 96), dtype=torch.float32),
            "value": torch.rand((1, 2, 4, 96), dtype=torch.float32),
            "attention_mask": None,
            "attention_dropout": 0,
            "scaling": 0.10206207261596575,
            "is_causal": True,
        }
        expected = sdpa_attention_forward(**torch_deepcopy(kwargs))[0]
        got = patched_sdpa_attention_forward(**torch_deepcopy(kwargs))[0]
        self.assertEqualArray(expected, got)

    def test_causal_mask_in_scaled_dot_product_attention(self):
        # see https://docs.pytorch.org/docs/stable/generated/...
        #       ...torch.nn.functional.scaled_dot_product_attention.html

        query = torch.rand((1, 2, 1, 96), dtype=torch.float32)
        key = torch.rand((1, 2, 4, 96), dtype=torch.float32)
        L, S = query.size(-2), key.size(-2)
        attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
        self.assertEqual(attn_bias.min().item(), 0)
        attn_causal_bias = attn_bias.clone()

        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_causal_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        self.assertEqual(attn_causal_bias.min().item(), -float("inf"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
