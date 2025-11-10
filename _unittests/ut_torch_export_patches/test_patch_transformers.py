import unittest
import torch
import transformers
import transformers.integrations.sdpa_attention as sdpa_attention
import onnx_diagnostic.torch_export_patches.patches.patch_transformers as patch_transformers
from onnx_diagnostic.ext_test_case import ExtTestCase, requires_transformers, ignore_warnings
from onnx_diagnostic.helpers.torch_helper import torch_deepcopy
from onnx_diagnostic.export.shape_helper import make_fake_with_dynamic_dimensions
from onnx_diagnostic.torch_models.hghub.hub_api import get_cached_configuration
from onnx_diagnostic.torch_export_patches import torch_export_patches
from onnx_diagnostic.torch_export_patches.patch_inputs import use_dyn_not_str
from onnx_diagnostic.torch_export_patches.patches.patch_transformers import patch_qwen2_5
from onnx_diagnostic.export.api import to_onnx


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

    @ignore_warnings(UserWarning)
    def test_sdpa_attention_forward_export_is_causal(self):
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
            "is_causal": True,
        }
        expected = sdpa_attention_forward(**torch_deepcopy(kwargs))[0]
        got = patched_sdpa_attention_forward(**torch_deepcopy(kwargs))[0]
        self.assertEqualArray(expected, got)

        class Model(torch.nn.Module):
            def forward(self, query, key, value):
                kwargs = {
                    "module": None,
                    "query": query,
                    "key": key,
                    "value": value,
                    "attention_mask": None,
                    "attention_dropout": 0,
                    "scaling": 0.10206207261596575,
                    "is_causal": True,
                }
                return patched_sdpa_attention_forward(**kwargs)[0]

        query, key, value = kwargs["query"], kwargs["key"], kwargs["value"]
        model = Model()
        got = model(query, key, value)
        self.assertEqualArray(expected, got)

        # static export
        ep = torch.export.export(model, (query, key, value))
        got = ep.module()(query, key, value)
        self.assertEqualArray(expected, got)

        # dynamic
        ds = ({0: "batch", 2: "seq1"}, {0: "batch", 2: "seq2"}, {0: "batch", 2: "seq2"})
        fake_inputs, _ = make_fake_with_dynamic_dimensions((query, key, value), ds)
        epd = torch.export.export(model, fake_inputs, dynamic_shapes=use_dyn_not_str(ds))
        got = epd.module()(query, key, value)
        self.assertEqualArray(expected, got)

    @ignore_warnings(UserWarning)
    def test_sdpa_attention_forward_export_is_causal_none(self):
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
            "is_causal": None,
        }
        expected = sdpa_attention_forward(**torch_deepcopy(kwargs))[0]
        got = patched_sdpa_attention_forward(**torch_deepcopy(kwargs))[0]
        self.assertEqualArray(expected, got)

        class Model(torch.nn.Module):
            def forward(self, query, key, value):
                kwargs = {
                    "module": None,
                    "query": query,
                    "key": key,
                    "value": value,
                    "attention_mask": None,
                    "attention_dropout": 0,
                    "scaling": 0.10206207261596575,
                    "is_causal": None,
                }
                return patched_sdpa_attention_forward(**kwargs)[0]

        query, key, value = kwargs["query"], kwargs["key"], kwargs["value"]
        model = Model()
        got = model(query, key, value)
        self.assertEqualArray(expected, got)

        # static export
        ep = torch.export.export(model, (query, key, value))
        got = ep.module()(query, key, value)
        self.assertEqualArray(expected, got)

        # dynamic
        ds = ({0: "batch", 2: "seq1"}, {0: "batch", 2: "seq2"}, {0: "batch", 2: "seq2"})
        fake_inputs, _ = make_fake_with_dynamic_dimensions((query, key, value), ds)
        epd = torch.export.export(model, fake_inputs, dynamic_shapes=use_dyn_not_str(ds))
        got = epd.module()(query, key, value)
        self.assertEqualArray(expected, got)

    @requires_transformers("4.55")
    @unittest.skipIf(not patch_qwen2_5, "Qwen25 not part of this transformers")
    def test_patched_qwen2_5_vl_rot_pos_emb(self):
        from onnx_diagnostic.torch_export_patches.patches.patch_transformers import (
            patched_Qwen2_5_VisionTransformerPretrainedModel,
        )

        config = get_cached_configuration("Qwen/Qwen2.5-VL-7B-Instruct")
        patched_class = patched_Qwen2_5_VisionTransformerPretrainedModel._PATCHED_CLASS_
        instance = patched_class(config.vision_config)
        grid_thw = torch.tensor([[1, 34, 38], [1, 34, 38]], dtype=torch.int64)
        expected = instance.rot_pos_emb(grid_thw)
        got = patched_Qwen2_5_VisionTransformerPretrainedModel.rot_pos_emb(instance, grid_thw)
        self.assertEqualArray(expected, got)

    @requires_transformers("4.55")
    @unittest.skipIf(not patch_qwen2_5, "Qwen25 not part of this transformers")
    def test_patched_qwen2_5_vl_get_window_index(self):
        from onnx_diagnostic.torch_export_patches.patches.patch_transformers import (
            patched_Qwen2_5_VisionTransformerPretrainedModel,
        )

        config = get_cached_configuration("Qwen/Qwen2.5-VL-7B-Instruct")
        patched_class = patched_Qwen2_5_VisionTransformerPretrainedModel._PATCHED_CLASS_
        instance = patched_class(config.vision_config)
        grid_thw = torch.tensor([[1, 34, 38]], dtype=torch.int64)
        window_index1, cu_window_seqlens1 = instance.get_window_index(grid_thw)
        window_index2, cu_window_seqlens2 = (
            patched_Qwen2_5_VisionTransformerPretrainedModel.get_window_index(
                instance, grid_thw
            )
        )
        self.assertEqualArray(window_index1, window_index2)
        self.assertEqualArray(torch.tensor(cu_window_seqlens1), cu_window_seqlens2)

        grid_thw = torch.tensor([[1, 34, 38], [1, 34, 38]], dtype=torch.int64)
        window_index1, cu_window_seqlens1 = instance.get_window_index(grid_thw)
        window_index2, cu_window_seqlens2 = (
            patched_Qwen2_5_VisionTransformerPretrainedModel.get_window_index(
                instance, grid_thw
            )
        )
        self.assertEqualArray(window_index1, window_index2)
        self.assertEqualArray(torch.tensor(cu_window_seqlens1), cu_window_seqlens2)

    @requires_transformers("4.55")
    @unittest.skipIf(not patch_qwen2_5, "Qwen25 not part of this transformers")
    def test_patched_qwen2_5_vl_forward(self):
        from onnx_diagnostic.torch_export_patches.patches.patch_transformers import (
            patched_Qwen2_5_VisionTransformerPretrainedModel,
        )

        config = get_cached_configuration("Qwen/Qwen2.5-VL-7B-Instruct")
        patched_class = patched_Qwen2_5_VisionTransformerPretrainedModel._PATCHED_CLASS_
        instance = patched_class(config.vision_config)
        hidden_states = torch.rand((1292, 1176), dtype=torch.float32)
        grid_thw = torch.tensor([[1, 34, 38]], dtype=torch.int64)
        expected = instance.forward(hidden_states, grid_thw)
        f_get_window_index = patched_class.get_window_index
        patched_class.get_window_index = (
            patched_Qwen2_5_VisionTransformerPretrainedModel.get_window_index
        )
        got = patched_Qwen2_5_VisionTransformerPretrainedModel.forward(
            instance, hidden_states, grid_thw
        )
        patched_class.get_window_index = f_get_window_index
        self.assertEqualArray(expected, got)

    @requires_transformers("4.55")
    @unittest.skipIf(not patch_qwen2_5, "Qwen25 not part of this transformers")
    def test_qwen_apply_multimodal_rotary_pos_emb(self):
        apply_multimodal_rotary_pos_emb = (
            transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.apply_multimodal_rotary_pos_emb
        )

        class Model(torch.nn.Module):
            def forward(self, q, k, cos, sin):
                return apply_multimodal_rotary_pos_emb(q, k, cos, sin, [16, 24, 24])

        inputs = (
            torch.rand((1, 16, 348, 128), dtype=torch.float16),
            torch.rand((1, 2, 348, 128), dtype=torch.float16),
            torch.rand((3, 1, 348, 128), dtype=torch.float16),
            torch.rand((3, 1, 348, 128), dtype=torch.float16),
        )
        model = Model()
        ds = (
            {0: "a", 1: "b", 2: "c"},
            {0: "a", 1: "e", 2: "c"},
            {2: "c"},
            {2: "c"},
        )
        with torch_export_patches(patch_torch=False, stop_if_static=2):
            onx = to_onnx(model, inputs, dynamic_shapes=ds, exporter="onnx-dynamo")

        proto = onx.model_proto
        self.dump_onnx("test_qwen_apply_multimodal_rotary_pos_emb.onnx", proto)
        self.assert_onnx_disc(
            "test_qwen_apply_multimodal_rotary_pos_emb",
            proto,
            model,
            inputs,
            verbose=1,
            atol=1e-3,
            rtol=1,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
