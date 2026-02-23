import unittest
import torch
import transformers
import transformers.integrations.sdpa_attention as sdpa_attention
import onnx
import onnx_diagnostic.torch_export_patches.patches.patch_transformers as patch_transformers
from onnx_diagnostic.ext_test_case import (
    ExtTestCase,
    requires_cuda,
    requires_onnxruntime,
    requires_transformers,
    requires_torch,
    ignore_warnings,
    has_onnxscript,
    has_transformers,
    requires_onnxscript,
)
from onnx_diagnostic.helpers.torch_helper import torch_deepcopy, fake_torchdynamo_exporting
from onnx_diagnostic.export.shape_helper import make_fake_with_dynamic_dimensions
from onnx_diagnostic.torch_models.hghub.hub_api import get_cached_configuration
from onnx_diagnostic.torch_export_patches import torch_export_patches
from onnx_diagnostic.torch_export_patches.patch_inputs import use_dyn_not_str
from onnx_diagnostic.torch_models.hghub import get_untrained_model_with_inputs
from onnx_diagnostic.torch_export_patches.patches.patch_transformers import (
    patch_qwen2_5,
    patch_funnel,
)
from onnx_diagnostic.export.api import to_onnx


class TestPatchPatchTransformers(ExtTestCase):
    @requires_transformers("4.55")
    @unittest.skipIf(
        not hasattr(transformers, "masking_utils")
        or not hasattr(transformers.masking_utils, "sdpa_mask_recent_torch"),
        "removed in transformers==5.0",
    )
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

    @requires_transformers("4.99")
    def test_sdpa_mask_patched(self):
        sdpa_mask = transformers.masking_utils.sdpa_mask
        patched_sdpa_mask = patch_transformers.patched_sdpa_mask
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
        expected = sdpa_mask(**kwargs)
        got = patched_sdpa_mask(**kwargs)
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
        expected = sdpa_mask(**kwargs)
        got = patched_sdpa_mask(**kwargs)
        self.assertEqualArray(expected, got)

    @requires_transformers("4.99")
    def test_sdpa_mask_recent_torch_is_running(self):
        def _copy_vmap_for_bhqkv(mask_function, bh_indices=True):
            dimensions = [(None, None, None, 0), (None, None, 0, None)]
            if bh_indices:
                dimensions.extend([(None, 0, None, None), (0, None, None, None)])
            for dims in dimensions:
                mask_function = torch.vmap(mask_function, in_dims=dims, out_dims=0)
            return mask_function

        def copy_of_sdpa_mask_recent_torch(
            batch_size,
            cache_position,
            kv_length,
            kv_offset=0,
            mask_function=transformers.masking_utils.causal_mask_function,
            attention_mask=None,
            local_size=None,
            allow_is_causal_skip=True,
            **kwargs,
        ):
            q_length = cache_position.shape[0]
            padding_mask = transformers.masking_utils.prepare_padding_mask(
                attention_mask, kv_length, kv_offset
            )
            if allow_is_causal_skip and transformers.masking_utils._ignore_causal_mask_sdpa(
                padding_mask, q_length, kv_length, kv_offset, local_size
            ):
                return None
            kv_arange = torch.arange(kv_length, device=cache_position.device)
            kv_arange += kv_offset
            if padding_mask is not None:
                mask_function = transformers.masking_utils.and_masks(
                    mask_function,
                    transformers.masking_utils.padding_mask_function(padding_mask),
                )

            batch_arange = torch.arange(batch_size, device=cache_position.device)
            head_arange = torch.arange(1, device=cache_position.device)
            with transformers.masking_utils.TransformGetItemToIndex():
                causal_mask = _copy_vmap_for_bhqkv(mask_function)(
                    batch_arange, head_arange, cache_position, kv_arange
                )
            return causal_mask

        sdpa_mask_recent_torch = copy_of_sdpa_mask_recent_torch
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
    @requires_onnxscript("0.6.2")
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
            [inputs],
            verbose=1,
            atol=1e-3,
            rtol=1,
        )

    @requires_transformers("4.55")
    @requires_onnxscript("0.6.2")
    @unittest.skipIf(not patch_qwen2_5, "Qwen25 not part of this transformers")
    def test_qwen_function_proto(self):
        from onnx_diagnostic.torch_export_patches.patches._patch_transformers_qwen2_5 import (
            LoopAttention23,
            LoopMHAAttention,
            PackedAttention,
        )

        LoopMHAAttention.to_function_proto()
        LoopAttention23.to_function_proto()
        PackedAttention.to_function_proto()

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
    # @unittest.skipIf(not patch_qwen2_5, "Qwen25 not part of this transformers")
    # see https://github.com/huggingface/transformers/pull/42564/files#diff-09bc594f9680f1d042fd485106c68022d77b59831697a00b3b38f12a3e40f395
    @unittest.skip(
        "vision_outputs = self.visual(pixel_values, "
        "grid_thw=image_grid_thw, return_dict=True, **kwargs)"
    )
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
        self.assertEqualAny(expected, got)

    @classmethod
    def _get_cu_seqlens(cls):
        return torch.tensor(
            [
                0,
                64,
                128,
                192,
                256,
                304,
                368,
                432,
                496,
                560,
                608,
                672,
                736,
                800,
                864,
                912,
                976,
                1040,
                1104,
                1168,
                1216,
                1232,
                1248,
                1264,
                1280,
                1292,
            ],
            dtype=torch.int64,
        )

    @requires_transformers("4.55")
    @unittest.skipIf(not patch_qwen2_5, "Qwen25 not part of this transformers")
    def test_patched_qwen2_5_vl_vision_attention_forward(self):
        from onnx_diagnostic.torch_export_patches.patches.patch_helper import (
            _is_torchdynamo_exporting,
        )
        from onnx_diagnostic.torch_export_patches.patches.patch_transformers import (
            patched_Qwen2_5_VLVisionAttention,
            PLUGS_Qwen25,
        )

        config = get_cached_configuration("Qwen/Qwen2.5-VL-7B-Instruct")
        config._attn_implementation = "sdpa"
        patched_class = patched_Qwen2_5_VLVisionAttention._PATCHED_CLASS_
        instance = patched_class(config.vision_config)
        inputs = dict(
            hidden_states=torch.rand((1292, 1280), dtype=torch.float32),
            cu_seqlens=self._get_cu_seqlens(),
            position_embeddings=(  # position_embeddings = cos, sin
                torch.rand((1292, 80), dtype=torch.float32),
                torch.rand((1292, 80), dtype=torch.float32),
            ),
        )
        expected = instance.forward(**inputs)
        got = patched_Qwen2_5_VLVisionAttention.forward(instance, **inputs)
        self.assertEqualArray(expected, got)
        with fake_torchdynamo_exporting():
            assert (
                _is_torchdynamo_exporting()
            ), f"exporting is not set to true? {torch.compiler.is_exporting_flag}"
            got = patched_Qwen2_5_VLVisionAttention.forward(instance, **inputs)
            self.assertEqualArray(expected, got, atol=1e-2)

        class Model(patched_class):
            def forward(
                self,
                hidden_states: torch.Tensor,
                cu_seqlens: torch.Tensor,
                rotary_pos_emb: torch.Tensor | None = None,
                position_embeddings1: torch.Tensor | None = None,
                position_embeddings2: torch.Tensor | None = None,
                **kwargs,
            ) -> torch.Tensor:
                return patched_Qwen2_5_VLVisionAttention.forward(
                    self,
                    hidden_states,
                    cu_seqlens,
                    rotary_pos_emb=rotary_pos_emb,
                    position_embeddings=(position_embeddings1, position_embeddings2),
                    **kwargs,
                )

        instance = Model(config.vision_config)
        instance.eval()

        ds = dict(
            hidden_states={0: "d1"},
            cu_seqlens={0: "d3"},
            position_embeddings1={0: "d1"},
            position_embeddings2={0: "d1"},
        )
        inputs.update(
            dict(
                position_embeddings1=inputs["position_embeddings"][0],
                position_embeddings2=inputs["position_embeddings"][1],
            )
        )
        del inputs["position_embeddings"]
        for exporter in ("custom", "onnx-dynamo"):
            # onnx-dynamo needs OpOverload(op='aten.sym_storage_offset' (transformers>=5.0?)
            if exporter == "onnx-dynamo" and not has_onnxscript("0.5.7"):
                raise unittest.SkipTest("needs onnxscript>=0.5.7")
            filename = self.get_dump_file(
                f"test_patched_qwen2_5_vl_vision_attention_forward.{exporter}.onnx"
            )
            to_onnx(
                instance,
                kwargs=inputs,
                dynamic_shapes=ds,
                exporter=exporter,
                filename=filename,
                onnx_plugs=PLUGS_Qwen25,
                target_opset=22,
            )
            # exporter_kwargs={"report":True} if exporter != "custom" else {}
            if torch.cuda.is_available():
                self.assert_onnx_disc(
                    f"test_patched_qwen2_5_vl_vision_attention_forward-{exporter}",
                    onnx.load(filename),
                    instance,
                    inputs,
                    atol=1e-3,
                    rtol=1,
                    providers=["CUDAExecutionProvider"],
                )
        self.clean_dump()

    @requires_transformers("4.99")
    @requires_torch("2.9.99")
    @unittest.skipIf(not patch_qwen2_5, "Qwen25 not part of this transformers")
    def test_qwen2_5_vl_vision_attention_iteration(self):
        from onnx_diagnostic.torch_export_patches.patches.patch_transformers import (
            patched_Qwen2_5_VLVisionAttentionOneIteration,
        )

        try:
            from onnxscript.function_libs.torch_lib.ops.core import aten_sym_storage_offset
        except ImportError:
            aten_sym_storage_offset = None

        model = patched_Qwen2_5_VLVisionAttentionOneIteration()
        inputs = (
            torch.tensor([736, 800], dtype=torch.int64),
            torch.rand((1, 16, 1292, 80), dtype=torch.float16),
            torch.rand((1, 16, 1292, 80), dtype=torch.float16),
            torch.rand((1, 16, 1292, 80), dtype=torch.float16),
        )
        ds = (
            {},
            {0: "batch", 1: "length", 2: "dim"},
            {0: "batch", 1: "length", 2: "dim"},
            {0: "batch", 1: "length", 2: "dim"},
        )
        for exporter in ("custom", "onnx-dynamo"):
            if exporter == "onnx-dynamo" and aten_sym_storage_offset is None:
                raise unittest.SkipTest("update onnxscript to make this test run")
            # onnx-dynamo needs OpOverload(op='aten.sym_storage_offset' (transformers>=5.0?)
            filename = self.get_dump_file(
                f"test_qwen2_5_vl_vision_attention_iteration.{exporter}.onnx"
            )
            to_onnx(model, inputs, dynamic_shapes=ds, exporter=exporter, filename=filename)
            # exporter_kwargs={"report":True} if exporter != "custom" else {}
            self.assert_onnx_disc(
                f"test_qwen2_5_vl_vision_attention_iteration-{exporter}",
                onnx.load(filename),
                model,
                inputs,
                atol=1e-3,
                rtol=1,
            )
        self.clean_dump()

    @classmethod
    def _get_seqlen(cls) -> torch.Tensor:
        return torch.tensor(
            [
                0,
                64,
                128,
                192,
                256,
                304,
                368,
                432,
                496,
                560,
                608,
                672,
                736,
                800,
                864,
                912,
                976,
                1040,
                1104,
                1168,
                1216,
                1232,
                1248,
                1264,
                1280,
                1292,
            ],
            dtype=torch.int64,
        )

    @unittest.skipIf(not patch_qwen2_5, "Qwen25 not part of this transformers")
    @requires_cuda()
    def test_plug_multi_head_attention_qwen25_packed_float16(self):
        from onnx_diagnostic.torch_export_patches.patches._patch_transformers_qwen2_5 import (
            qwen_sdpa_attention_versatile as qwen_sdpa_attention_packed_versatile,
        )

        with self.set_env("QWEN25ATTENTION", "PACKED"):
            inputs = (
                torch.rand((1, 16, 1292, 80), dtype=torch.float16).to("cuda"),
                torch.rand((1, 16, 1292, 80), dtype=torch.float16).to("cuda"),
                torch.rand((1, 16, 1292, 80), dtype=torch.float16).to("cuda"),
                self._get_seqlen().to("cuda"),
            )

            results = qwen_sdpa_attention_packed_versatile.verify(
                *inputs, scaling=0.5, num_heads=16
            )
            self.assertEqual(len(results.eager_outputs), len(results.onnx_outputs))
            self.assertEqual(len(results.eager_outputs), len(results.diffs))
            self.assertEqualArray(results.eager_outputs[0], results.onnx_outputs[0], atol=0.01)
            self.assertLess(results.diffs[0]["abs"], 0.01)

            results = qwen_sdpa_attention_packed_versatile.verify(
                *inputs, scaling=0.11180339887498948, num_heads=16
            )
            self.assertEqual(len(results.eager_outputs), len(results.onnx_outputs))
            self.assertEqual(len(results.eager_outputs), len(results.diffs))
            self.assertEqualArray(results.eager_outputs[0], results.onnx_outputs[0], atol=0.01)
            self.assertLess(results.diffs[0]["abs"], 0.01)

    @requires_onnxruntime("1.25")
    @unittest.skipIf(not patch_qwen2_5, "Qwen25 not part of this transformers")
    def test_plug_multi_head_attention_qwen25_loopmha_float16(self):
        from onnx_diagnostic.torch_export_patches.patches._patch_transformers_qwen2_5 import (
            qwen_sdpa_attention_versatile as qwen_sdpa_attention_loopmha_versatile,
        )

        with self.set_env("QWEN25ATTENTION", "LOOPMHA"):
            inputs = (
                torch.rand((1, 16, 1292, 80), dtype=torch.float16),
                torch.rand((1, 16, 1292, 80), dtype=torch.float16),
                torch.rand((1, 16, 1292, 80), dtype=torch.float16),
                self._get_seqlen(),
            )

            results = qwen_sdpa_attention_loopmha_versatile.verify(
                *inputs,
                scaling=0.5,
                num_heads=16,
                dump_onnx_model=self.get_dump_file(
                    "test_plug_packed_multi_head_attention_qwen25_loopmha_float16.onnx"
                ),
            )
            self.assertEqual(len(results.eager_outputs), len(results.onnx_outputs))
            self.assertEqual(len(results.eager_outputs), len(results.diffs))
            self.assertEqualArray(results.eager_outputs[0], results.onnx_outputs[0], atol=0.01)
            self.assertLess(results.diffs[0]["abs"], 0.01)

            results = qwen_sdpa_attention_loopmha_versatile.verify(
                *inputs, scaling=0.11180339887498948, num_heads=16
            )
            self.assertEqual(len(results.eager_outputs), len(results.onnx_outputs))
            self.assertEqual(len(results.eager_outputs), len(results.diffs))
            self.assertEqualArray(results.eager_outputs[0], results.onnx_outputs[0], atol=0.01)
            self.assertLess(results.diffs[0]["abs"], 0.01)

    @requires_onnxruntime("1.25")
    @unittest.skipIf(not patch_qwen2_5, "Qwen25 not part of this transformers")
    def test_plug_multi_head_attention_qwen25_loopmha_float32(self):
        from onnx_diagnostic.torch_export_patches.patches._patch_transformers_qwen2_5 import (
            qwen_sdpa_attention_versatile as qwen_sdpa_attention_loopmha_versatile,
        )

        with self.set_env("QWEN25ATTENTION", "LOOPMHA"):
            inputs = (
                torch.rand((1, 16, 1292, 80), dtype=torch.float32),
                torch.rand((1, 16, 1292, 80), dtype=torch.float32),
                torch.rand((1, 16, 1292, 80), dtype=torch.float32),
                self._get_seqlen(),
            )

            results = qwen_sdpa_attention_loopmha_versatile.verify(
                *inputs,
                scaling=0.5,
                num_heads=16,
                dump_onnx_model=self.get_dump_file(
                    "test_plug_packed_multi_head_attention_qwen25_loopmha_float16.onnx"
                ),
            )
            self.assertEqual(len(results.eager_outputs), len(results.onnx_outputs))
            self.assertEqual(len(results.eager_outputs), len(results.diffs))
            self.assertEqualArray(results.eager_outputs[0], results.onnx_outputs[0], atol=1e-5)
            self.assertLess(results.diffs[0]["abs"], 1e-5)

            results = qwen_sdpa_attention_loopmha_versatile.verify(
                *inputs, scaling=0.11180339887498948, num_heads=16
            )
            self.assertEqual(len(results.eager_outputs), len(results.onnx_outputs))
            self.assertEqual(len(results.eager_outputs), len(results.diffs))
            self.assertEqualArray(results.eager_outputs[0], results.onnx_outputs[0], atol=1e-5)
            self.assertLess(results.diffs[0]["abs"], 1e-5)

    @requires_onnxruntime("1.25")
    @unittest.skipIf(not patch_qwen2_5, "Qwen25 not part of this transformers")
    def test_plug_multi_head_attention_qwen25_loopa24_float16(self):
        from onnx_diagnostic.torch_export_patches.patches._patch_transformers_qwen2_5 import (
            qwen_sdpa_attention_versatile as qwen_sdpa_attention_loopa24_versatile,
        )

        with self.set_env("QWEN25ATTENTION", "LOOO24"):
            inputs = (
                torch.rand((1, 16, 1292, 80), dtype=torch.float16),
                torch.rand((1, 16, 1292, 80), dtype=torch.float16),
                torch.rand((1, 16, 1292, 80), dtype=torch.float16),
                self._get_seqlen(),
            )

            results = qwen_sdpa_attention_loopa24_versatile.verify(*inputs, scaling=0.5)
            self.assertEqual(len(results.eager_outputs), len(results.onnx_outputs))
            self.assertEqual(len(results.eager_outputs), len(results.diffs))
            self.assertEqualArray(results.eager_outputs[0], results.onnx_outputs[0], atol=1e-2)
            self.assertLess(results.diffs[0]["abs"], 1e-2)

            results = qwen_sdpa_attention_loopa24_versatile.verify(
                *inputs, scaling=0.11180339887498948
            )
            self.assertEqual(len(results.eager_outputs), len(results.onnx_outputs))
            self.assertEqual(len(results.eager_outputs), len(results.diffs))
            self.assertEqualArray(
                results.eager_outputs[0], results.onnx_outputs[0], atol=0.005
            )
            self.assertLess(results.diffs[0]["abs"], 0.005)

    @requires_onnxruntime("1.25")
    @unittest.skipIf(not patch_qwen2_5, "Qwen25 not part of this transformers")
    def test_plug_multi_head_attention_qwen25_loopa24_float32(self):
        from onnx_diagnostic.torch_export_patches.patches._patch_transformers_qwen2_5 import (
            qwen_sdpa_attention_versatile as qwen_sdpa_attention_loopa24_versatile,
        )

        with self.set_env("QWEN25ATTENTION", "LOOO24"):
            inputs = (
                torch.rand((1, 16, 1292, 80), dtype=torch.float32),
                torch.rand((1, 16, 1292, 80), dtype=torch.float32),
                torch.rand((1, 16, 1292, 80), dtype=torch.float32),
                self._get_seqlen(),
            )

            results = qwen_sdpa_attention_loopa24_versatile.verify(*inputs, scaling=0.5)
            self.assertEqual(len(results.eager_outputs), len(results.onnx_outputs))
            self.assertEqual(len(results.eager_outputs), len(results.diffs))
            self.assertEqualArray(results.eager_outputs[0], results.onnx_outputs[0], atol=1e-5)
            self.assertLess(results.diffs[0]["abs"], 1e-5)

            results = qwen_sdpa_attention_loopa24_versatile.verify(
                *inputs, scaling=0.11180339887498948
            )
            self.assertEqual(len(results.eager_outputs), len(results.onnx_outputs))
            self.assertEqual(len(results.eager_outputs), len(results.diffs))
            self.assertEqualArray(results.eager_outputs[0], results.onnx_outputs[0], atol=1e-5)
            self.assertLess(results.diffs[0]["abs"], 1e-5)

    @unittest.skipIf(not patch_funnel, "Funnel not part of this transformers")
    def test_model_funnel(self):
        from onnx_diagnostic.torch_export_patches.patches.patch_transformers import (
            patched_FunnelAttentionStructure,
            patched_FunnelRelMultiheadAttention,
        )

        pos = torch.tensor([0, 4, 5, 8], dtype=torch.long)
        stride = 2
        config = transformers.models.funnel.modeling_funnel.FunnelConfig()
        original = transformers.models.funnel.modeling_funnel.FunnelAttentionStructure(config)
        patched = patched_FunnelAttentionStructure()
        self.assertEqualArray(
            original.relative_pos(pos, stride=stride), patched.relative_pos(pos, stride=stride)
        )

        rmha = transformers.models.funnel.modeling_funnel.FunnelRelMultiheadAttention(
            config, 2
        )
        patched = patched_FunnelRelMultiheadAttention()
        patched.config = config
        for att in ["block_index", "r_r_bias", "scale", "r_kernel"]:
            setattr(patched, att, getattr(rmha, att))
        inputs = dict(
            position_embeds=[
                [torch.rand((24, 768)), None],
                [torch.rand((12, 768)), torch.rand((24, 768))],
                [torch.rand((6, 768)), torch.rand((12, 768))],
            ],
            q_head=torch.rand((2, 12, 12, 64)),
            context_len=12,
        )
        expected = rmha.relative_positional_attention(**inputs)
        got = patched.relative_positional_attention(**inputs)
        self.assertEqualArray(expected, got)

    def test_cache_dependant_input_preparation_exporting(self):
        from onnx_diagnostic.torch_export_patches.patches._patch_transformers_generation_mixin import (  # noqa: E501
            patched_GenerationMixin as GenerationMixin,
        )

        with self.subTest(case="case1"):
            input_ids = torch.randint(0, 16, (2, 8), dtype=torch.int64)[:, :0]
            inputs_embeds = torch.rand((2, 8), dtype=torch.float32)
            cache_position = torch.arange(0, 8, dtype=torch.int64)
            eager1, eager2 = GenerationMixin()._cache_dependant_input_preparation(
                input_ids, inputs_embeds, cache_position
            )
            export1, export2 = GenerationMixin()._cache_dependant_input_preparation_exporting(
                input_ids, inputs_embeds, cache_position
            )
            torch.testing.assert_close(eager1, export1)
            torch.testing.assert_close(eager2, export2)

        with self.subTest(case="case2"):
            raise unittest.SkipTest("torch 2.10+ has probably a bug here.")
            input_ids = torch.randint(0, 16, (2, 8), dtype=torch.int64)
            inputs_embeds = torch.rand((2, 8), dtype=torch.float32)
            cache_position = torch.arange(0, 8, dtype=torch.int64)
            eager1, eager2 = GenerationMixin()._cache_dependant_input_preparation(
                input_ids, inputs_embeds, cache_position
            )
            export1, export2 = GenerationMixin()._cache_dependant_input_preparation_exporting(
                input_ids, inputs_embeds, cache_position
            )
            torch.testing.assert_close(eager1, export1)
            torch.testing.assert_close(eager2, export2)

        with self.subTest(case="case3"):
            input_ids = torch.randint(0, 16, (2, 12), dtype=torch.int64)
            inputs_embeds = None
            cache_position = torch.arange(0, 8, dtype=torch.int64)
            eager1, eager2 = GenerationMixin()._cache_dependant_input_preparation(
                input_ids, inputs_embeds, cache_position
            )
            export1, export2 = GenerationMixin()._cache_dependant_input_preparation_exporting(
                input_ids, inputs_embeds, cache_position
            )
            torch.testing.assert_close(eager1, export1)
            torch.testing.assert_close(eager2, export2)

        with self.subTest(case="case4"):
            input_ids = torch.randint(0, 16, (2, 8), dtype=torch.int64)
            inputs_embeds = None
            cache_position = torch.arange(0, 8, dtype=torch.int64)
            eager1, eager2 = GenerationMixin()._cache_dependant_input_preparation(
                input_ids, inputs_embeds, cache_position
            )
            export1, export2 = GenerationMixin()._cache_dependant_input_preparation_exporting(
                input_ids, inputs_embeds, cache_position
            )
            torch.testing.assert_close(eager1, export1)
            torch.testing.assert_close(eager2, export2)

    @requires_transformers("4.57")
    def test_prepare_inputs_for_generation_decoder_llm(self):
        data = get_untrained_model_with_inputs(
            "hf-internal-testing/tiny-random-LlamaForCausalLM"
        )
        model = data["model"]
        config = model.config
        torch_device = "cpu"

        with torch_export_patches(patch_transformers=True):
            with self.subTest(case="case1"):
                self.assertTrue("GenerationMixin" in str(model.prepare_inputs_for_generation))

            input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]]).to(torch_device)
            cache_position = torch.arange(input_ids.shape[1], device=input_ids.device)

            with self.subTest(case="case2"):
                input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]]).to(torch_device)
                model_inputs = model.prepare_inputs_for_generation(
                    input_ids, cache_position=cache_position
                )
                self.assertTrue(torch.all(model_inputs["input_ids"] == input_ids))

            with self.subTest(case="case3"):
                attention_mask = torch.tensor([[1, 1, 1], [1, 1, 1]]).to(torch_device)
                model_inputs = model.prepare_inputs_for_generation(
                    input_ids, attention_mask=attention_mask, cache_position=cache_position
                )
                self.assertTrue(torch.all(model_inputs["attention_mask"] == attention_mask))
                self.assertTrue(model_inputs["position_ids"].shape == input_ids.shape)

            with self.subTest(case="case4"):
                self.assertFalse("use_cache" in model_inputs)
                model_inputs = model.prepare_inputs_for_generation(
                    input_ids, use_cache=True, foo="bar", cache_position=cache_position
                )
                self.assertTrue(model_inputs["use_cache"] is True)
                self.assertTrue(model_inputs["foo"] == "bar")

            init_input_ids = input_ids[:, :2]
            dynamic_cache = transformers.cache_utils.DynamicCache(config=config)
            dynamic_cache = model(
                init_input_ids, past_key_values=dynamic_cache
            ).past_key_values

            with self.subTest(case="case5"):
                if not has_transformers("4.57"):
                    raise unittest.SkipTest("transformers 4.57+.")
                if has_transformers("5.2.99"):
                    raise unittest.SkipTest("transformers 5.2+.")
                with self.assertRaises((AttributeError, TypeError)):
                    model_inputs = model.prepare_inputs_for_generation(
                        input_ids, past_key_values=dynamic_cache
                    )

            with self.subTest(case="case6"):
                cache_position = torch.arange(input_ids.shape[-1], dtype=torch.long).to(
                    torch_device
                )
                cache_position = cache_position[dynamic_cache.get_seq_length() :]
                model_inputs = model.prepare_inputs_for_generation(
                    input_ids,
                    past_key_values=dynamic_cache,
                    cache_position=cache_position,
                    attention_mask=attention_mask,
                )
                self.assertTrue("past_key_values" in model_inputs)
                self.assertTrue(torch.all(model_inputs["cache_position"] == cache_position))
                self.assertTrue(
                    model_inputs["input_ids"].shape[-1] == 1
                )  # 1 = 3 fed tokens - 2 tokens in the cache
                self.assertTrue(model_inputs["position_ids"].shape[-1] == 1)
                self.assertTrue(
                    model_inputs["attention_mask"].shape[-1] == 3
                )  # we still need the full attention mask!

            with self.subTest(case="case6.2"):
                max_cache_len = 10
                batch_size = 2
                query_length = input_ids.shape[-1] - init_input_ids.shape[-1]
                static_cache = transformers.cache_utils.StaticCache(
                    config=config, max_cache_len=max_cache_len
                )
                static_cache = model(
                    init_input_ids, past_key_values=static_cache
                ).past_key_values
                model_inputs = model.prepare_inputs_for_generation(
                    input_ids,
                    past_key_values=static_cache,
                    cache_position=cache_position,
                    attention_mask=attention_mask,
                )
                self.assertTrue("past_key_values" in model_inputs)
                self.assertTrue(
                    list(model_inputs["attention_mask"].shape)
                    == [batch_size, 1, query_length, max_cache_len]
                )

            with self.subTest(case="case7"):
                if not has_transformers("4.57"):
                    raise unittest.SkipTest("transformers 4.57+.")
                init_inputs_embeds = model.get_input_embeddings()(init_input_ids)
                model_inputs = model.prepare_inputs_for_generation(
                    input_ids,
                    past_key_values=dynamic_cache,
                    inputs_embeds=init_inputs_embeds,
                    cache_position=cache_position,
                )
                self.assertTrue(model_inputs["input_ids"] is not None)
                self.assertTrue(model_inputs["inputs_embeds"] is None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
