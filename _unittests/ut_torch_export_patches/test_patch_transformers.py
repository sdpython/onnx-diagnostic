import unittest
import torch
import transformers
import transformers.integrations.sdpa_attention as sdpa_attention
import onnx
import onnx_diagnostic.torch_export_patches.patches.patch_transformers as patch_transformers
from onnx_diagnostic.ext_test_case import (
    ExtTestCase,
    requires_cuda,
    requires_transformers,
    requires_torch,
    ignore_warnings,
    has_onnxscript,
)
from onnx_diagnostic.helpers.torch_helper import torch_deepcopy, fake_torchdynamo_exporting
from onnx_diagnostic.export.shape_helper import make_fake_with_dynamic_dimensions
from onnx_diagnostic.torch_models.hghub.hub_api import get_cached_configuration
from onnx_diagnostic.torch_export_patches import torch_export_patches
from onnx_diagnostic.torch_export_patches.patch_inputs import use_dyn_not_str
from onnx_diagnostic.torch_export_patches.patches.patch_transformers import patch_qwen2_5
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

    @unittest.skipIf(not patch_qwen2_5, "Qwen25 not part of this transformers")
    @requires_cuda()
    def test_plug_packed_multi_head_attention_qwen25_packed(self):
        from onnx_diagnostic.torch_export_patches.patches._patch_transformers_qwen2_5 import (
            qwen_sdpa_attention_packed_versatile,
        )

        inputs = (
            torch.rand((1, 16, 1292, 80), dtype=torch.float16).to("cuda"),
            torch.rand((1, 16, 1292, 80), dtype=torch.float16).to("cuda"),
            torch.rand((1, 16, 1292, 80), dtype=torch.float16).to("cuda"),
            torch.tensor(
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
            ).to("cuda"),
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

    @unittest.skipIf(not patch_qwen2_5, "Qwen25 not part of this transformers")
    def test_plug_packed_multi_head_attention_qwen25_loopmha(self):
        from onnx_diagnostic.torch_export_patches.patches._patch_transformers_qwen2_5 import (
            qwen_sdpa_attention_loopmha_versatile,
        )

        inputs = (
            torch.rand((1, 16, 1292, 80), dtype=torch.float16),
            torch.rand((1, 16, 1292, 80), dtype=torch.float16),
            torch.rand((1, 16, 1292, 80), dtype=torch.float16),
            torch.tensor(
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
            ),
        )

        results = qwen_sdpa_attention_loopmha_versatile.verify(
            *inputs,
            scaling=0.5,
            num_heads=16,
            dump_onnx_model=self.get_dump_file(
                "test_plug_packed_multi_head_attention_qwen25_loopmha.onnx"
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
