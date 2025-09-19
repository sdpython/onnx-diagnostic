import unittest
import torch
import transformers
from onnx_diagnostic.ext_test_case import (
    ExtTestCase,
    hide_stdout,
    requires_diffusers,
    requires_torch,
    requires_transformers,
)
from onnx_diagnostic.torch_models.hghub.model_inputs import get_untrained_model_with_inputs
from onnx_diagnostic.torch_export_patches import torch_export_patches
from onnx_diagnostic.torch_export_patches.patch_inputs import use_dyn_not_str


class TestTasksImageToVideo(ExtTestCase):
    @hide_stdout()
    @requires_diffusers("0.35")
    @requires_transformers("4.55")
    @requires_torch("2.8.99")
    def test_image_to_video(self):
        kwargs = {
            "_diffusers_version": "0.34.0.dev0",
            "_class_name": "CosmosTransformer3DModel",
            "max_size": [128, 240, 240],
            "text_embed_dim": 128,
            "use_cache": True,
            "in_channels": 3,
            "out_channels": 16,
            "num_layers": 2,
            "model_type": "dia",
            "patch_size": [1, 2, 2],
            "rope_scale": [1.0, 3.0, 3.0],
            "attention_head_dim": 16,
            "mlp_ratio": 0.4,
            "initializer_range": 0.02,
            "num_attention_heads": 16,
            "is_encoder_decoder": True,
            "adaln_lora_dim": 16,
            "concat_padding_mask": True,
            "extra_pos_embed_type": None,
        }
        config = transformers.DiaConfig(**kwargs)
        mid = "nvidia/Cosmos-Predict2-2B-Video2World"
        data = get_untrained_model_with_inputs(
            mid,
            verbose=1,
            add_second_input=True,
            subfolder="transformer",
            config=config,
            inputs_kwargs=dict(image_height=8 * 50, image_width=8 * 80),
        )
        self.assertEqual(data["task"], "image-to-video")
        model, inputs, ds = data["model"], data["inputs"], data["dynamic_shapes"]
        model(**inputs)
        model(**data["inputs2"])
        with torch.fx.experimental._config.patch(
            backed_size_oblivious=True
        ), torch_export_patches(
            patch_transformers=True, patch_diffusers=True, verbose=10, stop_if_static=1
        ):
            torch.export.export(
                model, (), kwargs=inputs, dynamic_shapes=use_dyn_not_str(ds), strict=False
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
