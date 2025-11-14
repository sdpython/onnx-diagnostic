import unittest
import torch
from onnx_diagnostic.ext_test_case import ExtTestCase, hide_stdout, requires_transformers
from onnx_diagnostic.helpers.torch_helper import torch_deepcopy
from onnx_diagnostic.torch_models.hghub.model_inputs import get_untrained_model_with_inputs
from onnx_diagnostic.torch_export_patches import torch_export_patches
from onnx_diagnostic.torch_export_patches.patch_inputs import use_dyn_not_str


class TestTasksFeatureExtration(ExtTestCase):
    @hide_stdout()
    @requires_transformers("4.53.99")
    def test_feature_extraction_bart_base(self):
        """
        ata=dict(
            input_ids:T7s2x12,
            attention_mask:T7s2x12,
            past_key_values:EncoderDecoderCache(
                self_attention_cache=DynamicCache(
                      key_cache=#6[T1s2x12x30x64,...
                    value_cache=#6[T1s2x12x30x64,...
                cross_attention_cache=DynamicCache(
                      key_cache=#6[T1s2x12x4x64
                    value_cache=#6[T1s2x12x4x64
        """
        mid = "facebook/bart-base"
        data = get_untrained_model_with_inputs(mid, verbose=1, add_second_input=True)
        self.assertEqual(data["task"], "feature-extraction")
        self.assertIn((data["size"], data["n_weights"]), [(409583616, 102395904)])
        model, inputs, ds = data["model"], data["inputs"], data["dynamic_shapes"]
        print(f"-- {self.string_type(inputs, with_shape=True)}")
        model(**torch_deepcopy(inputs))
        model(**data["inputs2"])
        with torch_export_patches(patch_transformers=True, verbose=10):
            torch.export.export(
                model, (), kwargs=inputs, dynamic_shapes=use_dyn_not_str(ds), strict=False
            )

    @hide_stdout()
    def test_feature_extraction_tiny_bart(self):
        mid = "hf-tiny-model-private/tiny-random-PLBartForConditionalGeneration"
        data = get_untrained_model_with_inputs(mid, verbose=1, add_second_input=True)
        self.assertEqual(data["task"], "text2text-generation")
        self.assertIn((data["size"], data["n_weights"]), [(3243392, 810848)])
        model, inputs, ds = data["model"], data["inputs"], data["dynamic_shapes"]
        model(**inputs)
        model(**data["inputs2"])
        with torch_export_patches(patch_transformers=True, verbose=10):
            torch.export.export(
                model, (), kwargs=inputs, dynamic_shapes=use_dyn_not_str(ds), strict=False
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
