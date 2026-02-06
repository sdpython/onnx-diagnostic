import unittest
import torch
from onnx_diagnostic.ext_test_case import (
    ExtTestCase,
    hide_stdout,
    requires_transformers,
    requires_torch,
)
from onnx_diagnostic.helpers.torch_helper import torch_deepcopy
from onnx_diagnostic.torch_models.hghub.model_inputs import get_untrained_model_with_inputs
from onnx_diagnostic.torch_export_patches import torch_export_patches
from onnx_diagnostic.torch_export_patches.patch_inputs import use_dyn_not_str
from onnx_diagnostic.helpers.cache_helper import get_make_hybrid_cache


class TestTasksImageTextToText(ExtTestCase):
    @hide_stdout()
    @requires_transformers("5.1.999")
    @requires_torch("2.7.99")
    def test_image_text_to_text_idefics(self):
        mid = "HuggingFaceM4/tiny-random-idefics"
        data = get_untrained_model_with_inputs(mid, verbose=1, add_second_input=True)
        self.assertEqual(data["task"], "image-text-to-text")
        model, inputs, ds = data["model"], data["inputs"], data["dynamic_shapes"]
        expected = model(**torch_deepcopy(inputs))
        model(**data["inputs2"])
        with torch_export_patches(patch_transformers=True, verbose=10, patch_torch=False):
            ep = torch.export.export(
                model, (), kwargs=inputs, dynamic_shapes=use_dyn_not_str(ds), strict=False
            )
            # The conversion does not work. Tolerance is set to 1.
            self.assertEqualAny(expected, ep.module()(**inputs), atol=1)

    @hide_stdout()
    @requires_transformers("5.1.99")
    @requires_torch("2.7.99")
    def test_image_text_to_text_tiny_gemma3(self):
        """
        If the model tails because of
        ``if inputs_embeds[special_image_mask].numel() != image_features.numel():```,
        make sure this PR was merged:
        https://github.com/huggingface/transformers/pull/39962.
        """
        mid = "tiny-random/gemma-3"
        data = get_untrained_model_with_inputs(mid, verbose=1, add_second_input=True)
        self.assertEqual(data["task"], "image-text-to-text")
        # self.assertIn((data["size"], data["n_weights"]), [(17248576, 4312144)])
        model, inputs, ds = data["model"], data["inputs"], data["dynamic_shapes"]
        print("--", self.string_type(data["inputs"], with_shape=True))
        expected = model(**torch_deepcopy(inputs))
        model(**data["inputs2"])
        with torch_export_patches(patch_transformers=True, verbose=10):
            ep = torch.export.export(
                model, (), kwargs=inputs, dynamic_shapes=use_dyn_not_str(ds), strict=False
            )
            self.assertEqualAny(expected, ep.module()(**inputs))

    @hide_stdout()
    @requires_transformers("4.56.99")
    @requires_torch("2.8.99")
    def test_image_text_to_text_gemma3_4b_it(self):
        make_hybrid_cache = get_make_hybrid_cache()
        if make_hybrid_cache is None:
            raise unittest.SkipTest("not implemented yet for transformers>=5")
        mid = "google/gemma-3-4b-it"
        data = get_untrained_model_with_inputs(
            mid,
            verbose=1,
            add_second_input=False,
            # inputs_kwargs={
            #    "sequence_length": 281,
            #    "batch_size": 1,
            #    "max_sequence_length": 580,
            #    "n_images": 1,
            # },
        )
        self.assertEqual(data["task"], "image-text-to-text")
        # self.assertIn((data["size"], data["n_weights"]), [(17248576, 4312144)])
        model, inputs, ds = data["model"], data["inputs"], data["dynamic_shapes"]
        # inputs.pop("attention_mask")
        # ds.pop("attention_mask")
        expected = model(**torch_deepcopy(inputs))
        with torch_export_patches(patch_transformers=True, verbose=10):
            ep = torch.export.export(
                model, (), kwargs=inputs, dynamic_shapes=use_dyn_not_str(ds), strict=False
            )
            # The conversion does not work. Tolerance is set to 1.
            self.assertEqualAny(expected, ep.module()(**inputs))

    @hide_stdout()
    @requires_transformers("5.1.99")
    @requires_torch("2.7.99")
    def test_image_text_to_text_zai_glm(self):
        """
        If the model tails because of
        ``if inputs_embeds[special_image_mask].numel() != image_features.numel():```,
        make sure this PR was merged:
        https://github.com/huggingface/transformers/pull/39962.
        """
        mid = "zai-org/GLM-4.5V"
        data = get_untrained_model_with_inputs(mid, verbose=1, add_second_input=True)
        self.assertEqual(data["task"], "image-text-to-text")
        model, inputs, ds = data["model"], data["inputs"], data["dynamic_shapes"]
        print("--", self.string_type(data["inputs"], with_shape=True))
        expected = model(**torch_deepcopy(inputs))
        model(**data["inputs2"])
        with torch_export_patches(patch_transformers=True, verbose=10):
            ep = torch.export.export(
                model, (), kwargs=inputs, dynamic_shapes=use_dyn_not_str(ds), strict=False
            )
            self.assertEqualAny(expected, ep.module()(**inputs))


if __name__ == "__main__":
    unittest.main(verbosity=2)
