import unittest
import torch
from onnx_diagnostic.ext_test_case import ExtTestCase, hide_stdout, requires_torch
from onnx_diagnostic.torch_models.hghub.model_inputs import get_untrained_model_with_inputs
from onnx_diagnostic.torch_export_patches import torch_export_patches
from onnx_diagnostic.torch_export_patches.patch_inputs import use_dyn_not_str


class TestTasksZeroShotImageClassification(ExtTestCase):
    @requires_torch("2.7.99")
    @hide_stdout()
    def test_zero_shot_image_classification(self):
        mid = "openai/clip-vit-base-patch16"
        data = get_untrained_model_with_inputs(mid, verbose=1, add_second_input=True)
        self.assertEqual(data["task"], "zero-shot-image-classification")
        self.assertIn((data["size"], data["n_weights"]), [(188872708, 47218177)])
        model, inputs, ds = data["model"], data["inputs"], data["dynamic_shapes"]
        expected = model(**inputs)
        model(**data["inputs2"])
        with torch_export_patches(patch_torch=True, patch_transformers=True, verbose=10):
            ep = torch.export.export(
                model, (), kwargs=inputs, dynamic_shapes=use_dyn_not_str(ds), strict=False
            )
            self.assertEqualAny(expected, ep.module()(**inputs))


if __name__ == "__main__":
    unittest.main(verbosity=2)
