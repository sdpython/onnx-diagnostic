import unittest
import torch
from onnx_diagnostic.ext_test_case import (
    ExtTestCase,
    hide_stdout,
    requires_transformers,
    requires_torch,
)
from onnx_diagnostic.torch_models.hghub.model_inputs import get_untrained_model_with_inputs
from onnx_diagnostic.torch_export_patches import torch_export_patches
from onnx_diagnostic.torch_export_patches.patch_inputs import use_dyn_not_str


class TestTasks(ExtTestCase):
    @hide_stdout()
    @requires_transformers("4.52")
    @requires_torch("2.7.99")
    def test_image_text_to_text(self):
        mid = "HuggingFaceM4/tiny-random-idefics"
        data = get_untrained_model_with_inputs(mid, verbose=1, add_second_input=True)
        self.assertEqual(data["task"], "image-text-to-text")
        self.assertIn((data["size"], data["n_weights"]), [(12742888, 3185722)])
        model, inputs, ds = data["model"], data["inputs"], data["dynamic_shapes"]
        model(**inputs)
        model(**data["inputs2"])
        with torch_export_patches(patch_transformers=True, verbose=10):
            torch.export.export(
                model, (), kwargs=inputs, dynamic_shapes=use_dyn_not_str(ds), strict=False
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
