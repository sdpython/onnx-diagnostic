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


class TestTasksTextToTimage(ExtTestCase):
    @hide_stdout()
    @requires_transformers("4.52")
    @requires_torch("2.7.99")
    @ignore_errors(OSError)  # connectivity issues
    def test_text_to_image(self):
        mid = "diffusers/tiny-torch-full-checker"
        data = get_untrained_model_with_inputs(
            mid, verbose=1, add_second_input=True, subfolder="unet"
        )
        self.assertEqual(data["task"], "text-to-image")
        self.assertIn((data["size"], data["n_weights"]), [(5708048, 1427012)])
        model, inputs, ds = data["model"], data["inputs"], data["dynamic_shapes"]
        expected = model(**inputs)
        model(**data["inputs2"])
        with torch_export_patches(patch_transformers=True, verbose=10, stop_if_static=1):
            ep = torch.export.export(
                model, (), kwargs=inputs, dynamic_shapes=use_dyn_not_str(ds), strict=False
            )
            self.assertEqualAny(expected, ep.module()(**inputs))


if __name__ == "__main__":
    unittest.main(verbosity=2)
