import unittest
import torch
from onnx_diagnostic.ext_test_case import ExtTestCase, hide_stdout, has_transformers
from onnx_diagnostic.torch_models.hghub.model_inputs import get_untrained_model_with_inputs
from onnx_diagnostic.torch_export_patches import bypass_export_some_errors
from onnx_diagnostic.torch_export_patches.patch_inputs import use_dyn_not_str


class TestTasks(ExtTestCase):
    @hide_stdout()
    def test_image_classification(self):
        mid = "hf-internal-testing/tiny-random-BeitForImageClassification"
        data = get_untrained_model_with_inputs(mid, verbose=1, add_second_input=True)
        self.assertEqual(data["task"], "image-classification")
        self.assertIn((data["size"], data["n_weights"]), [(56880, 14220)])
        model, inputs, ds = data["model"], data["inputs"], data["dynamic_shapes"]
        model(**inputs)
        model(**data["inputs2"])
        if not has_transformers("4.51.999"):
            raise unittest.SkipTest("Requires transformers>=4.52")
        with bypass_export_some_errors(patch_transformers=True, verbose=10):
            torch.export.export(
                model, (), kwargs=inputs, dynamic_shapes=use_dyn_not_str(ds), strict=False
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
