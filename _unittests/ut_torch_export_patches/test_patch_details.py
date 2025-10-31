import unittest
from onnx_diagnostic.ext_test_case import ExtTestCase
from onnx_diagnostic.torch_export_patches import torch_export_patches
from onnx_diagnostic.torch_export_patches.patch_details import PatchDetails


class TestPatchDetails(ExtTestCase):
    def test_patch_details(self):
        details = PatchDetails()
        with torch_export_patches(
            patch_transformers=True,
            verbose=10,
            patch_torch=True,
            patch_diffusers=True,
            patch_details=details,
        ):
            pass
        self.assertGreater(details.n_patches, 1)
        data = details.data()
        print(data)


if __name__ == "__main__":
    unittest.main(verbosity=2)
