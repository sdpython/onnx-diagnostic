import unittest
import transformers
from onnx_diagnostic.ext_test_case import ExtTestCase, requires_transformers
from onnx_diagnostic.torch_export_patches import torch_export_patches
from onnx_diagnostic.torch_export_patches.patch_details import PatchDetails
from onnx_diagnostic.torch_export_patches.patches.patch_transformers import patched_eager_mask


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
        self.assertEqual(len(data), details.n_patches)
        for kind, f1, f2 in details.patched:
            raw = details.format_diff(f1, f2, kind=kind, format="raw")
            if callable(f1):
                self.assertIn(f1.__name__, raw)
            self.assertIn(f2.__name__, raw)
            rst = details.format_diff(f1, f2, kind=kind, format="rst")
            self.assertIn("====", rst)

    @requires_transformers("4.55")
    def test_patch_diff(self):
        eager_mask = transformers.masking_utils.eager_mask
        self.assertEqual(eager_mask.__name__, "eager_mask")
        self.assertEqual(patched_eager_mask.__name__, "patched_eager_mask")
        diff = PatchDetails().format_diff(eager_mask, patched_eager_mask, format="rst")
        self.assertIn("+    # PATCHED:", diff)


if __name__ == "__main__":
    unittest.main(verbosity=2)
