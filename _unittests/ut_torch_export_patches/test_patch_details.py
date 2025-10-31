import unittest
import torch
import transformers
from onnx_diagnostic.ext_test_case import ExtTestCase, requires_transformers, hide_stdout
from onnx_diagnostic.torch_export_patches import torch_export_patches
from onnx_diagnostic.torch_export_patches.patch_inputs import use_dyn_not_str
from onnx_diagnostic.torch_export_patches.patch_details import PatchDetails, PatchInfo
from onnx_diagnostic.torch_models.hghub import get_untrained_model_with_inputs


class TestPatchDetails(ExtTestCase):
    @hide_stdout()
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
        for patch in details.patched:
            _kind, f1, f2 = patch.family, patch.function_to_patch, patch.patch
            raw = patch.format_diff(format="raw")
            if callable(f1):
                self.assertIn(f1.__name__, raw)
            self.assertIn(f2.__name__, raw)
            rst = patch.format_diff(format="rst")
            self.assertIn("====", rst)

    @requires_transformers("4.55")
    def test_patch_diff(self):
        from onnx_diagnostic.torch_export_patches.patches.patch_transformers import (
            patched_eager_mask,
        )

        eager_mask = transformers.masking_utils.eager_mask
        self.assertEqual(eager_mask.__name__, "eager_mask")
        self.assertEqual(patched_eager_mask.__name__, "patched_eager_mask")
        diff = PatchInfo(eager_mask, patched_eager_mask).format_diff(format="rst")
        self.assertIn("+    # PATCHED:", diff)

    def test_involved_patches(self):
        data = get_untrained_model_with_inputs("arnir0/Tiny-LLM", verbose=0)
        model, inputs, ds = data["model"], data["inputs"], data["dynamic_shapes"]
        details = PatchDetails()
        with torch_export_patches(
            patch_transformers=True, patch_details=details, patch_torch=False
        ):
            ep = torch.export.export(
                model, (), kwargs=inputs, dynamic_shapes=use_dyn_not_str(ds)
            )
        patches = details.patches_involded_in_graph(ep.graph)
        self.assertNotEmpty(patches)
        report = details.make_report(patches, format="rst")
        self.assertIn("====", report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
