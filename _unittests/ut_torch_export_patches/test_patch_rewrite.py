import unittest
from onnx_diagnostic.ext_test_case import ExtTestCase
from onnx_diagnostic.torch_export_patches.patch_module_helper import code_needing_rewriting


class TestPatchRewrite(ExtTestCase):
    def test_code_needing_rewriting(self):
        res = code_needing_rewriting("BartModel")
        self.assertEqual(len(res), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
