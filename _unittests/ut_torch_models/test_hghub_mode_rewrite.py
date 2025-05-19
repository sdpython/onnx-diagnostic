import unittest
from onnx_diagnostic.ext_test_case import ExtTestCase
from onnx_diagnostic.torch_models.hghub.hub_data import code_needing_rewriting


class TestHuggingFaceHubModelRewrite(ExtTestCase):

    def test_code_needing_rewriting(self):
        self.assertEqual(1, len(code_needing_rewriting("BartForConditionalGeneration")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
