import unittest
from onnx_diagnostic.ext_test_case import ExtTestCase
from onnx_diagnostic.doc import reset_torch_transformers


class TestDocDoc(ExtTestCase):

    def test_reset(self):
        reset_torch_transformers(None, None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
