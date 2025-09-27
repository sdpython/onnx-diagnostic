import unittest
from onnx_diagnostic.ext_test_case import ExtTestCase
from onnx_diagnostic.tasks.data import get_data


class TestTasks(ExtTestCase):
    def test_get_data(self):
        name = "dummies_imagetext2text_generation_gemma3.onnx"
        data = get_data(name)
        print(data)


if __name__ == "__main__":
    unittest.main(verbosity=2)
