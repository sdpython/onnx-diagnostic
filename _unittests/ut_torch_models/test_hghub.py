import unittest
import pandas
from onnx_diagnostic.ext_test_case import ExtTestCase, never_test
from onnx_diagnostic.torch_models.hghub.hub_api import enumerate_model_list


class TestHuggingFaceHub(ExtTestCase):
    def test_enumerate_model_list(self):
        models = list(enumerate_model_list(2, verbose=1, dump="test_enumerate_model_list.csv"))
        self.assertEqual(len(models), 2)
        df = pandas.read_csv("test_enumerate_model_list.csv")
        self.assertEqual(df.shape, (2, 11))

    @never_test()
    def test_hf_all_models(self):
        list(enumerate_model_list(-1, verbose=1, dump="test_hf_all_models.csv"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
