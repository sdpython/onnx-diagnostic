import unittest
import pandas
from onnx_diagnostic.ext_test_case import (
    ExtTestCase,
    never_test,
    requires_torch,
    requires_transformers,
)
from onnx_diagnostic.torch_models.hghub.hub_api import (
    enumerate_model_list,
    task_from_id,
    task_from_arch,
)


class TestHuggingFaceHub(ExtTestCase):

    @requires_transformers("4.50")  # we limit to some versions of the CI
    @requires_torch("2.7")
    def test_enumerate_model_list(self):
        models = list(
            enumerate_model_list(
                2,
                verbose=1,
                dump="test_enumerate_model_list.csv",
                filter="text-generation",
                library="transformers",
            )
        )
        self.assertEqual(len(models), 2)
        df = pandas.read_csv("test_enumerate_model_list.csv")
        self.assertEqual(df.shape, (2, 11))
        tasks = [task_from_id(c) for c in df.id]
        self.assertEqual(["text-generation", "text-generation"], tasks)

    @requires_transformers("4.50")
    @requires_torch("2.7")
    def test_task_from_id(self):
        task = task_from_id("arnir0/Tiny-LLM", True)
        self.assertEqual("text-generation", task)

    def test_task_from_arch(self):
        task = task_from_arch("LlamaForCausalLM")
        self.assertEqual("text-generation", task)

    @never_test()
    def test_hf_all_models(self):
        list(enumerate_model_list(-1, verbose=1, dump="test_hf_all_models.csv"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
