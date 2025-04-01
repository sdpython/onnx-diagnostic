import copy
import unittest
from onnx_diagnostic.ext_test_case import ExtTestCase
from onnx_diagnostic.torch_models.test_helper import get_inputs_for_task
from onnx_diagnostic.torch_models.hghub.model_inputs import get_get_inputs_function_for_tasks


class TestTestHelper(ExtTestCase):
    def test_get_inputs_for_task(self):
        fcts = get_get_inputs_function_for_tasks()
        for task in self.subloop(sorted(fcts)):
            data = get_inputs_for_task(task)
            self.assertIsInstance(data, dict)
            self.assertIn("inputs", data)
            self.assertIn("dynamic_shapes", data)
            copy.deepcopy(data["inputs"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
