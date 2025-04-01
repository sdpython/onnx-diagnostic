import copy
import unittest
from onnx_diagnostic.ext_test_case import ExtTestCase, hide_stdout
from onnx_diagnostic.torch_models.test_helper import get_inputs_for_task, validate_model
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

    @hide_stdout()
    def test_validate_model(self):
        mid = "arnir0/Tiny-LLM"
        summary, data = validate_model(mid, do_run=True, verbose=2)
        self.assertIsInstance(summary, dict)
        self.assertIsInstance(data, dict)
        validate_model(mid, do_run=True, verbose=2, quiet=True)

    @hide_stdout()
    def test_validate_model_dtype(self):
        mid = "arnir0/Tiny-LLM"
        summary, data = validate_model(
            mid, do_run=True, verbose=2, dtype="float32", device="cpu"
        )
        self.assertIsInstance(summary, dict)
        self.assertIsInstance(data, dict)
        validate_model(mid, do_run=True, verbose=2, quiet=True)

    @hide_stdout()
    def test_validate_model_export(self):
        mid = "arnir0/Tiny-LLM"
        summary, data = validate_model(
            mid,
            do_run=True,
            verbose=10,
            exporter="export-nostrict",
            dump_folder="dump_test_validate_model_export",
        )
        self.assertIsInstance(summary, dict)
        self.assertIsInstance(data, dict)


if __name__ == "__main__":
    unittest.main(verbosity=2)
