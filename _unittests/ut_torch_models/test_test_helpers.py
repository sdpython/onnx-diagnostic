import copy
import unittest
import packaging.version as pv
import torch
from onnx_diagnostic.ext_test_case import ExtTestCase, hide_stdout, ignore_warnings
from onnx_diagnostic.torch_models.test_helper import (
    get_inputs_for_task,
    validate_model,
    filter_inputs,
)
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
            patch=True,
        )
        self.assertIsInstance(summary, dict)
        self.assertIsInstance(data, dict)

    @hide_stdout()
    @ignore_warnings(FutureWarning)
    def test_validate_model_onnx(self):
        mid = "arnir0/Tiny-LLM"
        summary, data = validate_model(
            mid,
            do_run=True,
            verbose=10,
            exporter="onnx-dynamo",
            dump_folder="dump_test_validate_model_onnx",
            patch=True,
            stop_if_static=2 if pv.Version(torch.__version__) > pv.Version("2.6") else 0,
        )
        self.assertIsInstance(summary, dict)
        self.assertIsInstance(data, dict)
        self.assertLess(summary["disc_onnx_ort_run_abs"], 1e-4)

    def test_filter_inputs(self):
        inputs, ds = {"a": 1, "b": 2}, {"a": 20, "b": 30}
        ni, nd = filter_inputs(inputs, dynamic_shapes=ds, drop_names=["a"])
        self.assertEqual((ni, nd), ({"b": 2}, {"b": 30}))

        inputs, ds = (1, 2), {"a": 20, "b": 30}
        ni, nd = filter_inputs(inputs, dynamic_shapes=ds, drop_names=["b"], model=["a", "b"])
        self.assertEqual((ni, nd), ((1, None), {"a": 20}))

        inputs, ds = (1, 2), (20, 30)
        ni, nd = filter_inputs(inputs, dynamic_shapes=ds, drop_names=["b"], model=["a", "b"])
        self.assertEqual((ni, nd), ((1, None), (20, None)))

        inputs, ds = ((1,), {"b": 4}), {"a": 20, "b": 30}
        ni, nd = filter_inputs(inputs, dynamic_shapes=ds, drop_names=["b"], model=["a", "b"])
        self.assertEqual((ni, nd), ((1,), {"a": 20}))

        inputs, ds = ((1,), {"b": 4}), {"a": 20, "b": 30}
        ni, nd = filter_inputs(inputs, dynamic_shapes=ds, drop_names=["a"], model=["a", "b"])
        self.assertEqual((ni, nd), (((None,), {"b": 4}), {"b": 30}))


if __name__ == "__main__":
    unittest.main(verbosity=2)
