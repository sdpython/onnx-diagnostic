import unittest
from onnx_diagnostic.ext_test_case import (
    ExtTestCase,
    hide_stdout,
    ignore_warnings,
    requires_torch,
    requires_transformers,
    has_torch,
    has_transformers,
)
from onnx_diagnostic.torch_models.validate import validate_model

torch29_and_tr_main = not has_torch("2.9.9") and has_transformers("4.99999")


class TestValidateWholeModels3(ExtTestCase):
    @unittest.skipIf(torch29_and_tr_main, "combination not working")
    @requires_torch("2.7")
    @hide_stdout()
    @ignore_warnings(FutureWarning)
    @requires_transformers("4.51")
    def test_l_validate_model_modelbuilder(self):
        mid = "microsoft/phi-2"
        summary, data = validate_model(
            mid,
            do_run=True,
            verbose=10,
            exporter="modelbuilder",
            dump_folder="dump_test/validate_model_modelbuilder",
            patch=False,
        )
        self.assertIsInstance(summary, dict)
        self.assertIsInstance(data, dict)
        self.assertLess(summary["disc_onnx_ort_run_abs"], 3e-2)
        onnx_filename = data["onnx_filename"]
        self.assertExists(onnx_filename)
        self.clean_dump()

    @requires_torch("2.7")
    @hide_stdout()
    @ignore_warnings(FutureWarning)
    @requires_transformers("4.51")
    def test_m_validate_model_vit_model(self):
        mid = "ydshieh/tiny-random-ViTForImageClassification"
        summary, data = validate_model(
            mid,
            do_run=True,
            verbose=10,
            exporter="onnx-dynamo",
            dump_folder="dump_test/validate_model_vit_model",
            inputs2=True,
        )
        self.assertIsInstance(summary, dict)
        self.assertIsInstance(data, dict)
        self.assertLess(summary["disc_onnx_ort_run_abs"], 1e-3)
        self.assertLess(summary["disc_onnx_ort_run22_abs"], 1e-3)
        self.assertEqual("dict(pixel_values:A1s2x3x30x30)", summary["run_feeds_inputs"])
        self.assertEqual("dict(pixel_values:A1s3x3x31x31)", summary["run_feeds_inputs2"])
        self.assertEqual("#1[A1s2x2]", summary["run_output_inputs"])
        self.assertEqual("#1[A1s3x2]", summary["run_output_inputs2"])
        onnx_filename = data["onnx_filename"]
        self.assertExists(onnx_filename)
        self.clean_dump()


if __name__ == "__main__":
    unittest.main(verbosity=2)
