import unittest
from onnx_diagnostic.ext_test_case import (
    ExtTestCase,
    hide_stdout,
    requires_transformers,
    requires_torch,
)
from onnx_diagnostic.torch_models.validate import validate_model


class TestTasksMaskGeneration(ExtTestCase):
    @hide_stdout()
    @requires_transformers("4.53")
    @requires_torch("2.7.99")
    def test_text_generation(self):
        mid = "microsoft/phi-2"
        summary, data = validate_model(
            mid,
            do_run=True,
            verbose=10,
            exporter="onnx-dynamo",
            dump_folder="dump_test/microsoft_phi-2",
            inputs2=True,
            patch=True,
        )
        self.assertIsInstance(summary, dict)
        # multi-turn conversation
        self.assertLess(summary["disc_onnx_ort_run_abs"], 3e-2)
        # prompt processing
        self.assertLess(summary["disc_onnx_ort_run2_abs"], 3e-2)
        # token generation
        self.assertLess(summary["disc_onnx_ort_run3_abs"], 3e-2)
        self.assertIsInstance(data, dict)
        onnx_filename = data["onnx_filename"]
        self.assertExists(onnx_filename)


if __name__ == "__main__":
    unittest.main(verbosity=2)
