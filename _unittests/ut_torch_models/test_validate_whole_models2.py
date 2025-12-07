import unittest
import onnx
import torch
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


class TestValidateWholeModels2(ExtTestCase):
    @requires_torch("2.9")
    @hide_stdout()
    @ignore_warnings(FutureWarning)
    @requires_transformers("4.55")
    @unittest.skipIf(torch.__version__.startswith("2.9.0"), "no left space space on device?")
    def test_o_validate_phi35_4k_mini_instruct(self):
        mid = "microsoft/Phi-3-mini-4k-instruct"
        summary, data = validate_model(
            mid,
            do_run=True,
            verbose=10,
            exporter="custom",
            dump_folder="dump_test/validate_phi35_mini_instruct",
            inputs2=True,
            patch=True,
            rewrite=True,
            model_options={"rope_scaling": {"rope_type": "dynamic", "factor": 10.0}},
        )
        self.assertIsInstance(summary, dict)
        self.assertIsInstance(data, dict)
        onnx_filename = data["onnx_filename"]
        onx = onnx.load(onnx_filename)
        op_types = set(n.op_type for n in onx.graph.node)
        self.assertIn("If", op_types)
        self.clean_dump()


if __name__ == "__main__":
    unittest.main(verbosity=2)
