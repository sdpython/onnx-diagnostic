import unittest
import packaging.version as pv
import torch
from onnx_diagnostic.ext_test_case import (
    ExtTestCase,
    hide_stdout,
    requires_torch,
    requires_experimental,
    requires_transformers,
    requires_cuda,
)
from onnx_diagnostic.torch_models.validate import validate_model


class TestValidateModel(ExtTestCase):
    @requires_transformers("4.53")
    @requires_torch("2.7.99")
    @requires_experimental()
    @requires_cuda()
    @hide_stdout()
    def test_validate_tiny_llms_bfloat16(self):
        # python -m onnx_diagnostic validate -m microsoft/Phi-4-mini-reasoning
        #       --run -v 1 --export custom  -o dump_test --no-quiet --device cuda --patch
        summary, data = validate_model(
            "arnir0/Tiny-LLM",
            do_run=True,
            verbose=2,
            exporter="custom",
            do_same=True,
            patch=True,
            rewrite=True,
            stop_if_static=2 if pv.Version(torch.__version__) > pv.Version("2.6.1") else 0,
            dump_folder="dump_test/validate_tiny_llm",
            dtype="bfloat16",
            device="cuda",
            runtime="orteval",
        )
        self.assertLess(summary["disc_onnx_ort_run_abs"], 2e-2)
        self.assertIn("onnx_filename", data)

    @requires_transformers("4.53")
    @requires_torch("2.8.99")
    @requires_experimental()
    @hide_stdout()
    def test_validate_microsoft_phi4_reasoning(self):
        # python -m onnx_diagnostic validate -m microsoft/Phi-4-mini-reasoning
        #       --run -v 1 --export custom  -o dump_test --no-quiet --device cuda --patch
        summary, data = validate_model(
            "microsoft/Phi-4-mini-reasoning",
            do_run=True,
            verbose=2,
            exporter="custom",
            do_same=True,
            patch=True,
            rewrite=True,
            stop_if_static=2 if pv.Version(torch.__version__) > pv.Version("2.6.1") else 0,
            dump_folder="dump_test/validate_microsoft_phi4_reasoning",
        )
        self.assertLess(summary["disc_onnx_ort_run_abs"], 2e-5)
        self.assertIn("onnx_filename", data)

    @requires_transformers("4.53")
    @requires_torch("2.8.99")
    @requires_experimental()
    @hide_stdout()
    def test_validate_microsoft_phi3_mini_128k(self):
        # python -m onnx_diagnostic validate -m microsoft/Phi-3-mini-128k-instruct
        #       --run -v 1 --export custom  -o dump_test --no-quiet --device cuda --patch
        summary, data = validate_model(
            "microsoft/Phi-3-mini-128k-instruct",
            do_run=True,
            verbose=2,
            exporter="custom",
            do_same=True,
            patch=True,
            rewrite=True,
            stop_if_static=2 if pv.Version(torch.__version__) > pv.Version("2.6.1") else 0,
            dump_folder="dump_test/validate_microsoft_phi3_mini_128k",
        )
        self.assertLess(summary["disc_onnx_ort_run_abs"], 2e-5)
        self.assertIn("onnx_filename", data)


if __name__ == "__main__":
    unittest.main(verbosity=2)
