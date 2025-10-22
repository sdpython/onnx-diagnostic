import unittest
import subprocess
import sys
import torch
from onnx_diagnostic.ext_test_case import (
    ExtTestCase,
    hide_stdout,
    requires_torch,
    requires_experimental,
    requires_transformers,
)
from onnx_diagnostic.helpers.cache_helper import make_dynamic_cache
from onnx_diagnostic.torch_models.code_sample import code_sample, make_code_for_inputs


class TestCodeSample(ExtTestCase):
    @requires_transformers("4.53")
    @requires_torch("2.7.99")
    @requires_experimental()
    @hide_stdout()
    def test_code_sample_tiny_llm_custom(self):
        code = code_sample(
            "arnir0/Tiny-LLM",
            verbose=2,
            exporter="custom",
            patch=True,
            dump_folder="dump_test/validate_tiny_llm_custom",
            dtype="float16",
            device="cpu",
            optimization="default",
        )
        filename = self.get_dump_file("test_code_sample_tiny_llm_custom.py")
        with open(filename, "w") as f:
            f.write(code)
        cmds = [sys.executable, "-u", filename]
        p = subprocess.Popen(cmds, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        res = p.communicate()
        _out, err = res
        st = err.decode("ascii", errors="ignore")
        self.assertNotIn("Traceback", st)

    @requires_transformers("4.53")
    @requires_torch("2.7.99")
    @requires_experimental()
    @hide_stdout()
    def test_code_sample_tiny_llm_dynamo(self):
        code = code_sample(
            "arnir0/Tiny-LLM",
            verbose=2,
            exporter="onnx-dynamo",
            patch=True,
            dump_folder="dump_test/validate_tiny_llm_dynamo",
            dtype="float16",
            device="cpu",
            optimization="ir",
        )
        filename = self.get_dump_file("test_code_sample_tiny_llm_dynamo.py")
        with open(filename, "w") as f:
            f.write(code)
        cmds = [sys.executable, "-u", filename]
        p = subprocess.Popen(cmds, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        res = p.communicate()
        _out, err = res
        st = err.decode("ascii", errors="ignore")
        self.assertNotIn("Traceback", st)

    def test_make_code_for_inputs(self):
        values = [
            ("dict(a=True)", dict(a=True)),
            ("dict(a=1)", dict(a=1)),
            (
                "dict(a=torch.randint(3, size=(2,), dtype=torch.int64))",
                dict(a=torch.tensor([2, 3], dtype=torch.int64)),
            ),
            (
                "dict(a=torch.rand((2,), dtype=torch.float16))",
                dict(a=torch.tensor([2, 3], dtype=torch.float16)),
            ),
        ]
        for res, inputs in values:
            self.assertEqual(res, make_code_for_inputs(inputs))

        res = make_code_for_inputs(
            dict(
                cc=make_dynamic_cache(
                    [(torch.randn(2, 2, 2, 2), torch.randn(2, 2, 2, 2)) for i in range(2)]
                )
            )
        )
        self.assertEqual(
            "dict(cc=make_dynamic_cache([(torch.rand((2, 2, 2, 2), "
            "dtype=torch.float32),torch.rand((2, 2, 2, 2), dtype=torch.float32)), "
            "(torch.rand((2, 2, 2, 2), dtype=torch.float32),"
            "torch.rand((2, 2, 2, 2), dtype=torch.float32))]))",
            res,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
