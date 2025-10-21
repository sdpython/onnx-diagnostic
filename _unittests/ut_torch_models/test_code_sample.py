import unittest
import subprocess
import sys
from onnx_diagnostic.ext_test_case import (
    ExtTestCase,
    hide_stdout,
    requires_torch,
    requires_experimental,
    requires_transformers,
)
from onnx_diagnostic.torch_models.code_sample import code_sample


class TestCodeSample(ExtTestCase):
    @requires_transformers("4.53")
    @requires_torch("2.7.99")
    @requires_experimental()
    # @hide_stdout()
    def test_code_sample_tiny_llm(self):
        code = code_sample(
            "arnir0/Tiny-LLM",
            verbose=2,
            exporter="custom",
            patch=True,
            dump_folder="dump_test/validate_tiny_llm",
            dtype="float16",
            device="cuda",
        )
        filename = self.get_dump_file("test_code_sample_tiny_llm.py")
        with open(filename, "w") as f:
            f.write(code)
        cmds = [sys.executable, "-u", filename]
        p = subprocess.Popen(cmds, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        res = p.communicate()
        _out, err = res
        st = err.decode("ascii", errors="ignore")
        self.assertNotIn("Traceback", st)


if __name__ == "__main__":
    unittest.main(verbosity=2)
