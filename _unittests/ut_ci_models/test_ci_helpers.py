import unittest
from argparse import ArgumentParser
from onnx_diagnostic.ext_test_case import ExtTestCase
from onnx_diagnostic.ci_models.ci_helpers import (
    get_parser,
    get_versions,
    get_torch_dtype_from_command_line_args,
    simplify_model_id_for_a_filename,
)


class TestCiHelpers(ExtTestCase):
    def test_get_versions(self):
        self.assertIsInstance(get_versions(), dict)

    def test_get_parser(self):
        self.assertIsInstance(get_parser("any"), ArgumentParser)

    def test_get_torch_dtype_from_command_line_args(self):
        self.assertEqual(
            get_torch_dtype_from_command_line_args("float16"),
            get_torch_dtype_from_command_line_args("fp16"),
        )

    def test_simplify_model_id_for_a_filename(self):
        self.assertEqual(simplify_model_id_for_a_filename("m/n"), "m.n")


if __name__ == "__main__":
    unittest.main(verbosity=2)
