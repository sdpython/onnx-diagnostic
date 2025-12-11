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

    def test_torch_load(self):
        import torch

        filename = self.get_dump_file("test_torch_load.test.pt")

        def save():
            from onnx_diagnostic.helpers.cache_helper import make_dynamic_cache

            n_layers = 2
            bsize, nheads, slen, dim = 2, 4, 3, 7
            cache = make_dynamic_cache(
                [
                    (
                        torch.randn(bsize + 1, nheads, slen + 1, dim + 1),
                        torch.randn(bsize + 1, nheads, slen + 1, dim + 1),
                    )
                    for i in range(n_layers)
                ]
            )
            torch.save(dict(cache=cache), filename)

        save()
        restored = torch.load(filename, weights_only=False)
        self.assertIsInstance(restored, dict)
        self.assertEqual(restored["cache"].__class__.__name__, "DynamicCache")


if __name__ == "__main__":
    unittest.main(verbosity=2)
