import unittest
from onnx_diagnostic.ext_test_case import ExtTestCase
from onnx_diagnostic.helpers.args_helper import get_parsed_args, check_cuda_availability


class TestHelpers(ExtTestCase):
    def test_check_cuda_availability(self):
        check_cuda_availability()

    def test_args(self):
        try:
            args = get_parsed_args(
                "plot_custom_backend_llama",
                config=(
                    "medium",
                    "large or medium depending, large means closer to the real model",
                ),
                num_hidden_layers=(1, "number of hidden layers"),
                with_mask=(0, "tries with a mask as a secondary input"),
                optim=("", "Optimization to apply, empty string for all"),
                description="doc",
                new_args=["--config", "m"],
            )
        except SystemExit as e:
            raise AssertionError(f"SystemExist caught: {e}")
        self.assertEqual(args.config, "m")
        self.assertEqual(args.num_hidden_layers, 1)
        self.assertEqual(args.with_mask, 0)
        self.assertEqual(args.optim, "")

    def test_args_expose(self):
        try:
            args = get_parsed_args(
                "plot_custom_backend_llama",
                config=(
                    "medium",
                    "large or medium depending, large means closer to the real model",
                ),
                num_hidden_layers=(1, "number of hidden layers"),
                with_mask=(0, "tries with a mask as a secondary input"),
                optim=("", "Optimization to apply, empty string for all"),
                description="doc",
                new_args=["--config", "m"],
                expose="repeat,warmup",
            )
        except SystemExit as e:
            raise AssertionError(f"SystemExist caught: {e}")
        self.assertEqual(args.config, "m")
        self.assertEqual(args.num_hidden_layers, 1)
        self.assertEqual(args.with_mask, 0)
        self.assertEqual(args.optim, "")
        self.assertEqual(args.repeat, 10)
        self.assertEqual(args.warmup, 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
