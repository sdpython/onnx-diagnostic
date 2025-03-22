import unittest
from onnx_diagnostic.ext_test_case import ExtTestCase
from experimental_experiment.args import get_parsed_args


class TestHelpers(ExtTestCase):
    def test_args(self):
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
        )
        self.assertEqual(args.config, "medium")
        self.assertEqual(args.num_hidden_layers, 1)
        self.assertEqual(args.with_mask, 0)
        self.assertEqual(args.optim, "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
