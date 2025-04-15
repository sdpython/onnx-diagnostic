import unittest
import transformers
from onnx_diagnostic.ext_test_case import (
    ExtTestCase,
    requires_torch,
    requires_transformers,
)
from onnx_diagnostic.helpers.config_helper import config_class_from_architecture


class TestConfigHelper(ExtTestCase):
    @requires_transformers("4.50")  # we limit to some versions of the CI
    @requires_torch("2.7")
    def test_config_class_from_architecture(self):
        config = config_class_from_architecture("LlamaForCausalLM")
        self.assertEqual(config, transformers.LlamaConfig)


if __name__ == "__main__":
    unittest.main(verbosity=2)
