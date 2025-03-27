import unittest
import transformers
from onnx_diagnostic.ext_test_case import (
    ExtTestCase,
    hide_stdout,
    requires_torch,
    requires_transformers,
)
from onnx_diagnostic.torch_models.hghub.model_inputs import (
    config_class_from_architecture,
    get_untrained_model_with_inputs,
)


class TestHuggingFaceHubModel(ExtTestCase):
    @requires_transformers("4.50")  # we limit to some versions of the CI
    @requires_torch("2.7")
    def test_config_class_from_architecture(self):
        config = config_class_from_architecture("LlamaForCausalLM")
        self.assertEqual(config, transformers.LlamaConfig)

    @hide_stdout()
    def test_get_untrained_model_with_inputs(self):
        mid = "arnir0/Tiny-LLM"
        data = get_untrained_model_with_inputs(mid, verbose=1)
        model, inputs = data["model"], data["inputs"]
        model(**inputs)
        self.assertEqual(data["size"], 1858125824)
        self.assertEqual(data["n_weights"], 464531456)


if __name__ == "__main__":
    unittest.main(verbosity=2)
