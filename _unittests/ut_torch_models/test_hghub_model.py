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
    def test_get_untrained_model_with_inputs_tiny_llm(self):
        mid = "arnir0/Tiny-LLM"
        data = get_untrained_model_with_inputs(mid, verbose=1)
        model, inputs = data["model"], data["inputs"]
        model(**inputs)
        self.assertEqual((1858125824, 464531456), (data["size"], data["n_weights"]))

    @hide_stdout()
    def test_get_untrained_model_with_inputs_tiny_xlm_roberta(self):
        mid = "hf-internal-testing/tiny-xlm-roberta"  # XLMRobertaConfig
        data = get_untrained_model_with_inputs(mid, verbose=1)
        model, inputs = data["model"], data["inputs"]
        model(**inputs)
        self.assertEqual((126190824, 31547706), (data["size"], data["n_weights"]))

    @hide_stdout()
    def test_get_untrained_model_with_inputs_tiny_gpt_neo(self):
        mid = "hf-internal-testing/tiny-random-GPTNeoXForCausalLM"
        data = get_untrained_model_with_inputs(mid, verbose=1)
        model, inputs = data["model"], data["inputs"]
        model(**inputs)
        self.assertEqual((4291141632, 1072785408), (data["size"], data["n_weights"]))

    @hide_stdout()
    def test_get_untrained_model_with_inputs_phi_2(self):
        mid = "microsoft/phi-2"
        data = get_untrained_model_with_inputs(mid, verbose=1)
        model, inputs = data["model"], data["inputs"]
        model(**inputs)
        # different expected value for different version of transformers
        self.assertIn(
            (data["size"], data["n_weights"]),
            [(1040293888, 260073472), (1040498688, 260124672)],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
