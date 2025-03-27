import unittest
import transformers
from onnx_diagnostic.ext_test_case import (
    ExtTestCase,
    hide_stdout,
    long_test,
    requires_torch,
    requires_transformers,
)
from onnx_diagnostic.torch_models.hghub.model_inputs import (
    config_class_from_architecture,
    get_untrained_model_with_inputs,
)
from onnx_diagnostic.torch_models.hghub.hub_data import load_models_testing


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
        self.assertEqual(
            set(data),
            {"model", "inputs", "dynamic_shapes", "configuration", "size", "n_weights"},
        )
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

    @hide_stdout()
    def test_get_untrained_model_with_inputs_beit(self):
        mid = "hf-internal-testing/tiny-random-BeitForImageClassification"
        data = get_untrained_model_with_inputs(mid, verbose=1)
        model, inputs = data["model"], data["inputs"]
        model(**inputs)
        # different expected value for different version of transformers
        self.assertIn((data["size"], data["n_weights"]), [(30732296, 7683074)])

    @hide_stdout()
    @long_test()
    def test_get_untrained_model_Ltesting_models(self):
        # UNHIDE=1 LONGTEST=1 python _unittests/ut_torch_models/test_hghub_model.py -k L -f
        for mid in load_models_testing():
            with self.subTest(mid=mid):
                data = get_untrained_model_with_inputs(mid, verbose=1)
                model, inputs = data["model"], data["inputs"]
                model(**inputs)
                # different expected value for different version of transformers
                if data["size"] > 2**30:
                    raise AssertionError(
                        f"Model is too big, size={data["size"] // 2**20} Mb,"
                        f"config is\n{data['configuration']}"
                    )
                self.assertLess(data["size"], 2**30)
                self.assertLess(data["n_weights"], 2**28)


if __name__ == "__main__":
    unittest.main(verbosity=2)
