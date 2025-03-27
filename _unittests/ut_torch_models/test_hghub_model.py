import pprint
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
from onnx_diagnostic.torch_models.hghub.hub_api import get_pretrained_config
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
            {
                "model",
                "inputs",
                "dynamic_shapes",
                "configuration",
                "size",
                "n_weights",
                "input_kwargs",
                "model_kwargs",
            },
        )
        model, inputs = data["model"], data["inputs"]
        model(**inputs)
        self.assertEqual((51955968, 12988992), (data["size"], data["n_weights"]))

    @hide_stdout()
    def test_get_untrained_model_with_inputs_tiny_xlm_roberta(self):
        mid = "hf-internal-testing/tiny-xlm-roberta"  # XLMRobertaConfig
        data = get_untrained_model_with_inputs(mid, verbose=1)
        model, inputs = data["model"], data["inputs"]
        model(**inputs)
        self.assertEqual((8642088, 2160522), (data["size"], data["n_weights"]))

    @hide_stdout()
    def test_get_untrained_model_with_inputs_tiny_gpt_neo(self):
        mid = "hf-internal-testing/tiny-random-GPTNeoXForCausalLM"
        data = get_untrained_model_with_inputs(mid, verbose=1)
        model, inputs = data["model"], data["inputs"]
        model(**inputs)
        self.assertEqual((316712, 79178), (data["size"], data["n_weights"]))

    @hide_stdout()
    def test_get_untrained_model_with_inputs_phi_2(self):
        mid = "microsoft/phi-2"
        data = get_untrained_model_with_inputs(mid, verbose=1)
        model, inputs = data["model"], data["inputs"]
        model(**inputs)
        # different expected value for different version of transformers
        self.assertIn(
            (data["size"], data["n_weights"]),
            [(453330944, 113332736)],
        )

    @hide_stdout()
    def test_get_untrained_model_with_inputs_beit(self):
        mid = "hf-internal-testing/tiny-random-BeitForImageClassification"
        data = get_untrained_model_with_inputs(mid, verbose=1)
        model, inputs = data["model"], data["inputs"]
        model(**inputs)
        # different expected value for different version of transformers
        self.assertIn((data["size"], data["n_weights"]), [(111448, 27862)])

    @hide_stdout()
    def test_get_untrained_model_with_inputs_codellama(self):
        mid = "codellama/CodeLlama-7b-Python-hf"
        data = get_untrained_model_with_inputs(mid, verbose=1)
        model, inputs = data["model"], data["inputs"]
        model(**inputs)
        # different expected value for different version of transformers
        self.assertIn((data["size"], data["n_weights"]), [(410532864, 102633216)])

    @hide_stdout()
    @long_test()
    def test_get_untrained_model_Ltesting_models(self):
        def _diff(c1, c2):
            rows = [f"types {c1.__class__.__name__} <> {c2.__class__.__name__}"]
            for k, v in c1.__dict__.items():
                if isinstance(v, (str, dict, list, tuple, int, float)) and v != getattr(
                    c2, k, None
                ):
                    rows.append(f"{k} :: -- {v} ++ {getattr(c2, k, 'MISS')}")
            return "\n".join(rows)

        # UNHIDE=1 LONGTEST=1 python _unittests/ut_torch_models/test_hghub_model.py -k L -f
        for mid in load_models_testing():
            with self.subTest(mid=mid):
                data = get_untrained_model_with_inputs(mid, verbose=1)
                model, inputs = data["model"], data["inputs"]
                try:
                    model(**inputs)
                except Exception as e:
                    diff = _diff(get_pretrained_config(mid), data["configuration"])
                    raise AssertionError(
                        f"Computation failed due to {e}.\n--- pretrained\n"
                        f"{pprint.pformat(get_pretrained_config(mid))}\n"
                        f"--- modified\n{data['configuration']}\n"
                        f"--- diff\n{diff}"
                    ) from e
                # different expected value for different version of transformers
                if data["size"] > 2**30:
                    raise AssertionError(
                        f"Model is too big, size={data['size'] // 2**20} Mb,"
                        f"config is\n{data['configuration']}"
                    )
                self.assertLess(data["size"], 2**30)
                self.assertLess(data["n_weights"], 2**28)


if __name__ == "__main__":
    unittest.main(verbosity=2)
