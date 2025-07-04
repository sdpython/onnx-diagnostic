import pprint
import unittest
from onnx_diagnostic.ext_test_case import (
    ExtTestCase,
    hide_stdout,
    requires_torch,
    requires_transformers,
    ignore_errors,
)
from onnx_diagnostic.torch_models.hghub.model_inputs import get_untrained_model_with_inputs
from onnx_diagnostic.torch_models.hghub.hub_api import get_pretrained_config
from onnx_diagnostic.torch_models.hghub.hub_data import load_models_testing
from onnx_diagnostic.torch_export_patches import torch_export_patches


class TestHuggingFaceHubModel(ExtTestCase):
    @hide_stdout()
    def test_get_untrained_model_with_inputs_tiny_llm(self):
        mid = "arnir0/Tiny-LLM"
        data = get_untrained_model_with_inputs(mid, verbose=1, add_second_input=0)
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
                "task",
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
    @ignore_errors(OSError)
    def test_get_untrained_model_with_inputs_phi_2(self):
        mid = "microsoft/phi-2"
        data = get_untrained_model_with_inputs(mid, verbose=1)
        model, inputs = data["model"], data["inputs"]
        model(**inputs)
        # different expected value for different version of transformers
        self.assertIn(
            (data["size"], data["n_weights"]),
            [(453330944, 113332736), (453126144, 113281536)],
        )

    @hide_stdout()
    @ignore_errors(OSError)  # connectitivies issues
    def test_get_untrained_model_with_inputs_beit(self):
        mid = "hf-internal-testing/tiny-random-BeitForImageClassification"
        data = get_untrained_model_with_inputs(mid, verbose=1)
        model, inputs = data["model"], data["inputs"]
        model(**inputs)
        # different expected value for different version of transformers
        self.assertIn((data["size"], data["n_weights"]), [(111448, 27862), (56880, 14220)])

    @hide_stdout()
    @ignore_errors(OSError)
    def test_get_untrained_model_with_inputs_clip_vit(self):
        mid = "openai/clip-vit-base-patch16"
        data = get_untrained_model_with_inputs(mid, verbose=1)
        model, inputs = data["model"], data["inputs"]
        with torch_export_patches(patch_transformers=True):
            model(**inputs)
        # different expected value for different version of transformers
        self.assertIn((data["size"], data["n_weights"]), [(188872708, 47218177)])

    @hide_stdout()
    @requires_torch("2.7", "reduce test time")
    @requires_transformers("4.50", "reduce test time")
    @ignore_errors(OSError)  # connectivity issues may happen
    def test_get_untrained_model_Ltesting_models(self):
        # UNHIDE=1 python _unittests/ut_torch_models/test_hghub_model.py -k L -f
        def _diff(c1, c2):
            rows = [f"types {c1.__class__.__name__} <> {c2.__class__.__name__}"]
            for k, v in c1.__dict__.items():
                if isinstance(v, (str, dict, list, tuple, int, float)) and v != getattr(
                    c2, k, None
                ):
                    rows.append(f"{k} :: -- {v} ++ {getattr(c2, k, 'MISS')}")
            return "\n".join(rows)

        for mid in load_models_testing():
            with self.subTest(mid=mid):
                if mid in {
                    "hf-internal-testing/tiny-random-MaskFormerForInstanceSegmentation",
                    "hf-internal-testing/tiny-random-MoonshineForConditionalGeneration",
                    "fxmarty/pix2struct-tiny-random",
                    "hf-internal-testing/tiny-random-YolosModel",
                }:
                    print(f"-- not implemented yet for {mid!r}")
                    continue
                data = get_untrained_model_with_inputs(mid, verbose=1)
                model, inputs = data["model"], data["inputs"]
                if mid in {"sshleifer/tiny-marian-en-de"}:
                    print(f"-- not fully implemented yet for {mid!r}")
                    continue
                try:
                    model(**inputs)
                except Exception as e:
                    cf = get_pretrained_config(mid, use_only_preinstalled=True)
                    diff = _diff(cf, data["configuration"])
                    raise AssertionError(
                        f"Computation failed due to {e}.\n--- pretrained\n"
                        f"{pprint.pformat(cf)}\n--- modified\n{data['configuration']}\n"
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
