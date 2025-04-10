import pprint
import unittest
import torch
import transformers
from onnx_diagnostic.ext_test_case import (
    ExtTestCase,
    hide_stdout,
    requires_torch,
    requires_transformers,
    ignore_errors,
)
from onnx_diagnostic.torch_models.hghub.model_inputs import (
    config_class_from_architecture,
    get_untrained_model_with_inputs,
)
from onnx_diagnostic.torch_models.hghub.hub_api import get_pretrained_config
from onnx_diagnostic.torch_models.hghub.hub_data import load_models_testing
from onnx_diagnostic.torch_export_patches import bypass_export_some_errors


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
    def test_get_untrained_model_with_inputs_codellama(self):
        mid = "codellama/CodeLlama-7b-Python-hf"
        data = get_untrained_model_with_inputs(mid, verbose=1)
        model, inputs = data["model"], data["inputs"]
        model(**inputs)
        # different expected value for different version of transformers
        self.assertIn((data["size"], data["n_weights"]), [(410532864, 102633216)])

    @hide_stdout()
    def test_get_untrained_model_with_inputs_text2text_generation(self):
        mid = "sshleifer/tiny-marian-en-de"
        # mid = "Salesforce/codet5-small"
        data = get_untrained_model_with_inputs(mid, verbose=1)
        self.assertIn((data["size"], data["n_weights"]), [(473928, 118482)])
        model, inputs = data["model"], data["inputs"]
        raise unittest.SkipTest(f"not working for {mid!r}")
        model(**inputs)

    @hide_stdout()
    def test_get_untrained_model_with_inputs_automatic_speech_recognition(self):
        mid = "openai/whisper-tiny"
        data = get_untrained_model_with_inputs(mid, verbose=1)
        self.assertIn((data["size"], data["n_weights"]), [(132115968, 33028992)])
        model, inputs, ds = data["model"], data["inputs"], data["dynamic_shapes"]
        Dim = torch.export.Dim
        self.maxDiff = None
        self.assertIn("{0:Dim(batch),1:Dim(seq_length)}", self.string_type(ds))
        self.assertEqualAny(
            {
                "decoder_input_ids": {
                    0: Dim("batch", min=1, max=1024),
                    1: Dim("seq_length", min=1, max=4096),
                },
                "cache_position": {0: Dim("seq_length", min=1, max=4096)},
                "encoder_outputs": [{0: Dim("batch", min=1, max=1024)}],
                "past_key_values": [
                    [
                        [
                            {0: Dim("batch", min=1, max=1024)},
                            {0: Dim("batch", min=1, max=1024)},
                        ],
                        [
                            {0: Dim("batch", min=1, max=1024)},
                            {0: Dim("batch", min=1, max=1024)},
                        ],
                    ],
                    [
                        [
                            {0: Dim("batch", min=1, max=1024)},
                            {0: Dim("batch", min=1, max=1024)},
                        ],
                        [
                            {0: Dim("batch", min=1, max=1024)},
                            {0: Dim("batch", min=1, max=1024)},
                        ],
                    ],
                ],
            },
            ds,
        )
        model(**inputs)
        self.assertEqual(
            "#1[T1r3]",
            self.string_type(torch.utils._pytree.tree_flatten(inputs["encoder_outputs"])[0]),
        )
        with bypass_export_some_errors(patch_transformers=True, verbose=10):
            flat = torch.utils._pytree.tree_flatten(inputs["past_key_values"])[0]
            self.assertIsInstance(flat, list)
            self.assertIsInstance(flat[0], torch.Tensor)
            self.assertEqual(
                "#8[T1r4,T1r4,T1r4,T1r4,T1r4,T1r4,T1r4,T1r4]",
                self.string_type(flat),
            )
            torch.export.export(model, (), kwargs=inputs, dynamic_shapes=ds, strict=False)
        with bypass_export_some_errors(patch_transformers=True, verbose=10):
            flat = torch.utils._pytree.tree_flatten(inputs["past_key_values"])[0]
            self.assertIsInstance(flat, list)
            self.assertIsInstance(flat[0], torch.Tensor)
            self.assertEqual(
                "#8[T1r4,T1r4,T1r4,T1r4,T1r4,T1r4,T1r4,T1r4]",
                self.string_type(flat),
            )
            torch.export.export(model, (), kwargs=inputs, dynamic_shapes=ds, strict=False)

    @hide_stdout()
    def test_get_untrained_model_with_inputs_imagetext2text_generation(self):
        mid = "HuggingFaceM4/tiny-random-idefics"
        # mid = "Salesforce/codet5-small"
        data = get_untrained_model_with_inputs(mid, verbose=1)
        self.assertIn((data["size"], data["n_weights"]), [(12742888, 3185722)])
        model, inputs = data["model"], data["inputs"]
        model(**inputs)

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
                    "hf-internal-testing/tiny-random-BeitForImageClassification",
                    "hf-internal-testing/tiny-random-MaskFormerForInstanceSegmentation",
                    "hf-internal-testing/tiny-random-MoonshineForConditionalGeneration",
                    "fxmarty/pix2struct-tiny-random",
                    "hf-internal-testing/tiny-random-ViTMSNForImageClassification",
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
