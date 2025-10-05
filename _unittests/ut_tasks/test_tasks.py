import os
import unittest
import torch
from onnx_diagnostic.ext_test_case import (
    ExtTestCase,
    hide_stdout,
    has_transformers,
    requires_transformers,
)
from onnx_diagnostic.helpers.torch_helper import to_any, torch_deepcopy
from onnx_diagnostic.torch_models.hghub.model_inputs import get_untrained_model_with_inputs
from onnx_diagnostic.torch_export_patches import torch_export_patches
from onnx_diagnostic.torch_export_patches.patch_inputs import use_dyn_not_str


class TestTasks(ExtTestCase):
    def test_unittest_going(self):
        assert (
            os.environ.get("UNITTEST_GOING", "0") == "1"
        ), "UNITTEST_GOING=1 must be defined for these tests"

    @hide_stdout()
    def test_text2text_generation(self):
        mid = "sshleifer/tiny-marian-en-de"
        data = get_untrained_model_with_inputs(mid, verbose=1, add_second_input=True)
        self.assertEqual(data["task"], "text2text-generation")
        self.assertIn((data["size"], data["n_weights"]), [(473928, 118482)])
        model, inputs, ds = data["model"], data["inputs"], data["dynamic_shapes"]
        raise unittest.SkipTest(f"not working for {mid!r}")
        model(**inputs)
        model(**data["inputs2"])
        with torch_export_patches(patch_transformers=True, verbose=10):
            torch.export.export(
                model, (), kwargs=inputs, dynamic_shapes=use_dyn_not_str(ds), strict=False
            )

    @hide_stdout()
    def test_text_generation(self):
        mid = "arnir0/Tiny-LLM"
        data = get_untrained_model_with_inputs(mid, verbose=1, add_second_input=True)
        self.assertEqual(data["task"], "text-generation")
        self.assertIn((data["size"], data["n_weights"]), [(51955968, 12988992)])
        model, inputs, ds = data["model"], data["inputs"], data["dynamic_shapes"]
        model(**inputs)
        model(**data["inputs2"])
        with torch_export_patches(patch_transformers=True, verbose=10):
            torch.export.export(
                model, (), kwargs=inputs, dynamic_shapes=use_dyn_not_str(ds), strict=False
            )

    def test_text_generation_empty_cache(self):
        mid = "arnir0/Tiny-LLM"
        data = get_untrained_model_with_inputs(mid, add_second_input=True)
        model, inputs = data["model"], data["inputs"]
        self.assertIn("inputs_empty_cache", data)
        empty_inputs = torch_deepcopy(data["inputs_empty_cache"])
        model(**torch_deepcopy(empty_inputs))
        expected = model(**torch_deepcopy(inputs))
        self.assertEqual(
            {"attention_mask", "past_key_values", "input_ids", "position_ids"}, set(inputs)
        )
        with torch_export_patches(patch_transformers=True, verbose=1):
            ep = torch.export.export(
                model,
                (),
                kwargs=torch_deepcopy(inputs),
                dynamic_shapes=use_dyn_not_str(data["dynamic_shapes"]),
            )
            got = ep.module()(**torch_deepcopy(inputs))
            self.assertEqualArrayAny(expected, got)

    @hide_stdout()
    def test_automatic_speech_recognition_float32(self):
        mid = "openai/whisper-tiny"
        data = get_untrained_model_with_inputs(mid, verbose=1, add_second_input=True)
        self.assertEqual(data["task"], "automatic-speech-recognition")
        self.assertIn((data["size"], data["n_weights"]), [(132115968, 33028992)])
        model, inputs, ds = data["model"], data["inputs"], data["dynamic_shapes"]
        model(**data["inputs"])
        model(**data["inputs2"])
        self.maxDiff = None
        self.assertIn("{0:DYN(batch),1:DYN(seq_length)}", self.string_type(ds))
        self.assertEqualAny(
            {
                "decoder_input_ids": {0: "batch", 1: "seq_length"},
                "cache_position": {0: "seq_length"},
                "encoder_outputs": [{0: "batch"}],
                "past_key_values": [
                    [[{0: "batch"}, {0: "batch"}], [{0: "batch"}, {0: "batch"}]],
                    [[{0: "batch"}, {0: "batch"}], [{0: "batch"}, {0: "batch"}]],
                ],
            },
            ds,
        )
        model(**inputs)
        self.assertEqual(
            "#1[T1r3]",
            self.string_type(torch.utils._pytree.tree_flatten(inputs["encoder_outputs"])[0]),
        )
        with torch_export_patches(patch_transformers=True, verbose=10):
            flat = torch.utils._pytree.tree_flatten(inputs["past_key_values"])[0]
            self.assertIsInstance(flat, list)
            self.assertIsInstance(flat[0], torch.Tensor)
            self.assertEqual(
                "#8[T1r4,T1r4,T1r4,T1r4,T1r4,T1r4,T1r4,T1r4]",
                self.string_type(flat),
            )
            torch.export.export(
                model, (), kwargs=inputs, dynamic_shapes=use_dyn_not_str(ds), strict=False
            )
        with torch_export_patches(patch_transformers=True, verbose=10):
            flat = torch.utils._pytree.tree_flatten(inputs["past_key_values"])[0]
            self.assertIsInstance(flat, list)
            self.assertIsInstance(flat[0], torch.Tensor)
            self.assertEqual(
                "#8[T1r4,T1r4,T1r4,T1r4,T1r4,T1r4,T1r4,T1r4]",
                self.string_type(flat),
            )
            torch.export.export(
                model, (), kwargs=inputs, dynamic_shapes=use_dyn_not_str(ds), strict=False
            )

    @hide_stdout()
    def test_automatic_speech_recognition_float16(self):
        mid = "openai/whisper-tiny"
        data = get_untrained_model_with_inputs(mid, verbose=1, add_second_input=True)
        self.assertEqual(data["task"], "automatic-speech-recognition")
        self.assertIn((data["size"], data["n_weights"]), [(132115968, 33028992)])
        self.assertIn("encoder_outputs:BaseModelOutput", self.string_type(data["inputs"]))
        data["inputs"] = to_any(data["inputs"], torch.float16)
        self.assertIn("encoder_outputs:BaseModelOutput", self.string_type(data["inputs"]))
        data["inputs2"] = to_any(data["inputs2"], torch.float16)
        model, inputs, ds = data["model"], data["inputs"], data["dynamic_shapes"]
        model = to_any(model, torch.float16)
        model(**data["inputs2"])
        self.maxDiff = None
        self.assertIn("{0:DYN(batch),1:DYN(seq_length)}", self.string_type(ds))
        self.assertEqualAny(
            {
                "decoder_input_ids": {0: "batch", 1: "seq_length"},
                "cache_position": {0: "seq_length"},
                "encoder_outputs": [{0: "batch"}],
                "past_key_values": [
                    [[{0: "batch"}, {0: "batch"}], [{0: "batch"}, {0: "batch"}]],
                    [[{0: "batch"}, {0: "batch"}], [{0: "batch"}, {0: "batch"}]],
                ],
            },
            ds,
        )
        self.assertEqual(
            "#1[T10r3]",
            self.string_type(torch.utils._pytree.tree_flatten(inputs["encoder_outputs"])[0]),
        )
        with torch_export_patches(patch_transformers=True, verbose=10):
            model(**inputs)
            flat = torch.utils._pytree.tree_flatten(inputs["past_key_values"])[0]
            self.assertIsInstance(flat, list)
            self.assertIsInstance(flat[0], torch.Tensor)
            self.assertEqual(
                "#8[T10r4,T10r4,T10r4,T10r4,T10r4,T10r4,T10r4,T10r4]",
                self.string_type(flat),
            )
            torch.export.export(
                model, (), kwargs=inputs, dynamic_shapes=use_dyn_not_str(ds), strict=False
            )
        with torch_export_patches(patch_transformers=True, verbose=10):
            flat = torch.utils._pytree.tree_flatten(inputs["past_key_values"])[0]
            self.assertIsInstance(flat, list)
            self.assertIsInstance(flat[0], torch.Tensor)
            self.assertEqual(
                "#8[T10r4,T10r4,T10r4,T10r4,T10r4,T10r4,T10r4,T10r4]",
                self.string_type(flat),
            )
            torch.export.export(
                model, (), kwargs=inputs, dynamic_shapes=use_dyn_not_str(ds), strict=False
            )

    @hide_stdout()
    def test_fill_mask(self):
        mid = "google-bert/bert-base-multilingual-cased"
        data = get_untrained_model_with_inputs(mid, verbose=1, add_second_input=True)
        self.assertEqual(data["task"], "fill-mask")
        self.assertIn((data["size"], data["n_weights"]), [(428383212, 107095803)])
        model, inputs, ds = data["model"], data["inputs"], data["dynamic_shapes"]
        model(**inputs)
        model(**data["inputs2"])
        with torch_export_patches(patch_transformers=True, verbose=10):
            torch.export.export(
                model, (), kwargs=inputs, dynamic_shapes=use_dyn_not_str(ds), strict=False
            )

    @hide_stdout()
    @requires_transformers("4.53.99")
    def test_feature_extraction_bart_base(self):
        mid = "facebook/bart-base"
        data = get_untrained_model_with_inputs(mid, verbose=1, add_second_input=True)
        self.assertEqual(data["task"], "feature-extraction")
        self.assertIn((data["size"], data["n_weights"]), [(557681664, 139420416)])
        model, inputs, ds = data["model"], data["inputs"], data["dynamic_shapes"]
        model(**torch_deepcopy(inputs))
        model(**data["inputs2"])
        with torch_export_patches(patch_transformers=True, verbose=10):
            torch.export.export(
                model, (), kwargs=inputs, dynamic_shapes=use_dyn_not_str(ds), strict=False
            )

    @hide_stdout()
    def test_feature_extraction_tiny_bart(self):
        mid = "hf-tiny-model-private/tiny-random-PLBartForConditionalGeneration"
        data = get_untrained_model_with_inputs(mid, verbose=1, add_second_input=True)
        self.assertEqual(data["task"], "text2text-generation")
        self.assertIn((data["size"], data["n_weights"]), [(3243392, 810848)])
        model, inputs, ds = data["model"], data["inputs"], data["dynamic_shapes"]
        model(**inputs)
        model(**data["inputs2"])
        with torch_export_patches(patch_transformers=True, verbose=10):
            torch.export.export(
                model, (), kwargs=inputs, dynamic_shapes=use_dyn_not_str(ds), strict=False
            )

    @requires_transformers("4.51.999")
    @hide_stdout()
    def test_summarization(self):
        mid = "facebook/bart-large-cnn"
        data = get_untrained_model_with_inputs(mid, verbose=1, add_second_input=True)
        self.assertEqual(data["task"], "summarization")
        self.assertIn((data["size"], data["n_weights"]), [(1625161728, 406290432)])
        model, inputs, _ds = data["model"], data["inputs"], data["dynamic_shapes"]
        model(**inputs)
        model(**data["inputs2"])
        # with torch_export_patches(patch_transformers=True, verbose=10):
        #    torch.export.export(
        #        model, (), kwargs=inputs, dynamic_shapes=use_dyn_not_str(ds), strict=False
        #    )

    @hide_stdout()
    def test_text_classification(self):
        mid = "Intel/bert-base-uncased-mrpc"
        data = get_untrained_model_with_inputs(mid, verbose=1, add_second_input=True)
        self.assertEqual(data["task"], "text-classification")
        self.assertIn((data["size"], data["n_weights"]), [(154420232, 38605058)])
        model, inputs, ds = data["model"], data["inputs"], data["dynamic_shapes"]
        model(**inputs)
        model(**data["inputs2"])
        with torch_export_patches(patch_transformers=True, verbose=10):
            torch.export.export(
                model, (), kwargs=inputs, dynamic_shapes=use_dyn_not_str(ds), strict=False
            )

    @hide_stdout()
    def test_sentence_similary(self):
        mid = "sentence-transformers/all-MiniLM-L6-v1"
        data = get_untrained_model_with_inputs(mid, verbose=1, add_second_input=True)
        self.assertEqual(data["task"], "sentence-similarity")
        self.assertIn((data["size"], data["n_weights"]), [(62461440, 15615360)])
        model, inputs, ds = data["model"], data["inputs"], data["dynamic_shapes"]
        model(**inputs)
        model(**data["inputs2"])
        with torch_export_patches(patch_transformers=True, verbose=10):
            torch.export.export(
                model, (), kwargs=inputs, dynamic_shapes=use_dyn_not_str(ds), strict=False
            )

    @hide_stdout()
    def test_falcon_mamba_dev(self):
        mid = "tiiuae/falcon-mamba-tiny-dev"
        data = get_untrained_model_with_inputs(mid, verbose=1, add_second_input=True)
        self.assertEqual(data["task"], "text-generation")
        model, inputs, ds = data["model"], data["inputs"], data["dynamic_shapes"]
        model(**inputs)
        model(**data["inputs2"])
        self.assertIn((data["size"], data["n_weights"]), [(274958336, 68739584)])
        if not has_transformers("4.57.99"):
            raise unittest.SkipTest("The model has control flow.")
        with torch_export_patches(patch_transformers=True, verbose=10, stop_if_static=1):
            torch.export.export(
                model, (), kwargs=inputs, dynamic_shapes=use_dyn_not_str(ds), strict=False
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
