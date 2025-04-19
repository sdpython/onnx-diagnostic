import unittest
import torch
from onnx_diagnostic.ext_test_case import ExtTestCase, hide_stdout, has_transformers, has_torch
from onnx_diagnostic.torch_models.hghub.model_inputs import get_untrained_model_with_inputs
from onnx_diagnostic.torch_export_patches import bypass_export_some_errors
from onnx_diagnostic.torch_export_patches.patch_inputs import use_dyn_not_str


class TestTasks(ExtTestCase):
    @hide_stdout()
    def test_text2text_generation(self):
        mid = "sshleifer/tiny-marian-en-de"
        data = get_untrained_model_with_inputs(mid, verbose=1)
        self.assertIn((data["size"], data["n_weights"]), [(473928, 118482)])
        model, inputs, ds = data["model"], data["inputs"], data["dynamic_shapes"]
        raise unittest.SkipTest(f"not working for {mid!r}")
        model(**inputs)
        with bypass_export_some_errors(patch_transformers=True, verbose=10):
            torch.export.export(
                model, (), kwargs=inputs, dynamic_shapes=use_dyn_not_str(ds), strict=False
            )

    @hide_stdout()
    def test_automatic_speech_recognition(self):
        mid = "openai/whisper-tiny"
        data = get_untrained_model_with_inputs(mid, verbose=1)
        self.assertIn((data["size"], data["n_weights"]), [(132115968, 33028992)])
        model, inputs, ds = data["model"], data["inputs"], data["dynamic_shapes"]
        Dim = torch.export.Dim
        self.maxDiff = None
        self.assertIn("{0:Dim(batch),1:DYN(seq_length)}", self.string_type(ds))
        self.assertEqualAny(
            {
                "decoder_input_ids": {
                    0: Dim("batch", min=1, max=1024),
                    1: "seq_length",
                },
                "cache_position": {0: "seq_length"},
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
            torch.export.export(
                model, (), kwargs=inputs, dynamic_shapes=use_dyn_not_str(ds), strict=False
            )
        with bypass_export_some_errors(patch_transformers=True, verbose=10):
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
    def test_imagetext2text_generation(self):
        mid = "HuggingFaceM4/tiny-random-idefics"
        data = get_untrained_model_with_inputs(mid, verbose=1)
        self.assertIn((data["size"], data["n_weights"]), [(12742888, 3185722)])
        model, inputs, ds = data["model"], data["inputs"], data["dynamic_shapes"]
        model(**inputs)
        if not has_torch("2.10"):
            raise unittest.SkipTest("sym_max does not work with dynamic dimension")
        with bypass_export_some_errors(patch_transformers=True, verbose=10):
            torch.export.export(
                model, (), kwargs=inputs, dynamic_shapes=use_dyn_not_str(ds), strict=False
            )

    @hide_stdout()
    def test_fill_mask(self):
        mid = "google-bert/bert-base-multilingual-cased"
        data = get_untrained_model_with_inputs(mid, verbose=1)
        self.assertIn((data["size"], data["n_weights"]), [(428383212, 107095803)])
        model, inputs, ds = data["model"], data["inputs"], data["dynamic_shapes"]
        model(**inputs)
        with bypass_export_some_errors(patch_transformers=True, verbose=10):
            torch.export.export(
                model, (), kwargs=inputs, dynamic_shapes=use_dyn_not_str(ds), strict=False
            )

    @hide_stdout()
    def test_text_classification(self):
        mid = "Intel/bert-base-uncased-mrpc"
        data = get_untrained_model_with_inputs(mid, verbose=1)
        self.assertIn((data["size"], data["n_weights"]), [(154420232, 38605058)])
        model, inputs, ds = data["model"], data["inputs"], data["dynamic_shapes"]
        model(**inputs)
        with bypass_export_some_errors(patch_transformers=True, verbose=10):
            torch.export.export(
                model, (), kwargs=inputs, dynamic_shapes=use_dyn_not_str(ds), strict=False
            )

    @hide_stdout()
    def test_sentence_similary(self):
        mid = "sentence-transformers/all-MiniLM-L6-v1"
        data = get_untrained_model_with_inputs(mid, verbose=1)
        self.assertIn((data["size"], data["n_weights"]), [(62461440, 15615360)])
        model, inputs, ds = data["model"], data["inputs"], data["dynamic_shapes"]
        model(**inputs)
        with bypass_export_some_errors(patch_transformers=True, verbose=10):
            torch.export.export(
                model, (), kwargs=inputs, dynamic_shapes=use_dyn_not_str(ds), strict=False
            )

    @hide_stdout()
    def test_falcon_mamba_dev(self):
        mid = "tiiuae/falcon-mamba-tiny-dev"
        data = get_untrained_model_with_inputs(mid, verbose=1)
        model, inputs, ds = data["model"], data["inputs"], data["dynamic_shapes"]
        model(**inputs)
        self.assertIn((data["size"], data["n_weights"]), [(138640384, 34660096)])
        if not has_transformers("4.55"):
            raise unittest.SkipTest("The model has control flow.")
        with bypass_export_some_errors(patch_transformers=True, verbose=10, stop_if_static=1):
            torch.export.export(
                model, (), kwargs=inputs, dynamic_shapes=use_dyn_not_str(ds), strict=False
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
