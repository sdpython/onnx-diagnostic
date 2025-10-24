import unittest
import torch
from onnx_diagnostic.ext_test_case import (
    ExtTestCase,
    hide_stdout,
    requires_transformers,
    requires_torch,
)
from onnx_diagnostic.helpers.torch_helper import torch_deepcopy
from onnx_diagnostic.torch_models.hghub.model_inputs import get_untrained_model_with_inputs
from onnx_diagnostic.torch_export_patches import torch_export_patches
from onnx_diagnostic.torch_export_patches.patch_inputs import use_dyn_not_str
from onnx_diagnostic.export.shape_helper import make_fake_with_dynamic_dimensions


class TestTasksTextGeneration(ExtTestCase):
    @hide_stdout()
    @requires_transformers("4.53")
    @requires_torch("2.7.99")
    def test_text_generation_gemma3_for_causallm(self):
        mid = "hf-internal-testing/tiny-random-Gemma3ForCausalLM"
        data = get_untrained_model_with_inputs(mid, verbose=1, add_second_input=True)
        self.assertEqual(data["task"], "text-generation")
        model, inputs, ds = data["model"], data["inputs"], data["dynamic_shapes"]
        expected = model(**torch_deepcopy(inputs))
        model(**data["inputs2"])
        with torch_export_patches(patch_transformers=True, verbose=10, patch_torch=False):
            ep = torch.export.export(
                model, (), kwargs=inputs, dynamic_shapes=use_dyn_not_str(ds), strict=False
            )
            self.assertEqualAny(expected, ep.module()(**inputs))

    @hide_stdout()
    @requires_transformers("4.53")
    @requires_torch("2.7.99")
    def test_text_generation_phi_3_mini_128k_instruct(self):
        mid = "microsoft/Phi-3-mini-128k-instruct"
        data = get_untrained_model_with_inputs(mid, verbose=1, add_second_input=True)
        self.assertEqual(data["task"], "text-generation")
        model, inputs, ds = data["model"], data["inputs"], data["dynamic_shapes"]
        expected = model(**torch_deepcopy(inputs))
        model(**data["inputs2"])
        with torch_export_patches(patch_transformers=True, verbose=10, patch_torch=False):
            ep = torch.export.export(
                model, (), kwargs=inputs, dynamic_shapes=use_dyn_not_str(ds), strict=False
            )
            self.assertEqualAny(expected, ep.module()(**inputs))

    @hide_stdout()
    @requires_transformers("4.53")
    @requires_torch("2.7.99")
    def test_text_generation_tiny_llm(self):
        mid = "arnir0/Tiny-LLM"
        data = get_untrained_model_with_inputs(mid, verbose=1, add_second_input=True)
        self.assertEqual(data["task"], "text-generation")
        model, inputs, ds = data["model"], data["inputs"], data["dynamic_shapes"]
        inputs_copied = torch_deepcopy(inputs)
        expected = model(**torch_deepcopy(inputs))
        model(**data["inputs2"])
        fake = make_fake_with_dynamic_dimensions(inputs, dynamic_shapes=ds)[0]
        with torch_export_patches(patch_transformers=True, verbose=10, patch_torch=False):
            ep = torch.export.export(
                model, (), kwargs=fake, dynamic_shapes=use_dyn_not_str(ds), strict=False
            )
            # print(ep)
        got = ep.module()(**inputs_copied)
        self.assertEqualAny(expected.past_key_values, got.past_key_values)
        self.assertEqualArray(expected.logits, got.logits)


if __name__ == "__main__":
    unittest.main(verbosity=2)
