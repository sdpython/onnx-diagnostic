import unittest
import torch
from onnx_diagnostic.ext_test_case import (
    ExtTestCase,
    hide_stdout,
    requires_transformers,
    requires_torch,
    ignore_warnings,
)
from onnx_diagnostic.helpers.torch_helper import torch_deepcopy
from onnx_diagnostic.torch_models.hghub.model_inputs import get_untrained_model_with_inputs
from onnx_diagnostic.torch_export_patches import torch_export_patches
from onnx_diagnostic.torch_export_patches.patch_inputs import use_dyn_not_str
from onnx_diagnostic.export.shape_helper import make_fake_with_dynamic_dimensions
from onnx_diagnostic.helpers.rt_helper import onnx_generate, generate_and_validate


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
    @requires_torch("2.8.99")  # check_guards not supported
    def test_text_generation_tiny_llm(self):
        mid = "arnir0/Tiny-LLM"
        data = get_untrained_model_with_inputs(mid, verbose=1, add_second_input=True)
        self.assertEqual(data["task"], "text-generation")
        model, inputs, ds = data["model"], data["inputs"], data["dynamic_shapes"]
        inputs_copied = torch_deepcopy(inputs)
        expected = model(**torch_deepcopy(inputs))
        model(**data["inputs2"])
        fake = make_fake_with_dynamic_dimensions(inputs, dynamic_shapes=ds)[0]
        with torch_export_patches(patch_transformers=True, verbose=1, patch_torch=False):
            ep = torch.export.export(
                model, (), kwargs=fake, dynamic_shapes=use_dyn_not_str(ds), strict=False
            )
            # print(ep)
        rem = []
        for node in ep.graph.nodes:
            if "_assert" in str(node.target):
                rem.append(node)
        for node in rem:
            ep.graph.erase_node(node)
        ep.graph.lint()
        mod = ep.module(check_guards=False)
        got = mod(**inputs_copied)
        self.assertEqualAny(expected.past_key_values, got.past_key_values)
        self.assertEqualArray(expected.logits, got.logits)

    @hide_stdout()
    @requires_transformers("4.53")
    @requires_torch("2.8.99")  # check_guards not supported
    @ignore_warnings(FutureWarning)
    def test_text_generation_tiny_llm_prompt_validation(self):
        from experimental_experiment.torch_interpreter import to_onnx

        mid = "arnir0/Tiny-LLM"
        data = get_untrained_model_with_inputs(mid, verbose=1, add_second_input=True)
        prompt = data["inputs_prompt"]["input_ids"]
        self.assertEqual(data["task"], "text-generation")
        model, inputs, ds = data["model"], data["inputs"], data["dynamic_shapes"]
        with torch_export_patches(patch_transformers=True, verbose=1, patch_torch=False):
            onx = to_onnx(model, inputs, dynamic_shapes=ds)

        self.dump_onnx("test_text_generation_tiny_llm_prompt_validation.onnx", onx)
        onnx_sequence = onnx_generate(onx, prompt, max_new_tokens=3)
        torch_sequence = generate_and_validate(model, prompt, max_new_tokens=3)
        self.assertEqualArray(torch_sequence, onnx_sequence)


if __name__ == "__main__":
    unittest.main(verbosity=2)
