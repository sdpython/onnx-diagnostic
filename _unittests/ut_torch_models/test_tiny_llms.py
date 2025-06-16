import copy
import unittest
import torch
from onnx_diagnostic.ext_test_case import ExtTestCase, ignore_warnings
from onnx_diagnostic.torch_models.llms import get_tiny_llm
from onnx_diagnostic.helpers import string_type
from onnx_diagnostic.torch_export_patches import torch_export_patches
from onnx_diagnostic.torch_export_patches.patch_inputs import use_dyn_not_str


class TestTinyLlm(ExtTestCase):
    def test_tiny_llm_run_dynamic(self):
        data = get_tiny_llm()
        model, inputs = data["model"], data["inputs"]
        self.assertIn("DynamicCache", string_type(inputs))
        model(**inputs)

    @ignore_warnings(UserWarning)
    def test_tiny_llm_export_dynamic(self):
        data = get_tiny_llm()
        model, inputs = data["model"], data["inputs"]
        expected = model(**copy.deepcopy(inputs))
        self.assertEqual(
            {"attention_mask", "past_key_values", "input_ids", "position_ids"}, set(inputs)
        )
        with torch_export_patches(patch_transformers=True):
            ep = torch.export.export(
                model,
                (),
                kwargs=copy.deepcopy(inputs),
                dynamic_shapes=use_dyn_not_str(data["dynamic_shapes"]),
            )
            got = ep.module()(**inputs)
            self.assertEqualArrayAny(expected, got)

    def test_tiny_llm_run_static(self):
        data = get_tiny_llm(use_static_cache=True)
        model, inputs = data["model"], data["inputs"]
        self.assertIn("StaticCache", string_type(inputs))
        model(**inputs)

    @ignore_warnings(UserWarning)
    def test_tiny_llm_export_static(self):
        data = get_tiny_llm(use_static_cache=True)
        model, inputs = data["model"], data["inputs"]
        expected = model(**copy.deepcopy(inputs))
        self.assertEqual(
            {"attention_mask", "past_key_values", "input_ids", "cache_position"}, set(inputs)
        )
        with torch_export_patches(patch_transformers=True, stop_if_static=1):
            ep = torch.export.export(
                model,
                (),
                kwargs=copy.deepcopy(inputs),
                dynamic_shapes=use_dyn_not_str(data["dynamic_shapes"]),
            )
            got = ep.module()(**inputs)
            self.assertEqualArrayAny(expected, got)


if __name__ == "__main__":
    unittest.main(verbosity=2)
