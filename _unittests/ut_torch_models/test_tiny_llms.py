import copy
import unittest
import torch
from transformers.cache_utils import DynamicCache
from onnx_diagnostic.ext_test_case import ExtTestCase, ignore_warnings, requires_transformers
from onnx_diagnostic.torch_models.llms import get_tiny_llm
from onnx_diagnostic.helpers import string_type
from onnx_diagnostic.torch_export_patches import bypass_export_some_errors
from onnx_diagnostic.torch_export_patches.patches.patch_transformers import (
    patched_DynamicCache,
)


class TestTinyLlm(ExtTestCase):
    def test_get_tiny_llm(self):
        data = get_tiny_llm()
        model, inputs = data["model"], data["inputs"]
        self.assertIn("DynamicCache", string_type(inputs))
        model(**inputs)

    @ignore_warnings(UserWarning)
    @requires_transformers("4.52")
    def test_export_tiny_llm_1(self):
        data = get_tiny_llm()
        model, inputs = data["model"], data["inputs"]
        expected = model(**copy.deepcopy(inputs))
        self.assertEqual(
            {"attention_mask", "past_key_values", "input_ids", "position_ids"}, set(inputs)
        )
        ep = torch.export.export(
            model, (), kwargs=copy.deepcopy(inputs), dynamic_shapes=data["dynamic_shapes"]
        )
        got = ep.module()(**inputs)
        self.assertEqualArrayAny(expected, got)

    @ignore_warnings(UserWarning)
    def test_export_tiny_llm_2_bypassed(self):
        data = get_tiny_llm()
        model, inputs = data["model"], data["inputs"]
        expected = model(**copy.deepcopy(inputs))
        self.assertEqual(
            {"attention_mask", "past_key_values", "input_ids", "position_ids"}, set(inputs)
        )

        with bypass_export_some_errors(
            patch_torch=False, patch_transformers=True, catch_constraints=False, verbose=10
        ) as modificator:

            for k in patched_DynamicCache._PATCHES_:
                self.assertEqual(getattr(patched_DynamicCache, k), getattr(DynamicCache, k))

            inputs = modificator(copy.deepcopy(inputs))

            def debug():
                print("***", string_type(inputs, with_shape=True))
                print("***", data["dynamic_shapes"])
                import torch.export._draft_export

                ep, report = torch.export._draft_export.draft_export(
                    model,
                    (),
                    kwargs=inputs,
                    dynamic_shapes=data["dynamic_shapes"],
                    strict=False,
                )
                print(report)

            if self._debug():
                debug()

            ep = torch.export.export(
                model, (), kwargs=inputs, dynamic_shapes=data["dynamic_shapes"], strict=False
            )
            got = ep.module()(**inputs)
            self.assertEqualArrayAny(expected, got)


if __name__ == "__main__":
    unittest.main(verbosity=2)
