import unittest
import torch
from onnx_diagnostic.ext_test_case import ExtTestCase, ignore_warnings, requires_transformers
from onnx_diagnostic.torch_models.llms import get_tiny_llm
from onnx_diagnostic.helpers import string_type
from onnx_diagnostic.torch_export_patches import bypass_export_some_errors


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
        self.assertEqual(
            {"attention_mask", "past_key_values", "input_ids", "position_ids"}, set(inputs)
        )
        ep = torch.export.export(
            model, (), kwargs=inputs, dynamic_shapes=data["dynamic_shapes"]
        )
        assert ep

    @ignore_warnings(UserWarning)
    def test_export_tiny_llm_2_bypassed(self):
        data = get_tiny_llm()
        model, inputs = data["model"], data["inputs"]
        self.assertEqual(
            {"attention_mask", "past_key_values", "input_ids", "position_ids"}, set(inputs)
        )

        with bypass_export_some_errors(
            patch_torch=False, patch_transformers=True, catch_constraints=False
        ) as modificator:
            inputs = modificator(inputs)

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
            assert ep


if __name__ == "__main__":
    unittest.main(verbosity=2)
