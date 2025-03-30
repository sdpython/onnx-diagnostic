import unittest
import torch
from onnx_diagnostic.ext_test_case import ExtTestCase, ignore_warnings, requires_transformers
from onnx_diagnostic.torch_models.llms import get_phi2
from onnx_diagnostic.helper import string_type
from onnx_diagnostic.torch_export_patches import bypass_export_some_errors


class TestLlmPhi(ExtTestCase):
    def test_get_phi2(self):
        data = get_phi2(num_hidden_layers=2)
        model, inputs = data["model"], data["inputs"]
        self.assertIn("DynamicCache", string_type(inputs))
        model(**inputs)

    @ignore_warnings(UserWarning)
    @requires_transformers("4.52")
    def test_export_phi2_1(self):
        data = get_phi2(num_hidden_layers=2)
        model, inputs, ds = data["model"], data["inputs"], data["dynamic_shapes"]
        self.assertEqual(
            {"attention_mask", "past_key_values", "input_ids", "position_ids"}, set(inputs)
        )
        ep = torch.export.export(model, (), kwargs=inputs, dynamic_shapes=ds)
        assert ep

    @ignore_warnings(UserWarning)
    def test_export_phi2_2_bypassed(self):
        data = get_phi2(num_hidden_layers=2)
        model, inputs, ds = data["model"], data["inputs"], data["dynamic_shapes"]
        self.assertEqual(
            {"attention_mask", "past_key_values", "input_ids", "position_ids"}, set(inputs)
        )
        with bypass_export_some_errors(patch_transformers=True) as modificator:
            inputs = modificator(inputs)
            ep = torch.export.export(model, (), kwargs=inputs, dynamic_shapes=ds, strict=False)
            assert ep


if __name__ == "__main__":
    unittest.main(verbosity=2)
