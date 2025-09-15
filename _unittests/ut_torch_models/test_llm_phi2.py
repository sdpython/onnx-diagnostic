import unittest
import torch
from onnx_diagnostic.ext_test_case import (
    ExtTestCase,
    ignore_warnings,
    requires_transformers,
    requires_torch,
)
from onnx_diagnostic.torch_models.llms import get_phi2
from onnx_diagnostic.helpers import string_type
from onnx_diagnostic.torch_export_patches import torch_export_patches
from onnx_diagnostic.torch_export_patches.patch_inputs import use_dyn_not_str


class TestLlmPhi(ExtTestCase):
    def test_get_phi2(self):
        data = get_phi2(num_hidden_layers=2, batch_size=2)
        model, inputs = data["model"], data["inputs"]
        self.assertIn("DynamicCache", string_type(inputs))
        model(**inputs)

    @ignore_warnings(UserWarning)
    @requires_transformers("4.54")
    @requires_torch("2.9.99")
    def test_export_phi2_1_batch_size_1(self):
        # exporting vmap does not work
        data = get_phi2(num_hidden_layers=2, batch_size=1)
        model, inputs, ds = data["model"], data["inputs"], data["dynamic_shapes"]
        self.assertEqual(inputs["input_ids"].shape[0], 1)
        self.assertEqual(
            {"attention_mask", "past_key_values", "input_ids", "position_ids"}, set(inputs)
        )
        with torch.fx.experimental._config.patch(
            backed_size_oblivious=True
        ), torch_export_patches(patch_transformers=True):
            ep = torch.export.export(
                model, (), kwargs=inputs, dynamic_shapes=use_dyn_not_str(ds)
            )
        assert ep

    @ignore_warnings(UserWarning)
    @requires_transformers("4.54")
    @requires_torch("2.9.99")
    def test_export_phi2_1_batch_size_2(self):
        # exporting vmap does not work
        data = get_phi2(num_hidden_layers=2, batch_size=2)
        model, inputs, ds = data["model"], data["inputs"], data["dynamic_shapes"]
        self.assertEqual(inputs["input_ids"].shape[0], 2)
        self.assertEqual(
            {"attention_mask", "past_key_values", "input_ids", "position_ids"}, set(inputs)
        )
        with torch_export_patches(patch_transformers=True):
            ep = torch.export.export(
                model, (), kwargs=inputs, dynamic_shapes=use_dyn_not_str(ds)
            )
        assert ep


if __name__ == "__main__":
    unittest.main(verbosity=2)
