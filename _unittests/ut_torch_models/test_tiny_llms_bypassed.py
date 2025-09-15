import copy
import unittest
import torch
from onnx_diagnostic.ext_test_case import ExtTestCase, ignore_warnings, hide_stdout
from onnx_diagnostic.torch_models.llms import get_tiny_llm
from onnx_diagnostic.torch_models.llms import get_phi2
from onnx_diagnostic.helpers import string_type
from onnx_diagnostic.helpers.torch_helper import torch_deepcopy
from onnx_diagnostic.torch_export_patches import torch_export_patches
from onnx_diagnostic.torch_export_patches.patch_inputs import use_dyn_not_str


class TestTinyLlmBypassed(ExtTestCase):
    @ignore_warnings(UserWarning)
    @hide_stdout()
    def test_export_tiny_llm_2_bypassed(self):
        data = get_tiny_llm()
        model, inputs = data["model"], data["inputs"]
        expected = model(**copy.deepcopy(inputs))
        self.assertEqual(
            {"attention_mask", "past_key_values", "input_ids", "position_ids"}, set(inputs)
        )

        with torch_export_patches(
            patch_torch=False, patch_transformers=True, catch_constraints=False, verbose=10
        ) as modificator:

            inputs = modificator(copy.deepcopy(inputs))

            def debug():
                print("***", string_type(inputs, with_shape=True))
                print("***", data["dynamic_shapes"])
                import torch.export._draft_export

                _ep, report = torch.export._draft_export.draft_export(
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
                model,
                (),
                kwargs=inputs,
                dynamic_shapes=use_dyn_not_str(data["dynamic_shapes"]),
                strict=False,
            )
            got = ep.module()(**inputs)
            self.assertEqualArrayAny(expected, got, atol=1e-5)

    @ignore_warnings(UserWarning)
    def test_export_phi2_2_bypassed(self):
        data = get_phi2(num_hidden_layers=2)
        model, inputs, ds = data["model"], data["inputs"], data["dynamic_shapes"]
        self.assertEqual(
            {"attention_mask", "past_key_values", "input_ids", "position_ids"}, set(inputs)
        )
        model(**torch_deepcopy(inputs))
        ds = use_dyn_not_str(ds)
        with torch_export_patches(patch_transformers=True, stop_if_static=1) as modificator:
            inputs = modificator(inputs)
            ep = torch.export.export(model, (), kwargs=inputs, dynamic_shapes=ds, strict=False)
            assert ep
        with torch_export_patches(patch_transformers=True) as modificator:
            inputs = modificator(inputs)
            ep = torch.export.export(model, (), kwargs=inputs, dynamic_shapes=ds, strict=False)
            assert ep


if __name__ == "__main__":
    unittest.main(verbosity=2)
