import inspect
import unittest
import torch
from onnx_diagnostic.ext_test_case import (
    ExtTestCase,
    ignore_warnings,
    hide_stdout,
    requires_transformers,
)
from onnx_diagnostic.torch_models.llms import get_tiny_llm
from onnx_diagnostic.torch_export_patches import bypass_export_some_errors

try:
    from experimental_experiment.torch_interpreter import to_onnx, ExportOptions
except ImportError:
    to_onnx = None


class TestTinyLlmOnnx(ExtTestCase):
    @ignore_warnings((UserWarning, DeprecationWarning, FutureWarning))
    @requires_transformers("4.50.9999")
    @hide_stdout()
    def test_onnx_export_tiny_llm_official(self):
        data = get_tiny_llm()
        model, inputs = data["model"], data["inputs"]
        self.assertEqual(
            {"attention_mask", "past_key_values", "input_ids", "position_ids"}, set(inputs)
        )
        ep = torch.onnx.export(
            model,
            (),
            kwargs=inputs,
            dynamic_shapes=data["dynamic_shapes"],
            dynamo=True,
            optimize=True,
        )
        self.assert_onnx_disc(
            inspect.currentframe().f_code.co_name, ep.model_proto, model, inputs, verbose=1
        )

    @unittest.skipIf(not to_onnx, reason="missing experimental dependency")
    @requires_transformers("4.50.9999")
    @ignore_warnings((UserWarning, DeprecationWarning, FutureWarning))
    @hide_stdout()
    def test_onnx_export_tiny_llm_xdbg(self):
        data = get_tiny_llm()
        model, inputs = data["model"], data["inputs"]
        self.assertEqual(
            {"attention_mask", "past_key_values", "input_ids", "position_ids"}, set(inputs)
        )
        onx = to_onnx(
            model, (), kwargs=inputs, dynamic_shapes=data["dynamic_shapes"], verbose=1
        )
        self.assert_onnx_disc(
            inspect.currentframe().f_code.co_name, onx, model, inputs, verbose=1
        )

    @ignore_warnings((UserWarning, DeprecationWarning, FutureWarning))
    @hide_stdout()
    def test_bypass_onnx_export_tiny_llm_official(self):
        data = get_tiny_llm()
        model, inputs = data["model"], data["inputs"]
        self.assertEqual(
            {"attention_mask", "past_key_values", "input_ids", "position_ids"}, set(inputs)
        )
        with bypass_export_some_errors(patch_transformers=True, verbose=1) as modificator:
            new_inputs = modificator(inputs)
            ep = torch.onnx.export(
                model,
                (),
                kwargs=new_inputs,
                dynamic_shapes=data["dynamic_shapes"],
                dynamo=True,
                optimize=True,
            )
        self.assert_onnx_disc(
            inspect.currentframe().f_code.co_name, ep.model_proto, model, inputs, verbose=1
        )

    @unittest.skipIf(not to_onnx, reason="missing experimental dependency")
    @ignore_warnings((UserWarning, DeprecationWarning, FutureWarning))
    @hide_stdout()
    def test_bypass_onnx_export_tiny_llm_xdbg(self):
        data = get_tiny_llm()
        model, inputs = data["model"], data["inputs"]
        self.assertEqual(
            {"attention_mask", "past_key_values", "input_ids", "position_ids"}, set(inputs)
        )
        with bypass_export_some_errors(patch_transformers=True, verbose=1) as modificator:
            new_inputs = modificator(inputs)
            onx = to_onnx(
                model,
                (),
                kwargs=new_inputs,
                dynamic_shapes=data["dynamic_shapes"],
                verbose=1,
                export_options=ExportOptions(strict=False),
            )
        self.assert_onnx_disc(
            inspect.currentframe().f_code.co_name, onx, model, inputs, verbose=1
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
