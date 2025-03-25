import inspect
import unittest
import torch
from onnx_diagnostic.ext_test_case import ExtTestCase, ignore_warnings, hide_stdout
from onnx_diagnostic.torch_models.llms import get_tiny_llm
from onnx_diagnostic.torch_export_patches import bypass_export_some_errors

try:
    from experimental_experiment.torch_interpreter import to_onnx
except ImportError:
    to_onnx = None


class TestLlmsOnnx(ExtTestCase):
    @ignore_warnings((UserWarning, DeprecationWarning, FutureWarning))
    @hide_stdout()
    def test_onnx__export_tiny_llm(self):
        data = get_tiny_llm()
        model, inputs = data["model"], data["inputs"]
        self.assertEqual({"attention_mask", "past_key_values", "input_ids"}, set(inputs))
        with bypass_export_some_errors(
            patch_transformers=True, replace_dynamic_cache=True, verbose=1
        ) as modificator:
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
    def test_onnx_export_tiny_llm_cdbg(self):
        data = get_tiny_llm()
        model, inputs = data["model"], data["inputs"]
        self.assertEqual({"attention_mask", "past_key_values", "input_ids"}, set(inputs))
        with bypass_export_some_errors(
            patch_transformers=True, replace_dynamic_cache=True, verbose=1
        ) as modificator:
            new_inputs = modificator(inputs)
            onx = to_onnx(
                model, (), kwargs=new_inputs, dynamic_shapes=data["dynamic_shapes"], verbose=1
            )
        self.assert_onnx_disc(
            inspect.currentframe().f_code.co_name, onx, model, inputs, verbose=1
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
