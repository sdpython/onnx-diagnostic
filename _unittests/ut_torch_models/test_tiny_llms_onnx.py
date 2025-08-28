import copy
import inspect
import unittest
import torch
from onnx_diagnostic.ext_test_case import (
    ExtTestCase,
    ignore_warnings,
    hide_stdout,
    has_torch,
    requires_torch,
    requires_transformers,
)
from onnx_diagnostic.torch_models.llms import get_tiny_llm
from onnx_diagnostic.torch_export_patches import torch_export_patches

try:
    from experimental_experiment.torch_interpreter import to_onnx, ExportOptions
except ImportError:
    to_onnx = None


class TestTinyLlmOnnx(ExtTestCase):
    @ignore_warnings((UserWarning, DeprecationWarning, FutureWarning))
    @requires_transformers("4.52.9999")
    @hide_stdout()
    def test_onnx_export_tiny_llm_official(self):
        data = get_tiny_llm()
        model, inputs = data["model"], data["inputs"]
        self.assertEqual(
            {"attention_mask", "past_key_values", "input_ids", "position_ids"}, set(inputs)
        )
        with torch_export_patches(patch_transformers=True):
            ep = torch.onnx.export(
                model,
                (),
                kwargs=inputs,
                dynamic_shapes=data["dynamic_shapes"],
                dynamo=True,
                optimize=True,
            )
        # There are some discrepancies with torch==2.6
        if not has_torch("2.7"):
            raise unittest.SkipTest("discrepancies observed with torch<2.7")
        self.assert_onnx_disc(
            inspect.currentframe().f_code.co_name,
            ep.model_proto,
            model,
            inputs,
            verbose=1,
            use_ort=True,
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
        with torch_export_patches(patch_transformers=True):
            onx = to_onnx(
                model, (), kwargs=inputs, dynamic_shapes=data["dynamic_shapes"], verbose=1
            )
        self.assert_onnx_disc(
            inspect.currentframe().f_code.co_name, onx, model, inputs, verbose=1, use_ort=True
        )

    @ignore_warnings((UserWarning, DeprecationWarning, FutureWarning))
    @hide_stdout()
    @requires_torch("2.10.99")  # this test broke on CI but works locally
    def test_bypass_onnx_export_tiny_llm_official_nopositionids(self):
        data = get_tiny_llm()
        model, inputs, ds = data["model"], data["inputs"], data["dynamic_shapes"]
        del inputs["position_ids"]
        del ds["position_ids"]
        self.assertEqual({"attention_mask", "past_key_values", "input_ids"}, set(inputs))
        with torch_export_patches(patch_transformers=True, verbose=1) as modificator:
            new_inputs = modificator(copy.deepcopy(inputs))
            ep = torch.onnx.export(
                model,
                (),
                kwargs=new_inputs,
                dynamic_shapes=ds,
                dynamo=True,
                optimize=True,
            )
        self.assert_onnx_disc(
            inspect.currentframe().f_code.co_name,
            ep.model_proto,
            model,
            inputs,
            verbose=1,
            use_ort=True,
        )

    @ignore_warnings((UserWarning, DeprecationWarning, FutureWarning))
    @hide_stdout()
    def test_bypass_onnx_export_tiny_llm_official_full(self):
        try:
            from experimental_experiment.torch_interpreter import to_onnx
        except ImportError:
            to_onnx = None

        data = get_tiny_llm()
        model, inputs, ds = data["model"], data["inputs"], data["dynamic_shapes"]
        self.assertEqual(
            {"attention_mask", "past_key_values", "input_ids", "position_ids"}, set(inputs)
        )
        print("***", self.string_type(inputs, with_shape=True))
        print("---", type(model))
        with torch_export_patches(
            patch_transformers=True, verbose=1, stop_if_static=1
        ) as modificator:
            new_inputs = modificator(copy.deepcopy(inputs))
            if to_onnx:
                proto = to_onnx(model, (), kwargs=new_inputs, dynamic_shapes=ds)
            else:
                proto = torch.onnx.export(
                    model,
                    (),
                    kwargs=new_inputs,
                    dynamic_shapes=ds,
                    dynamo=True,
                    optimize=True,
                    report=True,
                    verify=False,
                ).model_proto
        # There are some discrepancies with torch==2.6
        if not has_torch("2.7"):
            raise unittest.SkipTest("discrepancies observed with torch<2.7")
        self.assert_onnx_disc(
            inspect.currentframe().f_code.co_name,
            proto,
            model,
            inputs,
            verbose=1,
            use_ort=True,
        )

    @unittest.skipIf(not to_onnx, reason="missing experimental dependency")
    @ignore_warnings((UserWarning, DeprecationWarning, FutureWarning))
    @hide_stdout()
    def test_bypass_onnx_export_tiny_llm_xdbg(self):
        data = get_tiny_llm()
        model, inputs, ds = data["model"], data["inputs"], data["dynamic_shapes"]
        self.assertEqual(
            {"attention_mask", "past_key_values", "input_ids", "position_ids"}, set(inputs)
        )
        with torch_export_patches(patch_transformers=True, verbose=2) as modificator:
            new_inputs = modificator(inputs)
            onx = to_onnx(
                model,
                (),
                kwargs=new_inputs,
                dynamic_shapes=ds,
                verbose=1,
                export_options=ExportOptions(strict=False),
            )
        self.assert_onnx_disc(
            inspect.currentframe().f_code.co_name, onx, model, inputs, verbose=1, use_ort=True
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
