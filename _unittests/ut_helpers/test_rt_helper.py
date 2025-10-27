import os
import unittest
import torch
from onnx_diagnostic.ext_test_case import ExtTestCase, hide_stdout
from onnx_diagnostic.helpers.rt_helper import onnx_generate
from onnx_diagnostic.torch_models.hghub import get_untrained_model_with_inputs
from onnx_diagnostic.torch_export_patches import torch_export_patches


class TestRtSession(ExtTestCase):
    @hide_stdout()
    def test_onnx_generate(self):
        from experimental_experiment.torch_interpreter import to_onnx

        mid = "arnir0/Tiny-LLM"
        print("-- test_onnx_generate: get model")
        data = get_untrained_model_with_inputs(mid)
        model, inputs, ds = data["model"], data["inputs"], data["dynamic_shapes"]
        del inputs["position_ids"]
        del ds["position_ids"]
        input_ids = inputs["input_ids"]
        folder = self.get_dump_folder("test_onnx_generate")
        model_name = os.path.join(folder, "model.onnx")
        print("-- test_onnx_generate: export model")
        with torch_export_patches(patch_transformers=True, patch_torch=False):
            to_onnx(
                model,
                (),
                kwargs=inputs,
                dynamic_shapes=ds,
                filename=model_name,
            )

        print("-- test_onnx_generate: generate")
        res = onnx_generate(model_name, input_ids[:1], 2, max_new_tokens=10)
        self.assertEqual(res.dtype, torch.int64)
        self.assertEqual(res.shape, (1, 13))
        print("-- test_onnx_generate: done")


if __name__ == "__main__":
    unittest.main(verbosity=2)
