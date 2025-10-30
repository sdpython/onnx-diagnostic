import os
import unittest
import torch
from onnx_diagnostic.ext_test_case import (
    ExtTestCase,
    has_onnxruntime_genai,
    hide_stdout,
    requires_transformers,
    requires_torch,
)
from onnx_diagnostic.helpers.rt_helper import (
    onnx_generate,
    generate_and_validate,
    onnx_generate_with_genai,
)
from onnx_diagnostic.torch_models.hghub import get_untrained_model_with_inputs
from onnx_diagnostic.torch_export_patches import torch_export_patches
from onnx_diagnostic.export.api import to_onnx


class TestRtSession(ExtTestCase):
    @requires_transformers("4.55")
    @requires_torch("2.9")
    @hide_stdout()
    def test_onnx_generate(self):
        mid = "arnir0/Tiny-LLM"
        print("-- test_onnx_generate: get model")
        data = get_untrained_model_with_inputs(mid)
        model, inputs, ds = data["model"], data["inputs"], data["dynamic_shapes"]
        configuration = data["configuration"]
        del inputs["position_ids"]
        del ds["position_ids"]
        input_ids = inputs["input_ids"]
        print("----", input_ids.shape)
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
                exporter="custom",
            )

        print("-- test_onnx_generate: generate")
        res, session = onnx_generate(
            model_name, input_ids[:1], 2, max_new_tokens=10, return_session=True
        )
        n_inputs = input_ids.shape[1]
        self.assertEqualArray(input_ids[:1], res[:, :n_inputs])
        self.assertEqual(res.dtype, torch.int64)
        self.assertEqual(res.shape, (1, 13))
        print("-- test_onnx_generate: done")
        # expected = model.generate(input_ids[:1], max_new_tokens=10)
        expected, _ = generate_and_validate(
            model, input_ids[:1], 2, max_new_tokens=10, session=session
        )
        self.assertEqualArray(input_ids[:1], expected[:, :n_inputs])
        print("******", res)
        print("******", expected)
        self.assertEqual(expected.dtype, torch.int64)
        self.assertEqual(expected.shape, (1, 13))
        self.assertEqualArray(expected, res)

        if not has_onnxruntime_genai():
            raise unittest.SkipTest("onnxruntime_genai is missing")

        res, session = onnx_generate_with_genai(
            model_name,
            input_ids[:1],
            max_new_tokens=10,
            return_session=True,
            transformers_config=configuration,
        )
        self.assertNotEmpty(session)
        self.assertEqualArray(expected, res)


if __name__ == "__main__":
    unittest.main(verbosity=2)
