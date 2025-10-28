import os
import unittest
import torch
from onnx_diagnostic.ext_test_case import ExtTestCase, hide_stdout
from onnx_diagnostic.helpers.rt_helper import onnx_generate
from onnx_diagnostic.torch_models.hghub import get_untrained_model_with_inputs
from onnx_diagnostic.torch_export_patches import torch_export_patches
from onnx_diagnostic.export.api import to_onnx


class TestRtSession(ExtTestCase):
    def simple_generate_with_cache(
        self, model, input_ids: torch.Tensor, eos_token_id: int, max_new_tokens: int = 100
    ):
        # First call: prefill
        outputs = model(
            input_ids,
            attention_mask=torch.ones(
                input_ids.shape, dtype=input_ids.dtype, device=input_ids.device
            ),
            use_cache=True,
        )

        # Next calls: decode
        for _ in range(max_new_tokens):
            next_token_logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values
            next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            if next_token_id.item() == eos_token_id:
                break
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)
            outputs = model(
                next_token_id,
                use_cache=True,
                past_key_values=past_key_values,
                attention_mask=torch.ones(
                    input_ids.shape, dtype=input_ids.dtype, device=input_ids.device
                ),
            )
        return input_ids

    @hide_stdout()
    def test_onnx_generate(self):
        mid = "arnir0/Tiny-LLM"
        print("-- test_onnx_generate: get model")
        data = get_untrained_model_with_inputs(mid)
        model, inputs, ds = data["model"], data["inputs"], data["dynamic_shapes"]
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
        res = onnx_generate(model_name, input_ids[:1], 2, max_new_tokens=10)
        n_inputs = input_ids.shape[1]
        self.assertEqualArray(input_ids[:1], res[:, :n_inputs])
        self.assertEqual(res.dtype, torch.int64)
        self.assertEqual(res.shape, (1, 13))
        print("-- test_onnx_generate: done")
        # expected = model.generate(input_ids[:1], max_new_tokens=10)
        expected = self.simple_generate_with_cache(model, input_ids[:1], 2, max_new_tokens=10)
        self.assertEqualArray(input_ids[:1], expected[:, :n_inputs])
        print("******", res)
        print("******", expected)
        self.assertEqual(expected.dtype, torch.int64)
        self.assertEqual(expected.shape, (1, 13))
        self.assertEqualArray(expected, res)


if __name__ == "__main__":
    unittest.main(verbosity=2)
