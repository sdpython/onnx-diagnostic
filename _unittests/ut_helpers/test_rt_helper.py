import os
import unittest
import torch
from onnx_diagnostic.ext_test_case import (
    ExtTestCase,
    hide_stdout,
    requires_transformers,
    requires_torch,
)
from onnx_diagnostic.helpers import max_diff, flatten_object
from onnx_diagnostic.helpers.rt_helper import onnx_generate, make_empty_cache
from onnx_diagnostic.helpers.torch_helper import torch_deepcopy
from onnx_diagnostic.helpers.ort_session import InferenceSessionForTorch
from onnx_diagnostic.torch_models.hghub import get_untrained_model_with_inputs
from onnx_diagnostic.torch_export_patches import torch_export_patches
from onnx_diagnostic.export.api import to_onnx


class TestRtSession(ExtTestCase):
    def simple_generate_with_cache(
        self,
        model,
        input_ids: torch.Tensor,
        eos_token_id: int,
        session: InferenceSessionForTorch,
        max_new_tokens: int = 100,
    ):
        # First call: prefill
        attention_mask = torch.ones(
            input_ids.shape, dtype=input_ids.dtype, device=input_ids.device
        )
        feeds = {
            **dict(zip(session.input_names[:2], [input_ids, attention_mask])),
            **make_empty_cache(
                input_ids.shape[0],
                session.input_names[2:],
                session.input_shapes[2:],
                session.input_types[2:],
            ),
        }
        onnx_results = session.run(None, feeds)

        outputs = model(input_ids, use_cache=True, attention_mask=attention_mask)

        diff = max_diff(outputs, onnx_results)
        assert diff["abs"] <= 0.1, (
            f"Unexpected issue with {type(model)}\ndiff={diff}"
            f"\ninput_ids.shape={input_ids.shape}"
            f"\nexpected={self.string_type(outputs, with_shape=True, with_min_max=True)}"
            f"\n     got=\n"
            f"{self.string_type(onnx_results, with_shape=True, with_min_max=True)}"
        )

        # Next calls: decode
        for iteration in range(max_new_tokens):
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            if next_token_id.item() == eos_token_id:
                break
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)
            attention_mask = torch.ones(
                input_ids.shape, dtype=input_ids.dtype, device=input_ids.device
            )
            feeds = dict(
                zip(
                    session.input_names,
                    [
                        t.detach()
                        for t in torch_deepcopy(
                            flatten_object(
                                [next_token_id, attention_mask, outputs.past_key_values]
                            )
                        )
                    ],
                )
            )
            onnx_results = session.run(None, feeds)
            outputs = model(
                next_token_id,
                use_cache=True,
                past_key_values=outputs.past_key_values,
                attention_mask=attention_mask,
            )
            diff = max_diff(outputs, onnx_results)
            assert diff["abs"] <= 0.1, (
                f"Unexpected issue with {type(model)}, iteration={iteration}"
                f"\ndiff={diff}\ninput_ids.shape={input_ids.shape}"
                f"\nexpected={self.string_type(outputs, with_shape=True, with_min_max=True)}"
                f"\n     got=\n"
                f"{self.string_type(onnx_results, with_shape=True, with_min_max=True)}"
            )
        return input_ids

    @requires_transformers("4.55")
    @requires_torch("2.9")
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
                exporter="modelbuilder",
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
            expected = self.simple_generate_with_cache(
                model, input_ids[:1], 2, max_new_tokens=10, session=session
            )
            self.assertEqualArray(input_ids[:1], expected[:, :n_inputs])
            print("******", res)
            print("******", expected)
            self.assertEqual(expected.dtype, torch.int64)
            self.assertEqual(expected.shape, (1, 13))
            self.assertEqualArray(expected, res)


if __name__ == "__main__":
    unittest.main(verbosity=2)
