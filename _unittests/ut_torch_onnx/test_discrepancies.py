import os
import unittest
import numpy as np
import onnx
from onnx_diagnostic.ext_test_case import ExtTestCase, ignore_warnings
from onnx_diagnostic.reference import OnnxruntimeEvaluator


class TestDiscrepancies(ExtTestCase):
    @ignore_warnings(DeprecationWarning)
    def test_attention_opset15_in_a_loop(self):
        import torch
        from onnx_diagnostic.torch_export_patches.patches._patch_transformers_attention import (  # noqa: E501
            patched_sdpa_attention_forward,
        )

        def qwen_sdpa_attention(
            query_states: torch.Tensor,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            cu_seqlens: torch.Tensor,
            scaling: float = 0,
            num_heads: int = 16,
        ) -> torch.Tensor:
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            splits = [
                torch.split(tensor, lengths.tolist(), dim=2)
                for tensor in (query_states, key_states, value_states)
            ]

            attn_outputs = [
                patched_sdpa_attention_forward(
                    None,
                    q,
                    k,
                    v,
                    attention_mask=None,
                    scaling=scaling,
                    dropout=0.0,
                    is_causal=False,
                )[0]
                for q, k, v in zip(*splits)
            ]
            attn_output = torch.cat(attn_outputs, dim=1)
            return attn_output

        model = onnx.load(
            os.path.join(os.path.dirname(__file__), "data", "attention_loopa24.onnx")
        )
        sess = self.check_ort(model)

        feeds = dict(
            c_lifted_tensor_0=np.array([0], dtype=np.int64),
            cat_2=np.array(
                [
                    0,
                    64,
                    128,
                    192,
                    256,
                    304,
                    368,
                    432,
                    496,
                    560,
                    608,
                    672,
                    736,
                    800,
                    864,
                    912,
                    976,
                    1040,
                    1104,
                    1168,
                    1216,
                    1232,
                    1248,
                    1264,
                    1280,
                    1292,
                ],
                dtype=np.int64,
            ),
            unsqueeze_4=np.random.randn(1, 16, 1292, 80).astype(np.float32),
            unsqueeze_5=np.random.randn(1, 16, 1292, 80).astype(np.float32),
            unsqueeze_6=np.random.randn(1, 16, 1292, 80).astype(np.float32),
        )

        dummy_inputs = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "dump_test",
            "replay",
            "qwen_sdpa_attention_loopa24",
            "onnx_inputs.pt",
        )
        if os.path.exists(dummy_inputs):
            print("-- use dummy inputs")
            feeds = {k: v.detach().cpu().numpy() for k, v in torch.load(dummy_inputs).items()}
            for k, v in feeds.items():
                print(f"-- {k}: {self.string_type(v, with_shape=True, with_min_max=True)}")

        # feeds["cat_2"] = np.array([0, 1292], dtype=np.int64)
        got = sess.run(None, feeds)
        self.assertEqual(len(got), 1)
        self.assertEqual((1, 1292, 16, 80), got[0].shape)
        expected = qwen_sdpa_attention(
            torch.from_numpy(feeds["unsqueeze_4"]),
            torch.from_numpy(feeds["unsqueeze_5"]),
            torch.from_numpy(feeds["unsqueeze_6"]),
            torch.from_numpy(feeds["cat_2"]),
            scaling=0.11180339753627777,
            num_heads=16,
        )
        self.assertEqualArray(expected, got[0], atol=1e-5)

        tfeeds = {k: torch.from_numpy(v) for k, v in feeds.items()}
        ev = OnnxruntimeEvaluator(model)
        got2 = ev.run(None, tfeeds)
        self.assertEqual(len(got2), 1)
        self.assertEqualArray(got[0], got2[0], atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
