import unittest
from onnx_diagnostic.ext_test_case import ExtTestCase
from onnx_diagnostic.torch_export_patches.patches.patch_transformers import (
    rewrite_loop_for_square_mask,
)


class TestPatchRewriting(ExtTestCase):
    def test_rewrite_loop_for_square_mask(self):
        import torch

        seq_length = 8
        dtype = torch.float32
        mask = torch.full([1, seq_length, seq_length], 1, dtype=dtype)

        def apply_mask(mask, seq):
            mask = mask.clone()
            for i in range(1, len(seq)):
                mask[..., seq[i - 1] : seq[i], seq[i - 1] : seq[i]] = 0
            return mask

        for seqi in [
            [1, 5, 8],
            [1, 5, 7],
            [2, 3, 6],
            [2, 3, 3, 6],
            [0, 1, 4, 5],
            [0, 0, 5, 6],
        ]:
            with self.subTest(seq=seqi):
                seq = torch.tensor(seqi, dtype=torch.int64)
                m1 = apply_mask(mask, seq)
                m2 = rewrite_loop_for_square_mask(mask, seq)
                self.assertEqualArray(m1, m2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
