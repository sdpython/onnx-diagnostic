import unittest
import torch
from onnx_diagnostic.ext_test_case import ExtTestCase, requires_torch
from onnx_diagnostic.helpers.torch_test_helper import (
    is_torchdynamo_exporting,
    fake_torchdynamo_exporting,
)
from onnx_diagnostic.helpers import string_type
from onnx_diagnostic.torch_export_patches.patch_expressions import (
    _iterate_patched_expressions,
    register_patched_expressions,
    patched_float_arange,
)


class TestOnnxExportErrors(ExtTestCase):

    def test_patched_expressions(self):
        res = list(_iterate_patched_expressions())
        names = {_[0] for _ in res}
        self.assertIn("float_arange", names)

    @requires_torch("2.7")
    def test_filter_position_ids(self):

        def filter_position_ids(
            patch_attention_mask: torch.Tensor,
            position_ids: torch.Tensor,
            boundaries: torch.Tensor,
            num_patches_per_side: int,
        ):
            for batch_idx, p_attn_mask in enumerate(patch_attention_mask):
                fractional_coords_h = torch.arange(0, 1 - 1e-6, 1 / p_attn_mask[:, 0].sum())
                fractional_coords_w = torch.arange(0, 1 - 1e-6, 1 / p_attn_mask[0].sum())

                bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
                bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)

                pos_ids = (
                    bucket_coords_h[:, None] * num_patches_per_side + bucket_coords_w
                ).flatten()
                position_ids[batch_idx][p_attn_mask.view(-1)] = pos_ids
            return position_ids

        def float_arange(start, end, step):
            length = torch.sym_int((end - start) / step + (step * (1 - 1e-6)))
            torch._check(length > 0)
            res = torch.arange(0, length)
            torch._check(res.is_contiguous())
            fres = res.to(torch.float32)
            fstart = torch.tensor(start, dtype=torch.float32)
            return fres + fstart

        def scan_filter_position_ids(
            patch_attention_mask: torch.Tensor,
            position_ids: torch.Tensor,
            boundaries: torch.Tensor,
            num_patches_per_side: int,
        ):

            def body(p_attn_mask, position_ids_row):
                h_len = torch.tensor(1) / p_attn_mask[:, 0].sum()
                w_len = torch.tensor(1) / p_attn_mask[0].sum()
                fractional_coords_h = patched_float_arange(
                    torch.tensor(0.0), torch.tensor(1 - 1e-6), h_len
                )
                fractional_coords_w = patched_float_arange(
                    torch.tensor(0.0), torch.tensor(1 - 1e-6), w_len
                )

                # torch.arange(0, 1 - 1e-6, 1 / p_attn_mask[:, 0].sum().item())
                # torch.arange(0, 1 - 1e-6, 1 / p_attn_mask[0].sum().item())

                bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
                bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)

                pos_ids = (
                    bucket_coords_h[:, None] * num_patches_per_side + bucket_coords_w
                ).flatten()

                row = position_ids_row.clone()
                row[p_attn_mask.view(-1)] = pos_ids
                return [row]

            return torch.ops.higher_order.scan(
                body, [], [patch_attention_mask, position_ids], additional_inputs=[]
            )

        class Model(torch.nn.Module):
            def forward(self, patch_attention_mask, position_ids, boundaries):
                if is_torchdynamo_exporting():
                    res = scan_filter_position_ids(
                        patch_attention_mask, position_ids, boundaries, 32
                    )
                    return res[0]
                return filter_position_ids(patch_attention_mask, position_ids, boundaries, 32)

        # 32
        # T9s32x32x32[False,True:A0.978515625],
        # T7s32x1024[0,0:A0.0],
        # T1s31[0.03125,0.96875:A0.5]]
        register_patched_expressions()
        patch_attention_mask = torch.randint(0, 20, (32, 32, 32)) >= 1
        patch_attention_mask[:, :, :] = True
        position_ids = torch.zeros((32, 1024), dtype=torch.int64)
        boundaries = (torch.arange(33).to(torch.float32) / 33)[1:-1]
        inputs = (patch_attention_mask, position_ids, boundaries)
        model = Model()
        expected = model(*inputs)
        with fake_torchdynamo_exporting():
            got = model(*inputs)
        self.assertEqual(type(expected), type(got))
        self.assertEqual(
            string_type(expected, with_shape=True), string_type(got, with_shape=True)
        )
        self.assertEqualArray(expected, got)

        DYN = torch.export.Dim.DYNAMIC
        ep = torch.export.export(model, inputs, dynamic_shapes=({0: DYN}, {0: DYN}, {0: DYN}))
        self.assertEqualArray(expected, ep.module()(*inputs))


if __name__ == "__main__":
    unittest.main(verbosity=2)
