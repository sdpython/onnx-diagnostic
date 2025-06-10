"""
.. _l-plot-nonzero:

Half certain nonzero
====================

:func:`torch.nonzero` returns the indices of the first zero found
in a tensor. The output shape is unknown in the generic case
but... If you have a 2D tensor with at least a nonzero value
in every row, you can guess the dimension. But :func:`torch.export.export`
does not know what you know.


A Model
+++++++
"""

import torch
from onnx_diagnostic import doc


class Model(torch.nn.Module):
    def adaptive_enc_mask(self, x_len, chunk_start_idx, left_window=0, right_window=0):
        chunk_start_idx = torch.Tensor(chunk_start_idx).long()
        start_pad = torch.cat((torch.tensor([0], dtype=torch.int64), chunk_start_idx), dim=0)
        end_pad = torch.cat((chunk_start_idx, torch.tensor([x_len], dtype=torch.int64)), dim=0)
        seq_range = torch.arange(0, x_len).unsqueeze(-1)
        idx = ((seq_range < end_pad) & (seq_range >= start_pad)).nonzero()[:, 1]
        seq_range_expand = torch.arange(0, x_len).unsqueeze(0).expand(x_len, -1)
        idx_left = idx - left_window
        idx_left[idx_left < 0] = 0
        boundary_left = start_pad[idx_left]
        mask_left = seq_range_expand >= boundary_left.unsqueeze(-1)
        idx_right = idx + right_window
        idx_right[idx_right > len(chunk_start_idx)] = len(chunk_start_idx)
        boundary_right = end_pad[idx_right]
        mask_right = seq_range_expand < boundary_right.unsqueeze(-1)
        return mask_left & mask_right

    def forward(self, x, y):
        return self.adaptive_enc_mask(
            x.shape[1], torch.tensor([], dtype=torch.int64), left_window=y.shape[0]
        )


model = Model()
x, y = torch.rand((2, 546)), torch.rand((18,))
z = model(x, y)
print(f"y.shape={x.shape}, y.shape={y.shape}, z.shape={z.shape}")

# %%
# Export
# ++++++

DYN = torch.export.Dim.DYNAMIC
ep = torch.export.export(model, (x, y), dynamic_shapes=({0: DYN, 1: DYN}, {0: DYN}))
print(ep)


# %%
# We can see the following line in the exported program.
# It tells what it cannot verify.
# ``torch.ops.aten._assert_scalar.default(eq,``
# ``"Runtime assertion failed for expression Eq(s16, u0) on node 'eq'");``


# %%
doc.plot_legend("dynamic shapes\nnonzero", "dynamic shapes", "yellow")
