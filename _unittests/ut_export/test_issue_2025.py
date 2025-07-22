import unittest
import numpy as np
import torch
from onnx_diagnostic.ext_test_case import ExtTestCase


class TestIssues2025(ExtTestCase):
    def test_issue_158786_qwen2vl(self):
        # https://github.com/pytorch/pytorch/issues/158786
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.spatial_merge_size = 2  # Default

            def forward(self, a):
                pos_ids = []
                for t, h, w in a:
                    t = t.item()
                    h = h.item()
                    w = w.item()
                    torch._constrain_as_size(t)
                    torch._constrain_as_size(h)
                    torch._constrain_as_size(w)
                    hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
                    hpos_ids = hpos_ids.reshape(
                        h // self.spatial_merge_size,
                        self.spatial_merge_size,
                        w // self.spatial_merge_size,
                        self.spatial_merge_size,
                    )
                    hpos_ids = hpos_ids.permute(0, 2, 1, 3)
                    hpos_ids = hpos_ids.flatten()

                    wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
                    wpos_ids = wpos_ids.reshape(
                        h // self.spatial_merge_size,
                        self.spatial_merge_size,
                        w // self.spatial_merge_size,
                        self.spatial_merge_size,
                    )
                    wpos_ids = wpos_ids.permute(0, 2, 1, 3)
                    wpos_ids = wpos_ids.flatten()
                    pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
                pos_ids = torch.cat(pos_ids, dim=0)
                return pos_ids

        model = Model()
        inputs = torch.tensor(np.array([1, 98, 146]).reshape(1, 3))
        ep = torch.export.export(model, (inputs,))
        self.assertIn("torch.ops.aten.cat.default", str(ep))


if __name__ == "__main__":
    unittest.main(verbosity=2)
