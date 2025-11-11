import unittest
from typing import Tuple
import torch
from onnx_diagnostic.ext_test_case import ExtTestCase
from onnx_diagnostic.export.control_flow import enable_code_export_control_flow, loop_for


class TestControlFlow(ExtTestCase):
    def test_loop_one(self):
        # https://github.com/pytorch/pytorch/issues/158786
        class Model(torch.nn.Module):
            def forward(self, n_iter, x):
                def body(i: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor]:
                    return (x[: i.item() + 1].unsqueeze(1),)

                return loop_for(body, n_iter, x)[0]

        model = Model()
        n_iter = torch.tensor(4, dtype=torch.int64)
        x = torch.arange(10, dtype=torch.float32)
        expected = torch.tensor(
            [0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4], dtype=x.dtype
        ).unsqueeze(1)
        got = model(n_iter, x)
        self.assertEqualArray(expected, got)

        with enable_code_export_control_flow():
            got = model(n_iter, x)
        self.assertEqualArray(expected, got)


if __name__ == "__main__":
    unittest.main(verbosity=2)
