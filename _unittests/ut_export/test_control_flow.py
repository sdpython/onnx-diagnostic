import unittest
import torch
from onnx_diagnostic.ext_test_case import ExtTestCase
from onnx_diagnostic.export.control_flow import loop_for


class TestControlFlow(ExtTestCase):
    def test_loop_for(self):
        class Model(torch.nn.Module):
            def forward(self, n_iter, x):
                def body(i, x):
                    return x[: i.item() + 1].unsqueeze(1)

                return loop_for(n_iter, body, (x,))

        model = Model()
        n_iter = torch.tensor(4, dtype=torch.int64)
        x = torch.arange(10, dtype=torch.float32)
        expected = torch.tensor([0, 0, 1, 0, 1, 2, 0, 1, 2, 3], dtype=x.dtype).unsqueeze(1)
        got = model(n_iter, x)
        self.assertEqualArray(expected, got)

        ep = torch.export.export(
            model, (n_iter, x), dynamic_shapes=({}, ({0: torch.export.Dim.DYNAMIC}))
        )
        names = set(m for m, _ in ep.module().named_modules())
        self.assertIn("", names)


if __name__ == "__main__":
    unittest.main(verbosity=2)
