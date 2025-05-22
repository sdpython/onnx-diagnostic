import unittest
import torch
from onnx_diagnostic.ext_test_case import ExtTestCase, requires_torch
from onnx_diagnostic.torch_export_patches.patch_helper import py_vmap


class TestPatchHelper(ExtTestCase):
    def test_vmap(self):
        f = lambda x, y: x * y + 1  # noqa: E731
        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.tensor([0.1, 0.2, 0.3])
        expected = torch.vmap(f)(x, y)
        got = py_vmap(f)(x, y)
        self.assertEqualArray(expected, got)

    @requires_torch("2.9")
    def test_export_vmap(self):
        class Model(torch.nn.Module):
            def forward(self, x, y):
                f = lambda x, y: x * y + 1  # noqa: E731
                return torch.vmap(f)(x, y)

        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.tensor([0.1, 0.2, 0.3])
        DYN = torch.export.Dim.DYNAMIC
        torch.export.export(Model(), (x, y), ({0: DYN}, {1: DYN}))

    def test_export_py_vmap(self):
        class Model(torch.nn.Module):
            def forward(self, x, y):
                f = lambda x, y: x * y + 1  # noqa: E731
                return py_vmap(f)(x, y)

        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.tensor([0.1, 0.2, 0.3])
        torch.export.export(Model(), (x, y))

    def test_vmap_outdim(self):
        f = lambda x: x**2  # noqa: E731
        x = torch.randn(2, 5)
        expected = torch.vmap(f, out_dims=1)(x)
        got = py_vmap(f, out_dims=1)(x)
        self.assertEqualArray(expected, got)

    def test_vmap_dict(self):
        f = lambda d: torch.dot(d["x"], d["y"])  # noqa: E731
        x, y = torch.randn(2, 5), torch.randn(5)
        input = {"x": x, "y": y}
        _expected = torch.vmap(f, in_dims=({"x": 0, "y": None},))(input)
        self.assertRaise(
            lambda: py_vmap(f, in_dims=({"x": 0, "y": None},))(input), AssertionError
        )
        # self.assertEqualArray(_expected, got)

    def test_vmap_tuple(self):
        x, y = torch.randn(2, 5), torch.randn(5)
        expected = torch.vmap(torch.dot, in_dims=(0, None))(x, y)
        got = py_vmap(torch.dot, in_dims=(0, None))(x, y)
        self.assertEqualArray(expected, got)


if __name__ == "__main__":
    unittest.main(verbosity=2)
