import unittest
import torch
from onnx_diagnostic.ext_test_case import ExtTestCase, hide_stdout
from onnx_diagnostic.torch_export_patches.patch_module import transform_method


class TestPatchModule(ExtTestCase):
    def test_rewrite_forward_return1(self):

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                if x.sum() > 0:
                    return x + y
                else:
                    return torch.abs(x) + y

        x, y = torch.rand((3, 4)), torch.rand((3, 4))
        expected = Model()(x, y)

        rewritten = transform_method(Model.forward)
        Model.forward = rewritten.func
        Model()(x, y)

        DYN = torch.export.Dim.DYNAMIC
        ds = ({0: DYN, 1: DYN}, {0: DYN, 1: DYN})
        ep = torch.export.export(Model(), (x, y), dynamic_shapes=ds)
        self.assertIn("cond", [str(getattr(n, "target", "?")) for n in ep.graph.nodes])
        got = ep.module()(x, y)
        self.assertEqualArray(expected, got)

    @hide_stdout()
    def test_rewrite_forward_return2(self):

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                if x.sum() > 0:
                    return x + y, x - y
                else:
                    return torch.abs(x) + y, torch.abs(x) - y

        x, y = torch.rand((3, 4)), torch.rand((3, 4))
        expected = Model()(x, y)

        rewritten = transform_method(Model.forward, verbose=10)
        Model.forward = rewritten.func
        Model()(x, y)

        DYN = torch.export.Dim.DYNAMIC
        ds = ({0: DYN, 1: DYN}, {0: DYN, 1: DYN})
        ep = torch.export.export(Model(), (x, y), dynamic_shapes=ds)
        self.assertIn("cond", [str(getattr(n, "target", "?")) for n in ep.graph.nodes])
        got = ep.module()(x, y)
        self.assertEqualAny(expected, got)
        self.assertEqualAny(Model()(-x, y), ep.module()(-x, y))

    def test_rewrite_forward_assign1(self):

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                if x.sum() > 0:
                    z = x + y
                else:
                    z = torch.abs(x) + y
                return z

        x, y = torch.rand((3, 4)), torch.rand((3, 4))
        expected = Model()(x, y)

        rewritten = transform_method(Model.forward, verbose=0)
        Model.forward = rewritten.func
        Model()(x, y)

        DYN = torch.export.Dim.DYNAMIC
        ds = ({0: DYN, 1: DYN}, {0: DYN, 1: DYN})
        ep = torch.export.export(Model(), (x, y), dynamic_shapes=ds)
        self.assertIn("cond", [str(getattr(n, "target", "?")) for n in ep.graph.nodes])
        got = ep.module()(x, y)
        self.assertEqualArray(expected, got)
        self.assertEqualArray(Model()(-x, y), ep.module()(-x, y))

    def test_rewrite_forward_assign2(self):

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                if x.sum() > 0:
                    w, z = x + y, x - y
                else:
                    w, z = torch.abs(x) + y, torch.abs(x) - y
                return w, z

        x, y = torch.rand((3, 4)), torch.rand((3, 4))
        expected = Model()(x, y)

        rewritten = transform_method(Model.forward, verbose=0)
        Model.forward = rewritten.func
        Model()(x, y)

        DYN = torch.export.Dim.DYNAMIC
        ds = ({0: DYN, 1: DYN}, {0: DYN, 1: DYN})
        ep = torch.export.export(Model(), (x, y), dynamic_shapes=ds)
        self.assertIn("cond", [str(getattr(n, "target", "?")) for n in ep.graph.nodes])
        got = ep.module()(x, y)
        self.assertEqualAny(expected, got)
        self.assertEqualAny(Model()(-x, y), ep.module()(-x, y))

    def test_rewrite_forward_noelse(self):

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                if x.sum() > 0:
                    x = torch.abs(x)
                return x + y

        x, y = torch.rand((3, 4)), torch.rand((3, 4))
        expected = Model()(x, y)

        rewritten = transform_method(Model.forward, verbose=0)
        Model.forward = rewritten.func
        Model()(x, y)

        DYN = torch.export.Dim.DYNAMIC
        ds = ({0: DYN, 1: DYN}, {0: DYN, 1: DYN})
        ep = torch.export.export(Model(), (x, y), dynamic_shapes=ds)
        self.assertIn("cond", [str(getattr(n, "target", "?")) for n in ep.graph.nodes])
        got = ep.module()(x, y)
        self.assertEqualAny(expected, got)
        self.assertEqualAny(Model()(-x, y), ep.module()(-x, y))
        self.assertEqualAny(Model()(-x, y), ep.module()(-x, y))


if __name__ == "__main__":
    unittest.main(verbosity=2)
