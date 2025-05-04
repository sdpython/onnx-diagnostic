import unittest
import torch
from onnx_diagnostic.ext_test_case import ExtTestCase, hide_stdout
from onnx_diagnostic.torch_export_patches.patch_module import transform_method


class TestPatchModule(ExtTestCase):
    def test_rewrite_forward_return1(self):

        class Model(torch.nn.Module):
            def forward(self, x, y):
                if x.sum() > 0:
                    return x + y
                else:
                    return torch.abs(x) + y + 1

        x, y = torch.rand((3, 4)), torch.rand((3, 4))
        expected, expected_ = Model()(x, y), Model()(-x, y)

        rewritten = transform_method(Model.forward)
        self.assertIn("torch.abs(", rewritten.code)
        self.assertIn("'abs'", rewritten.dump)
        Model.forward = rewritten.func
        self.assertEqualAny(expected, Model()(x, y))
        self.assertEqualAny(expected_, Model()(-x, y))

        DYN = torch.export.Dim.DYNAMIC
        ds = ({0: DYN, 1: DYN}, {0: DYN, 1: DYN})
        ep = torch.export.export(Model(), (x, y), dynamic_shapes=ds)
        self.assertIn("cond", [str(getattr(n, "target", "?")) for n in ep.graph.nodes])
        self.assertEqualAny(expected, ep.module()(x, y))
        self.assertEqualAny(expected_, ep.module()(-x, y))

    @hide_stdout()
    def test_rewrite_forward_return2(self):

        class Model(torch.nn.Module):
            def forward(self, x, y):
                if x.sum() > 0:
                    return x + y, x - y
                else:
                    return torch.abs(x) + y + 1, torch.abs(x) - y + 1

        x, y = torch.rand((3, 4)), torch.rand((3, 4))
        expected, expected_ = Model()(x, y), Model()(-x, y)

        rewritten = transform_method(Model.forward, verbose=10)
        self.assertIn("torch.abs(", rewritten.code)
        self.assertIn("abs", rewritten.dump)
        Model.forward = rewritten.func
        self.assertEqualAny(expected, Model()(x, y))
        self.assertEqualAny(expected_, Model()(-x, y))

        DYN = torch.export.Dim.DYNAMIC
        ds = ({0: DYN, 1: DYN}, {0: DYN, 1: DYN})
        ep = torch.export.export(Model(), (x, y), dynamic_shapes=ds)
        self.assertIn("cond", [str(getattr(n, "target", "?")) for n in ep.graph.nodes])
        self.assertEqualAny(expected, ep.module()(x, y))
        self.assertEqualAny(expected_, ep.module()(-x, y))

    def test_rewrite_forward_assign1(self):

        class Model(torch.nn.Module):
            def forward(self, x, y):
                if x.sum() > 0:
                    z = x + y
                else:
                    z = torch.abs(x) + y + 1
                return z

        x, y = torch.rand((3, 4)), torch.rand((3, 4))
        expected, expected_ = Model()(x, y), Model()(-x, y)

        rewritten = transform_method(Model.forward, verbose=0)
        self.assertIn("torch.abs(", rewritten.code)
        self.assertIn("abs", rewritten.dump)
        Model.forward = rewritten.func
        self.assertEqualAny(expected, Model()(x, y))
        self.assertEqualAny(expected_, Model()(-x, y))

        DYN = torch.export.Dim.DYNAMIC
        ds = ({0: DYN, 1: DYN}, {0: DYN, 1: DYN})
        ep = torch.export.export(Model(), (x, y), dynamic_shapes=ds)
        self.assertIn("cond", [str(getattr(n, "target", "?")) for n in ep.graph.nodes])
        self.assertEqualAny(expected, ep.module()(x, y))
        self.assertEqualArray(expected_, ep.module()(-x, y))

    def test_rewrite_forward_assign2(self):

        class Model(torch.nn.Module):
            def forward(self, x, y):
                if x.sum() > 0:
                    w, z = x + y, x - y
                else:
                    w, z = torch.abs(x) + y + 1, torch.abs(x) - y + 1
                return w, z

        x, y = torch.rand((3, 4)), torch.rand((3, 4))
        expected, expected_ = Model()(x, y), Model()(-x, y)

        rewritten = transform_method(Model.forward, verbose=0)
        self.assertIn("torch.abs(", rewritten.code)
        self.assertIn("abs", rewritten.dump)
        Model.forward = rewritten.func
        self.assertEqualAny(expected, Model()(x, y))
        self.assertEqualAny(expected_, Model()(-x, y))

        DYN = torch.export.Dim.DYNAMIC
        ds = ({0: DYN, 1: DYN}, {0: DYN, 1: DYN})
        ep = torch.export.export(Model(), (x, y), dynamic_shapes=ds)
        self.assertIn("cond", [str(getattr(n, "target", "?")) for n in ep.graph.nodes])
        self.assertEqualAny(expected, ep.module()(x, y))
        self.assertEqualAny(expected_, ep.module()(-x, y))

    def test_rewrite_forward_assign_noelse(self):

        class Model(torch.nn.Module):
            def forward(self, x, y):
                if x.sum() > 0:
                    x = torch.abs(x) + 1
                return x + y

        x, y = torch.rand((3, 4)), torch.rand((3, 4))
        expected, expected_ = Model()(x, y), Model()(-x, y)

        rewritten = transform_method(Model.forward, verbose=0)
        self.assertIn("torch.abs(", rewritten.code)
        self.assertIn("abs", rewritten.dump)
        Model.forward = rewritten.func
        self.assertEqualAny(expected, Model()(x, y))
        self.assertEqualAny(expected_, Model()(-x, y))

        DYN = torch.export.Dim.DYNAMIC
        ds = ({0: DYN, 1: DYN}, {0: DYN, 1: DYN})
        ep = torch.export.export(Model(), (x, y), dynamic_shapes=ds)
        self.assertIn("cond", [str(getattr(n, "target", "?")) for n in ep.graph.nodes])
        self.assertEqualAny(expected, ep.module()(x, y))
        self.assertEqualAny(expected_, ep.module()(-x, y))

    def test_rewrite_forward_return_noelse(self):

        class Model(torch.nn.Module):
            def forward(self, x, y):
                if x.sum() > 0:
                    return torch.abs(x) + 1 + y
                return x + y

        self.assertRaise(
            lambda: transform_method(Model.forward, verbose=0), NotImplementedError
        )

    def test_rewrite_forward_assign2_in_2(self):

        class Model(torch.nn.Module):
            def forward(self, x, y):
                if x.sum() > 0:
                    w = x + y
                    z = x - y
                else:
                    w = torch.abs(x) + y + 1
                    z = torch.abs(x) - y + 1
                return w, z

        x, y = torch.rand((3, 4)), torch.rand((3, 4))
        expected, expected_ = Model()(x, y), Model()(-x, y)

        rewritten = transform_method(Model.forward, verbose=0)
        self.assertIn("torch.abs(", rewritten.code)
        self.assertIn("abs", rewritten.dump)
        Model.forward = rewritten.func
        self.assertEqualAny(expected, Model()(x, y))
        self.assertEqualAny(expected_, Model()(-x, y))

        DYN = torch.export.Dim.DYNAMIC
        ds = ({0: DYN, 1: DYN}, {0: DYN, 1: DYN})
        ep = torch.export.export(Model(), (x, y), dynamic_shapes=ds)
        self.assertIn("cond", [str(getattr(n, "target", "?")) for n in ep.graph.nodes])
        self.assertEqualAny(expected, ep.module()(x, y))
        self.assertEqualAny(expected_, ep.module()(-x, y))

    def test_rewrite_forward_assign2_in_3(self):

        class Model(torch.nn.Module):
            def forward(self, x, y):
                if x.sum() > 0:
                    w = x + y
                    z = x - y
                else:
                    u = y + 1
                    w = torch.abs(x) + u
                    z = torch.abs(x) - u
                return w, z

        x, y = torch.rand((3, 4)), torch.rand((3, 4))
        expected, expected_ = Model()(x, y), Model()(-x, y)

        rewritten = transform_method(Model.forward, verbose=0)
        self.assertIn("torch.abs(", rewritten.code)
        self.assertIn("abs", rewritten.dump)
        code = rewritten.code
        assert ("w, z, u" in code and "u, w, z" not in code) or (
            "w, z, u" not in code and "u, w, z" in code
        ), f"Order mismatch in\n{code}"
        Model.forward = rewritten.func
        self.assertEqualAny(expected, Model()(x, y))
        self.assertEqualAny(expected_, Model()(-x, y))

        DYN = torch.export.Dim.DYNAMIC
        ds = ({0: DYN, 1: DYN}, {0: DYN, 1: DYN})
        ep = torch.export.export(Model(), (x, y), dynamic_shapes=ds)
        self.assertIn("cond", [str(getattr(n, "target", "?")) for n in ep.graph.nodes])
        self.assertEqualAny(expected, ep.module()(x, y))
        self.assertEqualAny(expected_, ep.module()(-x, y))


if __name__ == "__main__":
    unittest.main(verbosity=2)
