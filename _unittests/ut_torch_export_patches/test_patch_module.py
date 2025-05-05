import ast
import inspect
import textwrap
import unittest
import torch
from onnx_diagnostic.ext_test_case import ExtTestCase, hide_stdout
from onnx_diagnostic.torch_export_patches.patch_module import (
    transform_method,
    inplace_add_parent,
)


class TestPatchModule(ExtTestCase):
    def test_parent(self):
        class Model(torch.nn.Module):
            def forward(self, x, y):
                if x.sum() > 0:
                    return x + y
                else:
                    return torch.abs(x) + y + 1

        src = inspect.getsource(Model.forward)
        tree = ast.parse(textwrap.dedent(src))
        inplace_add_parent(tree)
        assert all(
            hasattr(node, "parent") for node in ast.walk(tree)
        ), f"Missing parent in {ast.dump(tree, indent=2)}"

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

        rewritten = transform_method(Model.forward, verbose=self.verbose)
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

        rewritten = transform_method(Model.forward, verbose=self.verbose)
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

        rewritten = transform_method(Model.forward, verbose=self.verbose)
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
            lambda: transform_method(Model.forward, verbose=self.verbose), NotImplementedError
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

        rewritten = transform_method(Model.forward, verbose=self.verbose)
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

        rewritten = transform_method(Model.forward, verbose=self.verbose)
        self.assertIn("torch.abs(", rewritten.code)
        self.assertIn("abs", rewritten.dump)
        code = rewritten.code
        assert "w, z, u" not in code and "u, w, z" not in code, f"None dropped\n{code}"
        Model.forward = rewritten.func
        self.assertEqualAny(expected, Model()(x, y))
        self.assertEqualAny(expected_, Model()(-x, y))

        DYN = torch.export.Dim.DYNAMIC
        ds = ({0: DYN, 1: DYN}, {0: DYN, 1: DYN})
        ep = torch.export.export(Model(), (x, y), dynamic_shapes=ds)
        self.assertIn("cond", [str(getattr(n, "target", "?")) for n in ep.graph.nodes])
        self.assertEqualAny(expected, ep.module()(x, y))
        self.assertEqualAny(expected_, ep.module()(-x, y))

    def test_assign_nested_check(self):

        torch_cond = torch.cond

        class Model(torch.nn.Module):
            def forward(self, x, y):
                def torch_cond_then_3(y, x):

                    def torch_cond_then_1(y, x):
                        w = x + y
                        z = x - y
                        return (w, z)

                    def torch_cond_else_1(y, x):
                        u = x + 10
                        w = x + torch.abs(y) + u
                        z = x - torch.abs(y) + u
                        return (w, z)

                    w, z = torch_cond(
                        y.sum() > 0, torch_cond_then_1, torch_cond_else_1, [y, x]
                    )
                    return (w, z)

                def torch_cond_else_3(y, x):

                    def torch_cond_then_2(y):
                        u = y + 1
                        return u

                    def torch_cond_else_2(y):
                        u = torch.abs(y) + 10
                        return u

                    u = torch_cond(y.sum() > 0, torch_cond_then_2, torch_cond_else_2, [y])
                    w = torch.abs(x) + u
                    z = torch.abs(x) - u
                    return (w, z)

                w, z = torch_cond(x.sum() > 0, torch_cond_then_3, torch_cond_else_3, [y, x])
                return (w, z)

        x, y = torch.rand((3, 4)), torch.rand((3, 4))
        Model()(x, y)

    def test_rewrite_forward_assign_nested(self):

        class Model(torch.nn.Module):
            def forward(self, x, y):
                if x.sum() > 0:
                    if y.sum() > 0:
                        w = x + y
                        z = x - y
                    else:
                        u = x + 10
                        w = x + torch.abs(y) + u
                        z = x - torch.abs(y) + u
                else:
                    if y.sum() > 0:
                        u = y + 1
                    else:
                        u = torch.abs(y) + 10
                    w = torch.abs(x) + u
                    z = torch.abs(x) - u
                return w, z

        x, y = torch.rand((3, 4)), torch.rand((3, 4))
        expected, expected_, expected_0, expected_1 = (
            Model()(x, y),
            Model()(-x, y),
            Model()(x, -y),
            Model()(-x, -y),
        )

        rewritten = transform_method(Model.forward, verbose=self.verbose)
        self.assertIn("torch.abs(", rewritten.code)
        self.assertIn("abs", rewritten.dump)
        code = rewritten.code
        self.assertIn("branch_cond_else_3", code)
        Model.forward = rewritten.func
        self.assertEqualAny(expected, Model()(x, y))
        self.assertEqualAny(expected_, Model()(-x, y))
        self.assertEqualAny(expected_0, Model()(x, -y))
        self.assertEqualAny(expected_1, Model()(-x, -y))

        DYN = torch.export.Dim.DYNAMIC
        ds = ({0: DYN, 1: DYN}, {0: DYN, 1: DYN})
        ep = torch.export.export(Model(), (x, y), dynamic_shapes=ds)
        self.assertIn("cond", [str(getattr(n, "target", "?")) for n in ep.graph.nodes])
        self.assertEqualAny(expected, ep.module()(x, y))
        self.assertEqualAny(expected_, ep.module()(-x, y))
        self.assertEqualAny(expected_0, ep.module()(x, -y))
        self.assertEqualAny(expected_1, ep.module()(-x, -y))

    def test_rewrite_forward_none(self):

        class Model(torch.nn.Module):
            def forward(self, x, y):
                if x is None:
                    x = torch.abs(y)
                return x + y

        x, y = torch.rand((3, 4)), torch.rand((3, 4))
        expected, expected_ = Model()(x, y), Model()(-x, y)

        rewritten = transform_method(Model.forward, verbose=self.verbose)
        self.assertIn("torch.abs(", rewritten.code)
        self.assertIn("abs", rewritten.dump)
        Model.forward = rewritten.func
        self.assertEqualAny(expected, Model()(x, y))
        self.assertEqualAny(expected_, Model()(-x, y))

        DYN = torch.export.Dim.DYNAMIC
        ds = ({0: DYN, 1: DYN}, {0: DYN, 1: DYN})
        ep = torch.export.export(Model(), (x, y), dynamic_shapes=ds)
        self.assertEqualAny(expected, ep.module()(x, y))
        self.assertEqualAny(expected_, ep.module()(-x, y))

    def test_rewrite_PLBartEncoderLayer(self):
        from transformers.models.plbart.modeling_plbart import PLBartEncoderLayer

        rewritten = transform_method(PLBartEncoderLayer.forward, verbose=self.verbose)
        self.assertIn(
            (
                "torch.cond(hidden_states.dtype == torch.float16 and "
                "(torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()), "
                "branch_cond_then_1, branch_cond_else_1, [hidden_states])"
            ),
            rewritten.code,
        )
        print()
        print(rewritten.code)


if __name__ == "__main__":
    unittest.main(verbosity=2)
