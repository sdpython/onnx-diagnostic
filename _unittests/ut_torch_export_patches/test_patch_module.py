import ast
import inspect
import textwrap
import unittest
import numpy as np
from scipy.spatial.distance import cdist
import torch
from onnx_diagnostic.ext_test_case import ExtTestCase, hide_stdout, has_torch
from onnx_diagnostic.torch_export_patches import torch_export_patches, torch_export_rewrite
from onnx_diagnostic.torch_export_patches.patch_module import (
    transform_method,
    inplace_add_parent,
    ShapeFinder,
    RewriteControlFlow,
)


class _ModelForATest(torch.nn.Module):
    def forward(self, x, y):
        if x.sum() > 0:
            return x + y
        else:
            return torch.abs(x) + y + 1


def _single_forward(x, y):
    if x.sum() > 0:
        return x + y
    else:
        return torch.abs(x) + y + 1


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

    def test_rewrite_test_in_forward_return1(self):

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
    def test_rewrite_test_in_forward_return2(self):

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

    def test_rewrite_test_in_forward_assign1(self):

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

    def test_rewrite_test_in_forward_assign2(self):

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

    def test_rewrite_test_in_forward_assign_noelse(self):

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

    def test_rewrite_test_in_forward_return_noelse(self):

        class Model(torch.nn.Module):
            def forward(self, x, y):
                if x.sum() > 0:
                    return torch.abs(x) + 1 + y
                return x + y

        self.assertRaise(
            lambda: transform_method(Model.forward, verbose=self.verbose), NotImplementedError
        )

    def test_rewrite_test_in_forward_assign2_in_2(self):

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

    def test_rewrite_test_in_forward_assign2_in_3(self):

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

    def test_rewrite_test_in_forward_assign_nested(self):

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

    def test_rewrite_test_in_forward_none(self):

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

    def test_rewrite_test_in_PLBartEncoderLayer(self):
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

    @hide_stdout()
    def test_torch_export_patch_method_tuple(self):
        class Model(torch.nn.Module):
            def forward(self, x, y):
                if x.sum() > 0:
                    return x + y
                else:
                    return torch.abs(x) + y + 1

        model = Model()
        x, y = torch.rand((4, 5)), torch.rand((4, 5))
        expected = model(x, y)
        DYN = torch.export.Dim.DYNAMIC
        ds = ({0: DYN, 1: DYN}, {0: DYN, 1: DYN})
        with torch_export_patches(rewrite=[(Model, "forward")], verbose=2):
            ep = torch.export.export(model, (x, y), dynamic_shapes=ds)
            got = ep.module()(x, y)
            self.assertEqualArray(expected, got)

    @hide_stdout()
    def test_torch_export_rewrite_method_tuple(self):
        class Model(torch.nn.Module):
            def forward(self, x, y):
                if x.sum() > 0:
                    return x + y
                else:
                    return torch.abs(x) + y + 1

        model = Model()
        x, y = torch.rand((4, 5)), torch.rand((4, 5))
        expected = model(x, y)
        DYN = torch.export.Dim.DYNAMIC
        ds = ({0: DYN, 1: DYN}, {0: DYN, 1: DYN})
        with torch_export_rewrite(rewrite=[(Model, "forward")], verbose=1):
            ep = torch.export.export(model, (x, y), dynamic_shapes=ds)
            got = ep.module()(x, y)
            self.assertEqualArray(expected, got)

    def test_torch_export_rewrite_method_only(self):
        model = _ModelForATest()
        x, y = torch.rand((4, 5)), torch.rand((4, 5))
        expected = model(x, y)
        DYN = torch.export.Dim.DYNAMIC
        ds = ({0: DYN, 1: DYN}, {0: DYN, 1: DYN})
        with torch_export_rewrite(rewrite=[_ModelForATest.forward], verbose=0):
            ep = torch.export.export(model, (x, y), dynamic_shapes=ds)
            got = ep.module()(x, y)
            self.assertEqualArray(expected, got)

    @hide_stdout()
    def test_torch_export_rewrite_function(self):
        class Model(torch.nn.Module):
            def forward(self, x, y):
                return _single_forward(x, y)

        model = Model()
        x, y = torch.rand((4, 5)), torch.rand((4, 5))
        expected = model(x, y)
        DYN = torch.export.Dim.DYNAMIC
        ds = ({0: DYN, 1: DYN}, {0: DYN, 1: DYN})
        with torch_export_rewrite(rewrite=[_single_forward], verbose=1):
            ep = torch.export.export(model, (x, y), dynamic_shapes=ds)
            got = ep.module()(x, y)
            self.assertEqualArray(expected, got)

    def test_shape_finder(self):
        expr = "range(x.shape[0])"
        node = ast.parse(expr)
        sh = ShapeFinder()
        sh.visit(node)
        self.assertEqual({"x"}, sh.found_shape)

    def test__find_loop_vars(self):
        code = textwrap.dedent(
            """
            for i in range(x.shape[0]):
                z[i, :] = ((x[i : i + 1, :] - y) ** 2).sum(dim=-1)
            """
        )
        node = ast.parse(code)
        tr = RewriteControlFlow()
        vars = tr._find_loop_vars(node.body[0])
        self.assertEqual(
            {
                "init": ["z"],
                "input": ["y"],
                "loop": ["i"],
                "output": [],
                "scan": [],
                "scan_shape": ["x"],
            },
            vars,
        )

    def test_rewrite_loop(self):

        class Model(torch.nn.Module):
            def forward(self, x, y):
                z = torch.empty((x.shape[0], y.shape[0]))
                for i in range(x.shape[0]):
                    z[i, :] = ((x[i, :] - y) ** 2).sum(dim=-1)
                return z

        x, y = torch.rand((3, 4)), torch.rand((5, 4))
        expected = Model()(x, y)
        self.assertEqualArray(
            expected.numpy(),
            cdist(x.numpy(), y.numpy(), metric="sqeuclidean").astype(np.float32),
            atol=1e-5,
        )

        class RewrittenModel(torch.nn.Module):
            def forward(self, x, y):
                def loop_body_0(x, y):
                    x = x.reshape((-1, *x.shape))
                    z = ((x - y) ** 2).sum(dim=-1)
                    return (z,)

                z = torch.ops.higher_order.scan(loop_body_0, [], [x], [y])
                return z[0]

        rewritten_expected = RewrittenModel()(x, y)
        self.assertEqualArray(expected, rewritten_expected)

        DYN = torch.export.Dim.DYNAMIC
        ds = ({0: DYN, 1: DYN}, {0: DYN, 1: DYN})
        torch.export.export(RewrittenModel(), (x, y), dynamic_shapes=ds)

        class RewrittenModelLoop(torch.nn.Module):
            def forward(self, z, iv, x, y):
                z = z.clone()
                i = iv.item()
                z[i, :] = ((x[i, :] - y) ** 2).sum(dim=-1)
                return (z, iv)

        inputs = (
            torch.empty((x.shape[0], y.shape[0])),
            torch.tensor([2], dtype=torch.int64),
            x,
            y,
        )
        RewrittenModelLoop()(*inputs)
        try:
            from experimental_experiment.torch_interpreter.tracing import CustomTracer
        except ImportError:
            CustomTracer = None
        if CustomTracer:
            graph = CustomTracer().trace(RewrittenModelLoop())
            self.assertNotEmpty(graph)

        # does not wiork
        # dsl = ({0: DYN, 1: DYN}, {}, {0: DYN, 1: DYN}, {0: DYN, 1: DYN})
        # torch.export.export(RewrittenModelLoop(), inputs, dynamic_shapes=dsl)

        class RewrittenModel2(torch.nn.Module):
            def forward(self, x, y):
                def loop_body_1(z, iv, x, y):
                    z = z.clone()
                    i = iv.item()
                    z[i, :] = ((x[i, :] - y) ** 2).sum(dim=-1)
                    return (z, iv)

                z = torch.empty((x.shape[0], y.shape[0]))
                r = torch.ops.higher_order.scan(
                    loop_body_1,
                    [z],
                    [torch.arange(x.shape[0], dtype=torch.int64).reshape((-1, 1))],
                    [x, y],
                )
                return r[0]

        rewritten = transform_method(Model.forward, verbose=self.verbose)
        self.assertIn("torch.ops.higher_order.scan(", rewritten.code)
        Model.forward = rewritten.func
        self.assertEqualAny(expected, Model()(x, y))

        rewritten_expected2 = RewrittenModel2()(x, y)
        self.assertEqualArray(expected, rewritten_expected2)

        if not has_torch("2.9"):
            raise unittest.SkipTest("skipped export, torch must be >= 2.9")

        torch.export.export(RewrittenModel2(), (x, y), dynamic_shapes=ds, strict=False)
        ep = torch.export.export(Model(), (x, y), dynamic_shapes=ds, strict=False)
        self.assertEqualAny(expected, ep.module()(x, y))

        """
                    position_encodings = torch.cat(
                [weight[:, :required_pos_encodings_columns]
                 for weight in broadcasted_weights], dim=-1
            )
        """


if __name__ == "__main__":
    unittest.main(verbosity=2)
