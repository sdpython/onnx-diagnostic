import unittest
import onnx
import onnx.helper as oh
import torch
from onnx_diagnostic.ext_test_case import ExtTestCase, has_torch, hide_stdout, ignore_warnings
from onnx_diagnostic.export.onnx_plug import EagerDirectReplacementWithOnnx
from onnx_diagnostic.export.api import to_onnx


class TestOnnxPlus(ExtTestCase):
    def test_onnx_plug_verify(self):
        def _test_customadd(x, y):
            return x + y

        def _test_customadd_shape(x, y):
            return torch.empty(torch.broadcast_shapes(x.shape, y.shape), dtype=x.dtype)

        def make_function_proto():
            return oh.make_function(
                "onnx_plug",
                "_test_customadd",
                ["x", "y"],
                ["z"],
                [oh.make_node("Add", ["x", "y"], ["z"])],
                opset_imports=[oh.make_opsetid("", 22)],
            )

        rep = EagerDirectReplacementWithOnnx(
            _test_customadd, _test_customadd_shape, make_function_proto(), 2, 1
        )

        x = torch.randn((3, 4), dtype=torch.float32)
        y = torch.randn((3, 1), dtype=torch.float32)
        self.assertEqualArray(_test_customadd(x, y), x + y)
        res = rep.verify(x, y)
        self.assertEqualAny(res.eager_outputs, (x + y,))
        self.assertEqual(len(res.diffs), 1)
        self.assertEqual(res.diffs[0]["abs"], 0)

    @hide_stdout()
    @ignore_warnings(FutureWarning)
    def test_onnx_plug_export_nokwargs(self):
        def _test_customsub(x, y):
            return x - y

        def _test_customsub_shape(x, y):
            return torch.empty(torch.broadcast_shapes(x.shape, y.shape), dtype=x.dtype)

        def make_function_proto():
            return oh.make_function(
                "onnx_plug",
                "_test_customsub",
                ["x", "y"],
                ["z"],
                [oh.make_node("Sub", ["x", "y"], ["z"])],
                opset_imports=[oh.make_opsetid("", 22)],
            )

        class Model(torch.nn.Module):
            def forward(self, x):
                y = x.sum(axis=1, keepdim=True)
                d = torch.ops.onnx_plug._test_customsub(x, y)
                return torch.abs(d)

        replacements = [
            EagerDirectReplacementWithOnnx(
                _test_customsub, _test_customsub_shape, make_function_proto(), 2, 1, verbose=1
            )
        ]

        x = torch.randn((3, 4), dtype=torch.float32)
        model = Model()
        expected = model(x)
        ds = ({0: "d1", 1: "d2"},)
        ep = torch.export.export(model, (x,), dynamic_shapes=self.use_dyn_not_str(ds))
        self.assertIn("torch.ops.onnx_plug._test_customsub.default", str(ep))
        got = ep.module()(x)
        self.assertEqualArray(expected, got)

        with self.subTest(exporter="custom"):
            onx = to_onnx(
                model,
                (x,),
                dynamic_shapes=ds,
                exporter="custom",
                onnx_plugs=replacements,
                target_opset=22,
            )
            self.assert_onnx_disc(
                "test_onnx_plug_export_nokwargs_custom", onx.model_proto, model, (x,)
            )

        if not has_torch("2.9"):
            raise unittest.SkipTest("onnx-dynamo + custom op not fully working on 2.8")
        with self.subTest(exporter="onnx-dynamo"):
            onx = to_onnx(
                model,
                (x,),
                dynamic_shapes=ds,
                exporter="onnx-dynamo",
                onnx_plugs=replacements,
                target_opset=22,
            )
            self.assert_onnx_disc(
                "test_onnx_plug_export_nokwargs_onnx_dynamo", onx.model_proto, model, (x,)
            )

    @unittest.skip("not ready yet")
    @hide_stdout()
    @ignore_warnings(FutureWarning)
    def test_onnx_plug_export_kwargs(self):
        def _test_customdiv(x, y, epsilon: float = 1e-5):
            return x / (y + epsilon)

        def _test_customdiv_shape(x, y, *args, **kwargs):
            return torch.empty(torch.broadcast_shapes(x.shape, y.shape), dtype=x.dtype)

        def make_function_proto():
            f = oh.make_function(
                "onnx_plug",
                "_test_customdiv",
                ["x", "y"],
                ["z"],
                [
                    oh.make_node("Constant", [], ["eps"]),
                    oh.make_node("Add", ["y", "eps"], ["yeps"]),
                    oh.make_node("Div", ["x", "yeps"], ["z"]),
                ],
                opset_imports=[oh.make_opsetid("", 22)],
                attributes=["epsilon"],
            )
            att = onnx.AttributeProto()
            att.name = "value_float"
            att.ref_attr_name = "epsilon"
            att.type = onnx.AttributeProto.FLOAT
            f.node[0].attribute.append(att)
            return f

        class Model(torch.nn.Module):
            def forward(self, x):
                y = x.sum(axis=1, keepdim=True)
                d = torch.ops.onnx_plug._test_customdiv(x, y, epsilon=3.5)
                return torch.abs(d)

        replacements = [
            EagerDirectReplacementWithOnnx(
                _test_customdiv,
                _test_customdiv_shape,
                make_function_proto(),
                2,
                1,
                kwargs=dict(epsilon=1e-5),
                verbose=1,
            )
        ]

        x = torch.randn((3, 4), dtype=torch.float32)
        model = Model()
        expected = model(x)
        ds = ({0: "d1", 1: "d2"},)
        ep = torch.export.export(model, (x,), dynamic_shapes=self.use_dyn_not_str(ds))
        self.assertIn("torch.ops.onnx_plug._test_customdiv.default", str(ep))
        got = ep.module()(x)
        self.assertEqualArray(expected, got)

        with self.subTest(exporter="custom"):
            onx = to_onnx(
                model,
                (x,),
                dynamic_shapes=ds,
                exporter="custom",
                onnx_plugs=replacements,
                target_opset=22,
            )
            self.assert_onnx_disc(
                "test_onnx_plug_export_kwargs_custom", onx.model_proto, model, (x,)
            )

        if not has_torch("2.9"):
            raise unittest.SkipTest("onnx-dynamo + custom op not fully working on 2.8")
        with self.subTest(exporter="onnx-dynamo"):
            onx = to_onnx(
                model,
                (x,),
                dynamic_shapes=ds,
                exporter="onnx-dynamo",
                onnx_plugs=replacements,
                target_opset=22,
            )
            self.assert_onnx_disc(
                "test_onnx_plug_export_kwargs_onnx_dynamo", onx.model_proto, model, (x,)
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
