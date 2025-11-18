import unittest
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
    def test_onnx_plug_export(self):
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
            self.assert_onnx_disc("test_onnx_plug_export_custom", onx.model_proto, model, (x,))

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
                "test_onnx_plug_export_onnx_dynamo", onx.model_proto, model, (x,)
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
