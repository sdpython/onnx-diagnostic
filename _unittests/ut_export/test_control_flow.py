import unittest
from typing import Tuple
import torch
from onnxscript import script, FLOAT, INT64
from onnxscript import opset18 as op
from onnx_diagnostic.ext_test_case import ExtTestCase
from onnx_diagnostic.export.control_flow import enable_code_export_control_flow, loop_for
from onnx_diagnostic.export.control_flow_research import simple_loop_for as loop_for_r
from onnx_diagnostic.export.api import to_onnx


class TestControlFlow(ExtTestCase):
    @unittest.skip("not working")
    def test_loop_one_research(self):
        class Model(torch.nn.Module):
            def forward(self, n_iter, x):
                def body(i: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor]:
                    return (x[: i.item() + 1].unsqueeze(1),)

                return loop_for_r(n_iter, body, (x,))[0]

        model = Model()
        n_iter = torch.tensor(4, dtype=torch.int64)
        x = torch.arange(10, dtype=torch.float32)
        expected = torch.tensor([0, 0, 1, 0, 1, 2, 0, 1, 2, 3], dtype=x.dtype).unsqueeze(1)
        got = model(n_iter, x)
        self.assertEqualArray(expected, got)

        with enable_code_export_control_flow():
            got = model(n_iter, x)
        self.assertEqualArray(expected, got)

        ep = torch.export.export(
            model, (n_iter, x), dynamic_shapes=({}, ({0: torch.export.Dim.DYNAMIC}))
        )
        print(ep)

    def test_onnxscript_loop(self):
        @script()
        def concatenation(N: INT64[1], x: FLOAT[None]) -> FLOAT[None, 1]:
            copy = op.Identity(x)
            res = op.SequenceEmpty()
            for i in range(N):
                res = op.SequenceInsert(res, op.Unsqueeze(copy[:i], [1]))
            return op.ConcatFromSequence(res, axis=1)

        onx = concatenation.to_model_proto()
        self.dump_onnx("test_onnxscript_loop.onnx", onx)

    def test_loop_one_custom(self):
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
        self.assertIn("torch.ops.onnx_higher_ops.loop_for_body_", str(ep))

        onx = to_onnx(
            model,
            (n_iter, x),
            dynamic_shapes=({}, ({0: torch.export.Dim.DYNAMIC})),
            exporter="custom",
            use_control_flow_dispatcher=True,
        ).model_proto
        self.dump_onnx("test_loop_one_custom.onnx", onx)
        self.assert_onnx_disc("test_loop_one_custom", onx, model, (n_iter, x))

    def test_loop_two_custom(self):
        class Model(torch.nn.Module):
            def forward(self, n_iter, x):
                def body(i, x):
                    return x[: i.item() + 1].unsqueeze(1), x[: i.item() + 1].unsqueeze(1) + 1

                res = loop_for(n_iter, body, (x,))
                return res[0] + res[1]

        model = Model()
        n_iter = torch.tensor(4, dtype=torch.int64)
        x = torch.arange(10, dtype=torch.float32)
        expected = torch.tensor([1, 1, 3, 1, 3, 5, 1, 3, 5, 7], dtype=x.dtype).unsqueeze(1)
        got = model(n_iter, x)
        self.assertEqualArray(expected, got)

        ep = torch.export.export(
            model, (n_iter, x), dynamic_shapes=({}, ({0: torch.export.Dim.DYNAMIC}))
        )
        self.assertIn("torch.ops.onnx_higher_ops.loop_for_body_", str(ep))

        onx = to_onnx(
            model,
            (n_iter, x),
            dynamic_shapes=({}, ({0: torch.export.Dim.DYNAMIC})),
            exporter="custom",
            use_control_flow_dispatcher=True,
        ).model_proto
        self.dump_onnx("test_loop_one_custom.onnx", onx)
        self.assert_onnx_disc("test_loop_one_custom", onx, model, (n_iter, x))

    def test_loop_two_custom_reduction_dim(self):
        class Model(torch.nn.Module):
            def forward(self, n_iter, x):
                def body(i, x):
                    return x[: i.item() + 1].unsqueeze(1), x[: i.item() + 1].unsqueeze(0) + 1

                res = loop_for(n_iter, body, (x,), reduction_dim=[0, 1])
                return res[0] + res[1].T

        model = Model()
        n_iter = torch.tensor(4, dtype=torch.int64)
        x = torch.arange(10, dtype=torch.float32)
        expected = torch.tensor([1, 1, 3, 1, 3, 5, 1, 3, 5, 7], dtype=x.dtype).unsqueeze(1)
        got = model(n_iter, x)
        self.assertEqualArray(expected, got)

        ep = torch.export.export(
            model, (n_iter, x), dynamic_shapes=({}, ({0: torch.export.Dim.DYNAMIC}))
        )
        self.assertIn("torch.ops.onnx_higher_ops.loop_for_body_", str(ep))

        onx = to_onnx(
            model,
            (n_iter, x),
            dynamic_shapes=({}, ({0: torch.export.Dim.DYNAMIC})),
            exporter="custom",
            use_control_flow_dispatcher=True,
        ).model_proto
        self.dump_onnx("test_loop_one_custom.onnx", onx)
        self.assert_onnx_disc("test_loop_one_custom", onx, model, (n_iter, x))


if __name__ == "__main__":
    unittest.main(verbosity=2)
