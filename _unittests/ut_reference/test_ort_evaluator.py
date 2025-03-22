import unittest
from typing import Optional
import numpy as np
from onnx import ModelProto, TensorProto
from onnx.checker import check_model
import onnx.helper as oh
import onnx.numpy_helper as onh
import torch
from onnx_diagnostic.ext_test_case import (
    ExtTestCase,
    hide_stdout,
    ignore_warnings,
    requires_cuda,
)
from onnx_diagnostic.reference import ExtendedReferenceEvaluator, OnnxruntimeEvaluator

TFLOAT = TensorProto.FLOAT


class TestOnnxruntimeEvaluatoruator(ExtTestCase):
    def _range(self, *shape, bias: Optional[float] = None):
        n = np.prod(shape)
        x = np.arange(n).astype(np.float32) / n
        if bias:
            x = x + bias
        return x.reshape(tuple(shape)).astype(np.float32)

    def _get_model(self) -> ModelProto:
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Unsqueeze", ["X", "zero"], ["xu1"]),
                    oh.make_node("Unsqueeze", ["xu1", "un"], ["xu2"]),
                    oh.make_node("Reshape", ["xu2", "shape1"], ["xm1"]),
                    oh.make_node("Reshape", ["Y", "shape2"], ["xm2c"]),
                    oh.make_node("Cast", ["xm2c"], ["xm2"], to=1),
                    oh.make_node("MatMul", ["xm1", "xm2"], ["xm"]),
                    oh.make_node("Reshape", ["xm", "shape3"], ["Z"]),
                ],
                "dummy",
                [
                    oh.make_tensor_value_info("X", TFLOAT, [32, 128]),
                    oh.make_tensor_value_info("Y", TFLOAT, [3, 5, 128, 64]),
                ],
                [oh.make_tensor_value_info("Z", TFLOAT, [3, 5, 32, 64])],
                [
                    onh.from_array(np.array([0], dtype=np.int64), name="zero"),
                    onh.from_array(np.array([1], dtype=np.int64), name="un"),
                    onh.from_array(np.array([1, 32, 128], dtype=np.int64), name="shape1"),
                    onh.from_array(np.array([15, 128, 64], dtype=np.int64), name="shape2"),
                    onh.from_array(np.array([3, 5, 32, 64], dtype=np.int64), name="shape3"),
                ],
            ),
            ir_version=9,
            opset_imports=[oh.make_opsetid("", 18)],
        )
        check_model(model)
        return model

    @ignore_warnings(DeprecationWarning)
    def test_ort_eval(self):
        model = self._get_model()

        feeds = {"X": self._range(32, 128), "Y": self._range(3, 5, 128, 64)}
        ref = ExtendedReferenceEvaluator(model, verbose=10)
        expected, out, _ = self.capture(lambda: ref.run(None, feeds)[0])
        self.assertIn("Reshape(xm, shape3) -> Z", out)

        ort_eval = OnnxruntimeEvaluator(model, verbose=10, opsets=20)
        got, out, _ = self.capture(lambda: ort_eval.run(None, feeds)[0])
        self.assertEqualArray(expected, got, atol=1e-4)
        self.assertIn("Reshape(xm, shape3) -> Z", out)

    @ignore_warnings(DeprecationWarning)
    @requires_cuda()
    @hide_stdout()
    def test_ort_eval_cuda(self):
        model = self._get_model()

        feeds = {"X": self._range(32, 128), "Y": self._range(3, 5, 128, 64)}
        ref = ExtendedReferenceEvaluator(model, verbose=10)
        expected = ref.run(None, feeds)[0]

        ort_eval = OnnxruntimeEvaluator(model, verbose=10, opsets=20, providers="cuda")
        got = ort_eval.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-1)

    @ignore_warnings(DeprecationWarning)
    @hide_stdout()
    def test_ort_eval_node_proto(self):
        model = self._get_model()

        feeds = {"X": self._range(32, 128), "zero": np.array([0], dtype=np.int64)}
        ref = ExtendedReferenceEvaluator(model.graph.node[0], verbose=10)
        expected = ref.run(None, feeds)

        ort_eval = OnnxruntimeEvaluator(model.graph.node[0], verbose=10, opsets=20)
        got = ort_eval.run(None, feeds)
        self.assertEqualArrayAny(expected, got, atol=1e-4)
        self.assertIsInstance(expected[0], np.ndarray)
        self.assertIsInstance(got[0], np.ndarray)

    @ignore_warnings(DeprecationWarning)
    @hide_stdout()
    def test_ort_eval_node_proto_torch(self):
        model = self._get_model()

        feeds_np = {"X": self._range(32, 128), "zero": np.array([0], dtype=np.int64)}
        feeds = {k: torch.from_numpy(v) for k, v in feeds_np.items()}
        ref = ExtendedReferenceEvaluator(model.graph.node[0], verbose=10)
        expected = ref.run(None, feeds_np)

        ort_eval = OnnxruntimeEvaluator(model.graph.node[0], verbose=10, opsets=20)
        got = ort_eval.run(None, feeds)
        self.assertIsInstance(got[0], torch.Tensor)
        self.assertEqualArray(expected[0], got[0], atol=1e-4)

    @hide_stdout()
    def test_local_function(self):
        new_domain = "custom"

        linear_regression = oh.make_function(
            new_domain,
            "LinearRegression",
            ["x", "a", "b"],
            ["y"],
            [
                oh.make_node("MatMul", ["x", "a"], ["xa"]),
                oh.make_node("Add", ["xa", "b"], ["y"]),
            ],
            [oh.make_opsetid("", 14)],
            [],
        )

        graph = oh.make_graph(
            [
                oh.make_node("LinearRegression", ["X", "A", "B"], ["Y1"], domain=new_domain),
                oh.make_node("Abs", ["Y1"], ["Y"]),
            ],
            "example",
            [
                oh.make_tensor_value_info("X", TFLOAT, [None, None]),
                oh.make_tensor_value_info("A", TFLOAT, [None, None]),
                oh.make_tensor_value_info("B", TFLOAT, [None, None]),
            ],
            [oh.make_tensor_value_info("Y", TFLOAT, None)],
        )

        onnx_model = oh.make_model(
            graph,
            opset_imports=[oh.make_opsetid("", 14), oh.make_opsetid(new_domain, 1)],
            functions=[linear_regression],
            ir_version=10,
        )
        feeds = {
            "X": np.random.randn(3, 3).astype(np.float32),
            "A": np.random.randn(3, 3).astype(np.float32),
            "B": np.random.randn(3, 3).astype(np.float32),
        }
        ref = ExtendedReferenceEvaluator(onnx_model)
        ort_eval = OnnxruntimeEvaluator(onnx_model, verbose=10, opsets=20)
        expected = ref.run(None, feeds)
        got = ort_eval.run(None, feeds)
        self.assertEqualArray(expected[0], got[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
