import unittest
import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh
import torch
from onnx_diagnostic.ext_test_case import ExtTestCase
from onnx_diagnostic.reference import ExtendedReferenceEvaluator, TorchEvaluator
from onnx_diagnostic.reference.torch_evaluator import get_kernels


TFLOAT = onnx.TensorProto.FLOAT
TINT64 = onnx.TensorProto.INT64


class TestTorchEvaluator(ExtTestCase):
    def test_kernels(self):
        ker = get_kernels()
        self.assertIsInstance(ker, dict)
        key = "", "Add", 1
        self.assertIn(key, ker)
        kernel = ker[key]
        self.assertEqual("Add_1", kernel.__name__)

    def _finalize_test(self, model, *args):
        onnx.checker.check_model(model)
        feeds = dict(zip([i.name for i in model.graph.input], args))

        expected = ExtendedReferenceEvaluator(model).run(
            None, {k: v.numpy() for k, v in feeds.items()}
        )
        rt = TorchEvaluator(model)
        got = rt.run(None, feeds)
        self.assertEqualAny(expected, [g.detach().numpy() for g in got])

    def test_op_binary(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Add", ["X", "un"], ["xy"]),
                    oh.make_node("Mul", ["xy", "Y"], ["xyy"]),
                    oh.make_node(
                        "Constant",
                        [],
                        ["deux"],
                        value=onh.from_array(np.array([2], dtype=np.float32)),
                    ),
                    oh.make_node("Div", ["xyy", "deux"], ["xyyy"]),
                    oh.make_node("Sub", ["xyyy", "Y"], ["Z"]),
                ],
                "dummy",
                [
                    oh.make_tensor_value_info("X", TFLOAT, ["a", "b"]),
                    oh.make_tensor_value_info("Y", TFLOAT, ["a", "b"]),
                ],
                [oh.make_tensor_value_info("Z", TFLOAT, ["a", "b"])],
                [onh.from_array(np.array([1], dtype=np.float32), name="un")],
            ),
            ir_version=9,
            opset_imports=[oh.make_opsetid("", 18)],
        )
        onnx.checker.check_model(model)

        rt = TorchEvaluator(model)
        self.assertEqual(5, len(rt.kernels))
        self.assertEqual(2, len(rt.constants))

        feeds = dict(
            X=torch.rand((4, 5), dtype=torch.float32),
            Y=torch.rand((4, 5), dtype=torch.float32),
        )

        expected = ExtendedReferenceEvaluator(model).run(
            None, {k: v.numpy() for k, v in feeds.items()}
        )
        got = rt.run(None, feeds)
        self.assertEqualAny(expected, [g.detach().numpy() for g in got])
        self.assertEqual(len(rt.last_used), len(model.graph.node))
        self.assertEqual(len(rt.kernels), len(model.graph.node))
        self.assertEqual([["X"], ["xy"], [], ["xyy"], ["Y", "xyyy"]], rt.last_used)
        for k, v in rt.runtime_info.items():
            if k in {"un", "deux"}:
                self.assertNotEmpty(v.value)
            else:
                self.assertEmpty(v.value)

    def test_op_slice_squeeze(self):
        X = oh.make_tensor_value_info("X", TFLOAT, [None, None])
        starts = oh.make_tensor_value_info("starts", TINT64, [None])
        ends = oh.make_tensor_value_info("ends", TINT64, [None])
        axes = oh.make_tensor_value_info("axes", TINT64, [None])
        Y = oh.make_tensor_value_info("Y", TINT64, [None])
        nodes = [
            oh.make_node("Slice", ["X", "starts", "ends", "axes"], ["T"]),
            oh.make_node("Squeeze", ["T", "axes"], ["Y"]),
        ]
        graph = oh.make_graph(nodes, "g", [X, starts, ends, axes], [Y])
        model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)])
        feeds = {
            "X": torch.tensor([[0]], dtype=torch.int64),
            "starts": torch.tensor([0], dtype=torch.int64),
            "ends": torch.tensor([1], dtype=torch.int64),
            "axes": torch.tensor([0], dtype=torch.int64),
        }
        expected = ExtendedReferenceEvaluator(model).run(
            None, {k: v.numpy() for k, v in feeds.items()}
        )
        rt = TorchEvaluator(model)
        got = rt.run(None, feeds)
        self.assertEqualAny(expected, [g.detach().numpy() for g in got])

    def test_op_shape(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Shape", ["X"], ["shape1"]),
                    oh.make_node("Shape", ["X"], ["shape2"], end=-1),
                ],
                "dummy",
                [oh.make_tensor_value_info("X", TFLOAT, ["a", "b"])],
                [
                    oh.make_tensor_value_info("shape1", TINT64, ["c"]),
                    oh.make_tensor_value_info("shape2", TINT64, ["d"]),
                ],
            ),
            ir_version=9,
            opset_imports=[oh.make_opsetid("", 18)],
        )
        onnx.checker.check_model(model)
        feeds = dict(X=torch.rand((4, 5), dtype=torch.float32))

        expected = ExtendedReferenceEvaluator(model).run(
            None, {k: v.numpy() for k, v in feeds.items()}
        )
        rt = TorchEvaluator(model)
        got = rt.run(None, feeds)
        self.assertEqualAny(expected, [g.detach().numpy() for g in got])

    def test_op_cast(self):
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("Cast", ["X"], ["Y"], to=TINT64)],
                "dummy",
                [oh.make_tensor_value_info("X", TFLOAT, ["a", "b"])],
                [oh.make_tensor_value_info("Y", TINT64, ["a", "b"])],
            ),
            ir_version=9,
            opset_imports=[oh.make_opsetid("", 18)],
        )
        self._finalize_test(model, torch.rand((4, 5, 6, 7), dtype=torch.float32))

    def test_op_transpose(self):
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("Transpose", ["X"], ["Y"], perm=[3, 0, 2, 1])],
                "dummy",
                [oh.make_tensor_value_info("X", TFLOAT, ["a", "b", "c", "d"])],
                [oh.make_tensor_value_info("Y", TFLOAT, ["d", "a", "c", "b"])],
            ),
            ir_version=9,
            opset_imports=[oh.make_opsetid("", 18)],
        )
        self._finalize_test(model, torch.rand((4, 5, 6, 7), dtype=torch.float32))

    def test_op_reshape(self):
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("Reshape", ["X", "shape"], ["Y"])],
                "dummy",
                [
                    oh.make_tensor_value_info("X", TFLOAT, ["a", "b", "c", "d"]),
                    oh.make_tensor_value_info("shape", TINT64, ["f"]),
                ],
                [oh.make_tensor_value_info("Y", TFLOAT, ["d", "a", "c", "b"])],
            ),
            ir_version=9,
            opset_imports=[oh.make_opsetid("", 18)],
        )
        self._finalize_test(
            model, torch.rand((4, 5, 6, 7), dtype=torch.float32), torch.tensor([7, 4, 6, 5])
        )

    def test_op_matmul(self):
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("MatMul", ["X", "Y"], ["Z"])],
                "dummy",
                [
                    oh.make_tensor_value_info("X", TFLOAT, ["a", "b", "c", "d"]),
                    oh.make_tensor_value_info("Y", TFLOAT, ["a", "b", "d", "f"]),
                ],
                [oh.make_tensor_value_info("Z", TFLOAT, ["a", "b", "c", "f"])],
            ),
            ir_version=9,
            opset_imports=[oh.make_opsetid("", 18)],
        )
        self._finalize_test(
            model,
            torch.rand((4, 5, 6, 7), dtype=torch.float32),
            torch.rand((4, 5, 7, 11), dtype=torch.float32),
        )

    def test_op_unsqueeze(self):
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("Unsqueeze", ["X", "axes"], ["Z"])],
                "dummy",
                [
                    oh.make_tensor_value_info("X", TFLOAT, ["a", "b", 1, "d"]),
                    oh.make_tensor_value_info("axes", TINT64, ["s"]),
                ],
                [oh.make_tensor_value_info("Z", TFLOAT, ["a", "b", "d"])],
            ),
            ir_version=9,
            opset_imports=[oh.make_opsetid("", 18)],
        )
        self._finalize_test(
            model,
            torch.rand((4, 5, 1, 7), dtype=torch.float32),
            torch.tensor([2], dtype=torch.int64),
        )

    def test_op_concat(self):
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("Concat", ["X", "Y"], ["Z"], axis=2)],
                "dummy",
                [
                    oh.make_tensor_value_info("X", TFLOAT, ["a", "b", 1, "d"]),
                    oh.make_tensor_value_info("Y", TFLOAT, ["a", "b", 1, "d"]),
                ],
                [oh.make_tensor_value_info("Z", TFLOAT, ["a", "b", "d"])],
            ),
            ir_version=9,
            opset_imports=[oh.make_opsetid("", 18)],
        )
        self._finalize_test(
            model,
            torch.rand((4, 5, 1, 7), dtype=torch.float32),
            torch.rand((4, 5, 2, 7), dtype=torch.float32),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
