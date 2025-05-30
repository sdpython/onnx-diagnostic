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


class TestTorchEvaluator(ExtTestCase):
    def test_kernels(self):
        ker = get_kernels()
        self.assertIsInstance(ker, dict)
        key = "", "Add", 1
        self.assertIn(key, ker)
        kernel = ker[key]
        self.assertEqual("Add_1", kernel.__name__)

    def test_binary_ops(self):
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
