import unittest
import numpy as np
import onnx
import onnx.helper as oh
import torch
from onnx_diagnostic.ext_test_case import ExtTestCase, skipif_ci_windows
from onnx_diagnostic.helpers import string_type, string_sig, pretty_onnx, get_onnx_signature

TFLOAT = onnx.TensorProto.FLOAT


class TestHelpers(ExtTestCase):
    @skipif_ci_windows("numpy does not choose the same default type on windows and linux")
    def test_string_type(self):
        a = np.array([1])
        obj = {"a": a, "b": [5.6], "c": (1,)}
        s = string_type(obj)
        self.assertEqual(s, "dict(a:A7r1,b:#1[float],c:(int,))")

    def test_string_dict(self):
        a = np.array([1], dtype=np.float32)
        obj = {"a": a, "b": {"r": 5.6}, "c": {1}}
        s = string_type(obj)
        self.assertEqual(s, "dict(a:A1r1,b:dict(r:float),c:{int})")

    def test_string_type_array(self):
        a = np.array([1], dtype=np.float32)
        t = torch.tensor([1])
        obj = {"a": a, "b": t}
        s = string_type(obj, with_shape=False)
        self.assertEqual(s, "dict(a:A1r1,b:T7r1)")
        s = string_type(obj, with_shape=True)
        self.assertEqual(s, "dict(a:A1s1,b:T7s1)")

    def test_string_sig_f(self):
        def f(a, b=3, c=4, e=5):
            pass

        ssig = string_sig(f, {"a": 1, "c": 8, "b": 3})
        self.assertEqual(ssig, "f(a=1, c=8)")

    def test_string_sig_cls(self):
        class A:
            def __init__(self, a, b=3, c=4, e=5):
                self.a, self.b, self.c, self.e = a, b, c, e

        ssig = string_sig(A(1, c=8))
        self.assertEqual(ssig, "A(a=1, c=8)")

    def test_pretty_onnx(self):
        proto = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Sigmoid", ["Y"], ["sy"]),
                    oh.make_node("Mul", ["Y", "sy"], ["ysy"]),
                    oh.make_node("Mul", ["X", "ysy"], ["final"]),
                ],
                "nd",
                [
                    oh.make_tensor_value_info("X", TFLOAT, [1, "b", "c"]),
                    oh.make_tensor_value_info("Y", TFLOAT, ["a", "b", "c"]),
                ],
                [oh.make_tensor_value_info("final", TFLOAT, ["a", "b", "c"])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        pretty_onnx(proto, shape_inference=True)
        pretty_onnx(proto.graph.input[0])
        pretty_onnx(proto.graph)
        pretty_onnx(proto.graph.node[0])

    def test_get_onnx_signature(self):
        proto = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Sigmoid", ["Y"], ["sy"]),
                    oh.make_node("Mul", ["Y", "sy"], ["ysy"]),
                    oh.make_node("Mul", ["X", "ysy"], ["final"]),
                ],
                "nd",
                [
                    oh.make_tensor_value_info("X", TFLOAT, [1, "b", "c"]),
                    oh.make_tensor_value_info("Y", TFLOAT, ["a", "b", "c"]),
                ],
                [oh.make_tensor_value_info("final", TFLOAT, ["a", "b", "c"])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        sig = get_onnx_signature(proto)
        self.assertEqual(sig, (("X", 1, (1, "b", "c")), ("Y", 1, ("a", "b", "c"))))


if __name__ == "__main__":
    unittest.main(verbosity=2)
