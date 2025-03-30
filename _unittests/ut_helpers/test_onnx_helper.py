import unittest
import numpy as np
import onnx.helper as oh
import onnx.numpy_helper as onh
from onnx import TensorProto
from onnx.checker import check_model
from onnx_diagnostic.ext_test_case import ExtTestCase, hide_stdout
from onnx_diagnostic.helpers.onnx_helper import (
    onnx_lighten,
    onnx_unlighten,
    onnx_find,
    _validate_function,
    check_model_ort,
)


TFLOAT = TensorProto.FLOAT


class TestOnnxTools(ExtTestCase):

    def _get_model(self):
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
                [oh.make_tensor_value_info("X", TFLOAT, [320, 1280])],
                [oh.make_tensor_value_info("Z", TFLOAT, [3, 5, 320, 640])],
                [
                    onh.from_array(
                        np.random.rand(3, 5, 1280, 640).astype(np.float32), name="Y"
                    ),
                    onh.from_array(np.array([0], dtype=np.int64), name="zero"),
                    onh.from_array(np.array([1], dtype=np.int64), name="un"),
                    onh.from_array(np.array([1, 320, 1280], dtype=np.int64), name="shape1"),
                    onh.from_array(np.array([15, 1280, 640], dtype=np.int64), name="shape2"),
                    onh.from_array(np.array([3, 5, 320, 640], dtype=np.int64), name="shape3"),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        return model

    def test_un_lighten_model(self):
        model = self._get_model()
        check_model(model)
        size1 = len(model.SerializeToString())
        (onx, stats), out, _ = self.capture(lambda: onnx_lighten(model, verbose=1))
        self.assertIsInstance(stats, dict)
        self.assertEqual(len(stats), 1)
        self.assertIsInstance(stats["Y"], dict)
        self.assertIn("remove initializer", out)
        # check_model(onx)
        new_model = onnx_unlighten(onx, stats)
        check_model(new_model)
        size2 = len(new_model.SerializeToString())
        self.assertEqual(size1, size2)
        check_model_ort(model)

    def test_onnx_find(self):
        model = self._get_model()
        res = onnx_find(model, watch={"xm2"})
        self.assertEqual(len(res), 2)
        self.assertIn("xm2", res[0].output)
        self.assertIn("xm2", res[1].input)

    @hide_stdout()
    def test__validate_function(self):
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
        _validate_function(linear_regression, verbose=1)

    def test_check_model_ort(self):
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
                [oh.make_tensor_value_info("X", TFLOAT, [320, 1280])],
                [oh.make_tensor_value_info("Z", TFLOAT, [3, 5, 320, 640])],
                [
                    onh.from_array(
                        np.random.rand(3, 5, 1280, 640).astype(np.float32), name="Y"
                    ),
                    onh.from_array(np.array([0], dtype=np.int64), name="zero"),
                    onh.from_array(np.array([1], dtype=np.int64), name="un"),
                    onh.from_array(np.array([1, 320, 1280], dtype=np.int64), name="shape1"),
                    onh.from_array(np.array([15, 1280, 640], dtype=np.int64), name="shape2"),
                    onh.from_array(np.array([3, 5, 320, 640], dtype=np.int64), name="shape3"),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        check_model_ort(model)


if __name__ == "__main__":
    unittest.main(verbosity=2)
