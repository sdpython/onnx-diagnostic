import unittest
from typing import Any, Dict, List
import numpy as np
import onnx.helper as oh
import onnx.numpy_helper as onh
from onnx import TensorProto, FunctionProto, ValueInfoProto
from onnx.checker import check_model
import torch
from onnx_diagnostic.ext_test_case import ExtTestCase, hide_stdout
from onnx_diagnostic.helpers.onnx_helper import (
    onnx_lighten,
    onnx_unlighten,
    onnx_find,
    _validate_function,
    check_model_ort,
    iterator_initializer_constant,
    from_array_extended,
    tensor_statistics,
)


TFLOAT = TensorProto.FLOAT


class TestOnnxHelper(ExtTestCase):

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

    def test_iterate_init(self):
        itype = TensorProto.FLOAT
        cst = np.arange(6).astype(np.float32)
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("IsNaN", ["x"], ["xi"]),
                    oh.make_node("IsNaN", ["y"], ["yi"]),
                    oh.make_node("Cast", ["xi"], ["xii"], to=TensorProto.INT64),
                    oh.make_node("Cast", ["yi"], ["yii"], to=TensorProto.INT64),
                    oh.make_node("Add", ["xii", "yii"], ["gggg"]),
                    oh.make_node("Cast", ["gggg"], ["final"], to=itype),
                ],
                "dummy",
                [oh.make_tensor_value_info("x", itype, [None, None])],
                [oh.make_tensor_value_info("final", itype, [None, None])],
                [from_array_extended(cst, name="y")],
            ),
            opset_imports=[oh.make_opsetid("", 20)],
            ir_version=10,
        )
        li = list(iterator_initializer_constant(model))
        self.assertEqual(len(li), 1)
        self.assertEqual(li[0][0], "y")
        self.assertEqualArray(li[0][1], cst)
        li = list(iterator_initializer_constant(model, use_numpy=False))
        self.assertEqual(len(li), 1)
        self.assertEqual(li[0][0], "y")
        self.assertEqualArray(li[0][1], cst)
        self.assertIsInstance(li[0][1], torch.Tensor)

    def _get_cdist_implementation(
        self,
        node_inputs: List[str],
        node_outputs: List[str],
        opsets: Dict[str, int],
        **kwargs: Any,
    ) -> FunctionProto:
        """
        Returns the CDist implementation as a function.
        """
        assert len(node_inputs) == 2
        assert len(node_outputs) == 1
        assert opsets
        assert "" in opsets
        assert set(kwargs) == {"metric"}, f"kwargs={kwargs}"
        metric = kwargs["metric"]
        assert metric in ("euclidean", "sqeuclidean")
        # subgraph
        nodes = [
            oh.make_node("Sub", ["next", "next_in"], ["diff"]),
            oh.make_node("Constant", [], ["axis"], value_ints=[1]),
            oh.make_node("ReduceSumSquare", ["diff", "axis"], ["scan_out"], keepdims=0),
            oh.make_node("Identity", ["next_in"], ["next_out"]),
        ]

        def make_value(name):
            value = ValueInfoProto()
            value.name = name
            return value

        graph = oh.make_graph(
            nodes,
            "loop",
            [make_value("next_in"), make_value("next")],
            [make_value("next_out"), make_value("scan_out")],
        )

        scan = oh.make_node(
            "Scan", ["xb", "xa"], ["next_out", "zout"], num_scan_inputs=1, body=graph
        )
        final = (
            oh.make_node("Sqrt", ["zout"], ["z"])
            if metric == "euclidean"
            else oh.make_node("Identity", ["zout"], ["z"])
        )
        return oh.make_function(
            "npx",
            f"CDist_{metric}",
            ["xa", "xb"],
            ["z"],
            [scan, final],
            [oh.make_opsetid("", opsets[""])],
        )

    def test_iterate_function(self):
        itype = TensorProto.FLOAT
        proto = self._get_cdist_implementation(
            ["X", "Y"], ["Z"], opsets={"": 18}, metric="euclidean"
        )
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node(proto.name, ["X", "Y"], ["Z"]),
                ],
                "dummy",
                [
                    oh.make_tensor_value_info("X", itype, [None, None]),
                    oh.make_tensor_value_info("Y", itype, [None, None]),
                ],
                [oh.make_tensor_value_info("final", itype, [None, None])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )
        model.functions.append(proto)
        li = list(iterator_initializer_constant(model))
        self.assertEqual(len(li), 1)
        self.assertEqual(li[0][0], "CDist_euclideanCDist_euclidean.axis")
        self.assertEqualArray(li[0][1], np.array([1], dtype=np.int64))
        li = list(iterator_initializer_constant(model, use_numpy=False))
        self.assertEqual(len(li), 1)
        self.assertEqual(li[0][0], "CDist_euclideanCDist_euclidean.axis")
        self.assertEqualArray(li[0][1], np.array([1], dtype=np.int64))
        self.assertIsInstance(li[0][1], torch.Tensor)

    def test_statistics(self):
        rnd = np.random.rand(40, 50).astype(np.float16)
        stat = tensor_statistics(rnd)
        self.assertEqual(stat["stype"], "FLOAT16")
        rnd = np.random.rand(40, 50).astype(np.float32)
        stat = tensor_statistics(rnd)
        self.assertEqual(stat["stype"], "FLOAT")


if __name__ == "__main__":
    unittest.main(verbosity=2)
