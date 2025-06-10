import unittest
import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh
import torch
from onnx_diagnostic.ext_test_case import ExtTestCase, ignore_warnings
from onnx_diagnostic.helpers.onnx_helper import from_array_extended
from onnx_diagnostic.helpers.torch_helper import onnx_dtype_to_torch_dtype
from onnx_diagnostic.reference import ExtendedReferenceEvaluator, TorchOnnxEvaluator
from onnx_diagnostic.reference.torch_ops import OpRunKernel, OpRunTensor
from onnx_diagnostic.reference.torch_evaluator import get_kernels


TFLOAT = onnx.TensorProto.FLOAT
TFLOAT16 = onnx.TensorProto.FLOAT16
TINT64 = onnx.TensorProto.INT64


class TestTorchOnnxEvaluator(ExtTestCase):
    def test_kernels(self):
        ker = get_kernels()
        self.assertIsInstance(ker, dict)
        key = "", "Add", 1
        self.assertIn(key, ker)
        kernel = ker[key]
        self.assertEqual("Add_1", kernel.__name__)

    def _finalize_test(self, model, *args, atol: float = 0, use_ort: bool = False):
        onnx.checker.check_model(model)
        feeds = dict(zip([i.name for i in model.graph.input], args))
        feeds_numpy = {k: v.numpy() for k, v in feeds.items()}

        if use_ort:
            import onnxruntime

            sess = onnxruntime.InferenceSession(
                model.SerializeToString(), providers=["CPUExecutionProvider"]
            )
            expected = sess.run(None, feeds_numpy)
        else:
            expected = ExtendedReferenceEvaluator(model).run(None, feeds_numpy)
        rt = TorchOnnxEvaluator(model)
        got = rt.run(None, feeds)
        self.assertEqualAny(expected, [g.detach().numpy() for g in got], atol=atol)

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

        rt = TorchOnnxEvaluator(model)
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

    def test_op_binary_cmp(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Neg", ["X"], ["nx"]),
                    oh.make_node("Reciprocal", ["nx"], ["rnx"]),
                    oh.make_node("Equal", ["X", "Y"], ["ae"]),
                    oh.make_node("Greater", ["X", "rnx"], ["a"]),
                    oh.make_node("GreaterOrEqual", ["X", "Y"], ["b"]),
                    oh.make_node("Less", ["X", "Y"], ["c"]),
                    oh.make_node("LessOrEqual", ["X", "Y"], ["d"]),
                    oh.make_node("And", ["ae", "a"], ["aa"]),
                    oh.make_node("And", ["aa", "b"], ["ab"]),
                    oh.make_node("Or", ["c", "d"], ["cd"]),
                    oh.make_node("Not", ["cd"], ["ncd"]),
                    oh.make_node("And", ["ab", "ncd"], ["Z"]),
                ],
                "dummy",
                [
                    oh.make_tensor_value_info("X", TFLOAT, ["a", "b"]),
                    oh.make_tensor_value_info("Y", TFLOAT, ["a", "b"]),
                ],
                [oh.make_tensor_value_info("Z", onnx.TensorProto.BOOL, ["a", "b"])],
            ),
            ir_version=9,
            opset_imports=[oh.make_opsetid("", 18)],
        )
        self._finalize_test(
            model,
            torch.abs(torch.rand(3, 4, 5, dtype=torch.float32)),
            torch.abs(torch.rand(3, 4, 5, dtype=torch.float32)),
        )

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
        rt = TorchOnnxEvaluator(model)
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
        rt = TorchOnnxEvaluator(model)
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
            model,
            torch.rand((4, 5, 6, 7), dtype=torch.float32),
            torch.tensor([7, 4, 6, 5], dtype=torch.int64),
        )

    def test_op_reshape_zero(self):
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
            model,
            torch.rand((4, 5, 6, 7), dtype=torch.float32),
            torch.tensor([7, 4, 0, 5], dtype=torch.int64),
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
            atol=1e-6,
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

    def test_op_gather(self):
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("Gather", ["X", "Y"], ["Z"], axis=1)],
                "dummy",
                [
                    oh.make_tensor_value_info("X", TFLOAT, ["a", "b", "c", "d"]),
                    oh.make_tensor_value_info("Y", TINT64, ["f"]),
                ],
                [oh.make_tensor_value_info("Z", TFLOAT, ["a", "b", "d"])],
            ),
            ir_version=9,
            opset_imports=[oh.make_opsetid("", 18)],
        )
        self._finalize_test(
            model,
            torch.rand((5, 4, 3, 2), dtype=torch.float32),
            torch.tensor([0, 1, 3], dtype=torch.int64),
        )

    def test_op_softmax(self):
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("Softmax", ["X"], ["Z"], axis=0)],
                "dummy",
                [oh.make_tensor_value_info("X", TFLOAT, ["a", "b", "c"])],
                [oh.make_tensor_value_info("Z", TFLOAT, ["a", "b", "c"])],
            ),
            ir_version=9,
            opset_imports=[oh.make_opsetid("", 18)],
        )
        self._finalize_test(
            model, torch.abs(torch.rand(3, 4, 5, dtype=torch.float32)), atol=1e-6
        )

    def test_op_tanh(self):
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("Tanh", ["X"], ["Z"])],
                "dummy",
                [oh.make_tensor_value_info("X", TFLOAT, ["a", "b", "c"])],
                [oh.make_tensor_value_info("Z", TFLOAT, ["a", "b", "c"])],
            ),
            ir_version=9,
            opset_imports=[oh.make_opsetid("", 18)],
        )
        self._finalize_test(
            model, torch.abs(torch.rand(3, 4, 5, dtype=torch.float32)), atol=1e-6
        )

    def test_op_reduce_max(self):
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("ReduceMax", ["X", "axes"], ["Z"])],
                "dummy",
                [
                    oh.make_tensor_value_info("X", TFLOAT, ["a", "b", "c"]),
                    oh.make_tensor_value_info("axes", TINT64, ["f"]),
                ],
                [oh.make_tensor_value_info("Z", TFLOAT, ["a", "b", "c"])],
            ),
            ir_version=9,
            opset_imports=[oh.make_opsetid("", 18)],
        )
        self._finalize_test(
            model,
            torch.rand(3, 4, 5, dtype=torch.float32),
            torch.tensor([1], dtype=torch.int64),
            atol=1e-6,
        )
        self._finalize_test(
            model,
            torch.rand(3, 4, 5, dtype=torch.float32),
            torch.tensor([1, 2], dtype=torch.int64),
            atol=1e-6,
        )

    def test_op_reduce_mean(self):
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("ReduceMean", ["X", "axes"], ["Z"])],
                "dummy",
                [
                    oh.make_tensor_value_info("X", TFLOAT, ["a", "b", "c"]),
                    oh.make_tensor_value_info("axes", TINT64, ["f"]),
                ],
                [oh.make_tensor_value_info("Z", TFLOAT, ["a", "b", "c"])],
            ),
            ir_version=9,
            opset_imports=[oh.make_opsetid("", 18)],
        )
        self._finalize_test(
            model,
            torch.rand(3, 4, 5, dtype=torch.float32),
            torch.tensor([1], dtype=torch.int64),
            atol=1e-6,
        )
        self._finalize_test(
            model,
            torch.rand(3, 4, 5, dtype=torch.float32),
            torch.tensor([1, 2], dtype=torch.int64),
            atol=1e-6,
        )

    def test_op_reduce_min(self):
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("ReduceMin", ["X", "axes"], ["Z"])],
                "dummy",
                [
                    oh.make_tensor_value_info("X", TFLOAT, ["a", "b", "c"]),
                    oh.make_tensor_value_info("axes", TINT64, ["f"]),
                ],
                [oh.make_tensor_value_info("Z", TFLOAT, ["a", "b", "c"])],
            ),
            ir_version=9,
            opset_imports=[oh.make_opsetid("", 18)],
        )
        self._finalize_test(
            model,
            torch.rand(3, 4, 5, dtype=torch.float32),
            torch.tensor([1], dtype=torch.int64),
            atol=1e-6,
        )
        self._finalize_test(
            model,
            torch.rand(3, 4, 5, dtype=torch.float32),
            torch.tensor([1, 2], dtype=torch.int64),
            atol=1e-6,
        )

    def test_op_reduce_min_no_axes(self):
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("ReduceMin", ["X"], ["Z"], keepdims=0)],
                "dummy",
                [oh.make_tensor_value_info("X", TFLOAT, ["a", "b", "c"])],
                [oh.make_tensor_value_info("Z", TFLOAT, ["a", "b", "c"])],
            ),
            ir_version=9,
            opset_imports=[oh.make_opsetid("", 18)],
        )
        self._finalize_test(
            model,
            torch.rand(3, 4, 5, dtype=torch.float32),
            atol=1e-6,
        )

    def test_op_reduce_min_17(self):
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("ReduceMin", ["X"], ["Z"], axes=[1])],
                "dummy",
                [oh.make_tensor_value_info("X", TFLOAT, ["a", "b", "c"])],
                [oh.make_tensor_value_info("Z", TFLOAT, ["a", "b", "c"])],
            ),
            ir_version=9,
            opset_imports=[oh.make_opsetid("", 17)],
        )
        self._finalize_test(
            model,
            torch.rand(3, 4, 5, dtype=torch.float32),
            atol=1e-6,
        )

    def test_op_reduce_min_17_no_axes(self):
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("ReduceMin", ["X"], ["Z"], keepdims=0)],
                "dummy",
                [oh.make_tensor_value_info("X", TFLOAT, ["a", "b", "c"])],
                [oh.make_tensor_value_info("Z", TFLOAT, ["a", "b", "c"])],
            ),
            ir_version=9,
            opset_imports=[oh.make_opsetid("", 17)],
        )
        self._finalize_test(
            model,
            torch.rand(3, 4, 5, dtype=torch.float32),
            atol=1e-6,
        )

    def test_op_reduce_sum(self):
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("ReduceSum", ["X", "axes"], ["Z"])],
                "dummy",
                [
                    oh.make_tensor_value_info("X", TFLOAT, ["a", "b", "c"]),
                    oh.make_tensor_value_info("axes", TINT64, ["f"]),
                ],
                [oh.make_tensor_value_info("Z", TFLOAT, ["a", "b", "c"])],
            ),
            ir_version=9,
            opset_imports=[oh.make_opsetid("", 18)],
        )
        self._finalize_test(
            model,
            torch.rand(3, 4, 5, dtype=torch.float32),
            torch.tensor([1], dtype=torch.int64),
            atol=1e-5,
        )
        self._finalize_test(
            model,
            torch.rand(3, 4, 5, dtype=torch.float32),
            torch.tensor([1, 2], dtype=torch.int64),
            atol=1e-5,
        )

    def test_op_where(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Greater", ["X", "Y"], ["cond"]),
                    oh.make_node("Where", ["cond", "X", "Y"], ["Z"]),
                ],
                "dummy",
                [
                    oh.make_tensor_value_info("X", TFLOAT, ["a", "b", "c"]),
                    oh.make_tensor_value_info("Y", TFLOAT, ["a", "b", "c"]),
                ],
                [oh.make_tensor_value_info("Z", TFLOAT, ["a", "b", "c"])],
            ),
            ir_version=9,
            opset_imports=[oh.make_opsetid("", 18)],
        )
        self._finalize_test(
            model,
            torch.rand(3, 4, 5, dtype=torch.float32),
            torch.rand(3, 4, 5, dtype=torch.float32),
        )

    def test_op_layer_normalization(self):
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("LayerNormalization", ["X", "W", "B"], ["Z"], axis=-1)],
                "dummy",
                [
                    oh.make_tensor_value_info("X", TFLOAT, ["a", "b", "c"]),
                    oh.make_tensor_value_info("W", TFLOAT, []),
                    oh.make_tensor_value_info("B", TFLOAT, []),
                ],
                [oh.make_tensor_value_info("Z", TFLOAT, ["a", "b", "c"])],
            ),
            ir_version=9,
            opset_imports=[oh.make_opsetid("", 18)],
        )
        self._finalize_test(
            model,
            torch.rand(3, 4, 5, dtype=torch.float32),
            torch.abs(torch.rand(5, dtype=torch.float32)),
            torch.rand(5, dtype=torch.float32),
            use_ort=False,
            atol=1e-4,
        )

    def test_op_layer_normalization_axis1(self):
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("LayerNormalization", ["X", "W", "B"], ["Z"], axis=1)],
                "dummy",
                [
                    oh.make_tensor_value_info("X", TFLOAT, ["a", "b", "c"]),
                    oh.make_tensor_value_info("W", TFLOAT, []),
                    oh.make_tensor_value_info("B", TFLOAT, []),
                ],
                [oh.make_tensor_value_info("Z", TFLOAT, ["a", "b", "c"])],
            ),
            ir_version=9,
            opset_imports=[oh.make_opsetid("", 18)],
        )
        self._finalize_test(
            model,
            torch.rand(3, 4, 5, dtype=torch.float32),
            torch.abs(torch.rand(4, 5, dtype=torch.float32)),
            torch.rand(4, 5, dtype=torch.float32),
            use_ort=False,
            atol=1e-4,
        )

    def test_op_layer_normalization_big_eps(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node(
                        "LayerNormalization", ["X", "W", "B"], ["Z"], axis=-1, epsilon=2.0
                    )
                ],
                "dummy",
                [
                    oh.make_tensor_value_info("X", TFLOAT, ["a", "b", "c"]),
                    oh.make_tensor_value_info("W", TFLOAT, []),
                    oh.make_tensor_value_info("B", TFLOAT, []),
                ],
                [oh.make_tensor_value_info("Z", TFLOAT, ["a", "b", "c"])],
            ),
            ir_version=9,
            opset_imports=[oh.make_opsetid("", 18)],
        )
        self._finalize_test(
            model,
            torch.rand(3, 4, 5, dtype=torch.float32),
            torch.abs(torch.rand(5, dtype=torch.float32)),
            torch.rand(5, dtype=torch.float32),
            use_ort=False,
            atol=1e-4,
        )

    def test_op_range_float(self):
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("Range", ["start", "limit", "delta"], ["Z"])],
                "dummy",
                [
                    oh.make_tensor_value_info("start", TFLOAT, []),
                    oh.make_tensor_value_info("limit", TFLOAT, []),
                    oh.make_tensor_value_info("delta", TFLOAT, []),
                ],
                [oh.make_tensor_value_info("Z", TFLOAT, ["a"])],
            ),
            ir_version=9,
            opset_imports=[oh.make_opsetid("", 18)],
        )
        self._finalize_test(
            model,
            torch.tensor(2.1, dtype=torch.float32),
            torch.tensor(5.1, dtype=torch.float32),
            torch.tensor(1, dtype=torch.float32),
        )

    def test_op_range_int64(self):
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("Range", ["start", "limit", "delta"], ["Z"])],
                "dummy",
                [
                    oh.make_tensor_value_info("start", TINT64, []),
                    oh.make_tensor_value_info("limit", TINT64, []),
                    oh.make_tensor_value_info("delta", TINT64, []),
                ],
                [oh.make_tensor_value_info("Z", TINT64, ["a"])],
            ),
            ir_version=9,
            opset_imports=[oh.make_opsetid("", 18)],
        )
        self._finalize_test(
            model,
            torch.tensor(2, dtype=torch.int64),
            torch.tensor(5, dtype=torch.int64),
            torch.tensor(1, dtype=torch.int64),
        )

    def test_op_range_int64_h2(self):
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("Range", ["start", "limit", "delta"], ["Z"])],
                "dummy",
                [
                    oh.make_tensor_value_info("start", TINT64, []),
                    oh.make_tensor_value_info("limit", TINT64, []),
                    oh.make_tensor_value_info("delta", TINT64, []),
                ],
                [oh.make_tensor_value_info("Z", TINT64, ["a"])],
            ),
            ir_version=9,
            opset_imports=[oh.make_opsetid("", 18)],
        )
        self._finalize_test(
            model,
            torch.tensor(2, dtype=torch.int64),
            torch.tensor(5, dtype=torch.int64),
            torch.tensor(2, dtype=torch.int64),
        )

    def test_op_expand(self):
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("Expand", ["X", "shape"], ["Y"])],
                "dummy",
                [
                    oh.make_tensor_value_info("X", TFLOAT, ["a", "b", "c", "d"]),
                    oh.make_tensor_value_info("shape", TINT64, ["f"]),
                ],
                [oh.make_tensor_value_info("Y", TFLOAT, ["aa", "ba", "ca", "da"])],
            ),
            ir_version=9,
            opset_imports=[oh.make_opsetid("", 18)],
        )
        self._finalize_test(
            model,
            torch.rand((1, 5, 6, 7), dtype=torch.float32),
            torch.tensor([4, 5, 1, 1], dtype=torch.int64),
        )

    def test_op_unary(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Cos", ["X"], ["nx"]),
                    oh.make_node("Sin", ["nx"], ["t"]),
                    oh.make_node("Exp", ["t"], ["u"]),
                    oh.make_node("Log", ["u"], ["uZ"]),
                    oh.make_node("Erf", ["uZ"], ["Z"]),
                ],
                "dummy",
                [oh.make_tensor_value_info("X", TFLOAT, ["a", "b"])],
                [oh.make_tensor_value_info("Z", TFLOAT, ["a", "b"])],
            ),
            ir_version=9,
            opset_imports=[oh.make_opsetid("", 18)],
        )
        onnx.checker.check_model(model)
        self._finalize_test(model, torch.abs(torch.rand(3, 4, dtype=torch.float32)), atol=1e-6)

    def test_op_pow(self):
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("Pow", ["X", "Y"], ["Z"])],
                "dummy",
                [
                    oh.make_tensor_value_info("X", TFLOAT, ["a", "b"]),
                    oh.make_tensor_value_info("Y", TFLOAT, ["a", "b"]),
                ],
                [oh.make_tensor_value_info("Z", TFLOAT, ["a", "b"])],
            ),
            ir_version=9,
            opset_imports=[oh.make_opsetid("", 18)],
        )
        onnx.checker.check_model(model)
        self._finalize_test(
            model,
            torch.abs(torch.rand(3, 4, 5, dtype=torch.float32)),
            torch.abs(torch.rand(3, 4, 5, dtype=torch.float32)),
            atol=1e-7,
        )

    def test_op_pow_op_int(self):
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("Pow", ["X", "Y"], ["Z"])],
                "dummy",
                [
                    oh.make_tensor_value_info("X", TFLOAT, ["a", "b"]),
                    oh.make_tensor_value_info("Y", TINT64, ["a", "b"]),
                ],
                [oh.make_tensor_value_info("Z", TFLOAT, ["a", "b"])],
            ),
            ir_version=9,
            opset_imports=[oh.make_opsetid("", 18)],
        )
        onnx.checker.check_model(model)
        self._finalize_test(
            model,
            torch.rand(3, 4, 5, dtype=torch.float32),
            torch.tensor([2], dtype=torch.int64),
            atol=1e-7,
        )

    def test_op_sqrt(self):
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("Sqrt", ["X"], ["Z"])],
                "dummy",
                [oh.make_tensor_value_info("X", TFLOAT, ["a", "b"])],
                [oh.make_tensor_value_info("Z", TFLOAT, ["a", "b"])],
            ),
            ir_version=9,
            opset_imports=[oh.make_opsetid("", 18)],
        )
        onnx.checker.check_model(model)
        self._finalize_test(model, torch.abs(torch.rand(3, 4, dtype=torch.float32)), atol=1e-6)

    def test_op_sigmoid(self):
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("Sigmoid", ["X"], ["Z"])],
                "dummy",
                [oh.make_tensor_value_info("X", TFLOAT, ["a", "b"])],
                [oh.make_tensor_value_info("Z", TFLOAT, ["a", "b"])],
            ),
            ir_version=9,
            opset_imports=[oh.make_opsetid("", 18)],
        )
        onnx.checker.check_model(model)
        self._finalize_test(model, torch.abs(torch.rand(3, 4, dtype=torch.float32)), atol=1e-6)

    def test_op_split(self):
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("Split", ["X"], ["Z1", "Z2"], axis=1, num_outputs=2)],
                "dummy",
                [oh.make_tensor_value_info("X", TFLOAT, ["a", "b"])],
                [
                    oh.make_tensor_value_info("Z1", TFLOAT, ["a", "b1"]),
                    oh.make_tensor_value_info("Z2", TFLOAT, ["a", "b2"]),
                ],
            ),
            ir_version=9,
            opset_imports=[oh.make_opsetid("", 18)],
        )
        onnx.checker.check_model(model)
        self._finalize_test(model, torch.rand(3, 5, dtype=torch.float32), use_ort=True)
        self._finalize_test(model, torch.rand(3, 6, dtype=torch.float32), use_ort=True)

    def test_op_split_op_sizes(self):
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("Split", ["X", "split"], ["Z1", "Z2"], axis=1)],
                "dummy",
                [
                    oh.make_tensor_value_info("X", TFLOAT, ["a", "b"]),
                    oh.make_tensor_value_info("split", TINT64, [2]),
                ],
                [
                    oh.make_tensor_value_info("Z1", TFLOAT, ["a", "b1"]),
                    oh.make_tensor_value_info("Z2", TFLOAT, ["a", "b2"]),
                ],
            ),
            ir_version=9,
            opset_imports=[oh.make_opsetid("", 18)],
        )
        onnx.checker.check_model(model)
        self._finalize_test(
            model,
            torch.rand(3, 5, dtype=torch.float32),
            torch.tensor([2, 3], dtype=torch.int64),
            use_ort=True,
        )

    def test_op_constant_of_shape(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node(
                        "ConstantOfShape",
                        ["shape"],
                        ["Z"],
                        value=from_array_extended(np.array([2], dtype=np.float16)),
                    )
                ],
                "dummy",
                [oh.make_tensor_value_info("shape", TINT64, ["a"])],
                [oh.make_tensor_value_info("Z", onnx.TensorProto.FLOAT16, ["a", "b"])],
            ),
            ir_version=9,
            opset_imports=[oh.make_opsetid("", 18)],
        )
        onnx.checker.check_model(model)
        self._finalize_test(model, torch.tensor([4, 5], dtype=torch.int64))

    def test_op_trilu(self):
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("Trilu", ["X"], ["Z"])],
                "dummy",
                [oh.make_tensor_value_info("X", TFLOAT, ["a", "b"])],
                [oh.make_tensor_value_info("Z", TFLOAT, ["a", "b"])],
            ),
            ir_version=9,
            opset_imports=[oh.make_opsetid("", 18)],
        )
        onnx.checker.check_model(model)
        self._finalize_test(model, torch.rand((4, 4), dtype=torch.float32))

    def test_op_trilu_1(self):
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("Trilu", ["X"], ["Z"], upper=0)],
                "dummy",
                [oh.make_tensor_value_info("X", TFLOAT, ["a", "b"])],
                [oh.make_tensor_value_info("Z", TFLOAT, ["a", "b"])],
            ),
            ir_version=9,
            opset_imports=[oh.make_opsetid("", 18)],
        )
        onnx.checker.check_model(model)
        self._finalize_test(model, torch.rand((4, 4), dtype=torch.float32))

    @ignore_warnings(DeprecationWarning)
    def test_op_trilu_k(self):
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("Trilu", ["X", "k"], ["Z"], upper=1)],
                "dummy",
                [
                    oh.make_tensor_value_info("X", TFLOAT, ["a", "b"]),
                    oh.make_tensor_value_info("k", TINT64, []),
                ],
                [oh.make_tensor_value_info("Z", TFLOAT, ["a", "b"])],
            ),
            ir_version=9,
            opset_imports=[oh.make_opsetid("", 18)],
        )
        onnx.checker.check_model(model)
        self._finalize_test(
            model,
            torch.rand((6, 6), dtype=torch.float32),
            torch.tensor([2], dtype=torch.int64),
        )

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
            [oh.make_opsetid("", 18)],
            [],
        )

        graph = oh.make_graph(
            [
                oh.make_node("LinearRegression", ["X", "A", "B"], ["Y1"], domain=new_domain),
                oh.make_node("Abs", ["Y1"], ["Y"]),
            ],
            "example",
            [
                oh.make_tensor_value_info("X", TFLOAT, ["a", "b"]),
                oh.make_tensor_value_info("A", TFLOAT, ["a", "b"]),
                oh.make_tensor_value_info("B", TFLOAT, ["a", "b"]),
            ],
            [oh.make_tensor_value_info("Y", TFLOAT, ["a", "b"])],
        )

        model = oh.make_model(
            graph,
            opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid(new_domain, 1)],
            functions=[linear_regression],
            ir_version=10,
        )
        self.assertNotEmpty(model.functions)
        self._finalize_test(
            model,
            torch.rand((3, 3), dtype=torch.float32),
            torch.rand((3, 3), dtype=torch.float32),
            torch.rand((3, 3), dtype=torch.float32),
            atol=1e-6,
        )

    def test_if(self):
        def _mkv_(name):
            value_info_proto = onnx.ValueInfoProto()
            value_info_proto.name = name
            return value_info_proto

        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("ReduceSum", ["X"], ["Xred"], keepdims=0),
                    oh.make_node("Add", ["X", "two"], ["X0"]),
                    oh.make_node("Add", ["X0", "zero"], ["X00"]),
                    oh.make_node("CastLike", ["one", "Xred"], ["one_c"]),
                    oh.make_node("Greater", ["Xred", "one_c"], ["cond"]),
                    oh.make_node(
                        "If",
                        ["cond"],
                        ["Z_c"],
                        then_branch=oh.make_graph(
                            [
                                oh.make_node("Constant", [], ["t2"], value_floats=[2.1]),
                                oh.make_node("Add", ["X00", "t2"], ["Y"]),
                            ],
                            "then",
                            [],
                            [_mkv_("Y")],
                        ),
                        else_branch=oh.make_graph(
                            [
                                oh.make_node("Constant", [], ["t2"], value_floats=[2.2]),
                                oh.make_node("Sub", ["X0", "t2"], ["Y"]),
                            ],
                            "else",
                            [],
                            [_mkv_("Y")],
                        ),
                    ),
                    oh.make_node("CastLike", ["Z_c", "X"], ["Z"]),
                ],
                "test",
                [
                    oh.make_tensor_value_info("X", TFLOAT, ["N"]),
                    oh.make_tensor_value_info("one", TFLOAT, ["N"]),
                ],
                [oh.make_tensor_value_info("Z", onnx.TensorProto.UNDEFINED, ["N"])],
                [
                    onh.from_array(np.array([0], dtype=np.float32), name="zero"),
                    onh.from_array(np.array([2], dtype=np.float32), name="two"),
                ],
            ),
            opset_imports=[oh.make_operatorsetid("", 18)],
            ir_version=10,
        )
        self._finalize_test(
            model,
            torch.tensor([1, 2, 3], dtype=torch.float32),
            torch.tensor([1], dtype=torch.float32),
        )

    def test_loop(self):
        x = np.array([1, 2, 3, 4, 5]).astype(np.float32)

        model = oh.make_model(
            graph=oh.make_graph(
                name="loop_test",
                inputs=[
                    oh.make_tensor_value_info("trip_count", TINT64, ["a"]),
                    oh.make_tensor_value_info("cond", onnx.TensorProto.BOOL, [1]),
                ],
                outputs=[oh.make_tensor_value_info("res", TFLOAT, [])],
                nodes=[
                    oh.make_node("SequenceEmpty", [], ["seq_empty"], dtype=TFLOAT),
                    oh.make_node(
                        "Loop",
                        inputs=["trip_count", "cond", "seq_empty"],
                        outputs=["seq_res"],
                        body=oh.make_graph(
                            [
                                oh.make_node(
                                    "Identity", inputs=["cond_in"], outputs=["cond_out"]
                                ),
                                oh.make_node(
                                    "Constant",
                                    inputs=[],
                                    outputs=["x"],
                                    value=oh.make_tensor(
                                        name="const_tensor_x",
                                        data_type=TFLOAT,
                                        dims=x.shape,
                                        vals=x.flatten().astype(float),
                                    ),
                                ),
                                oh.make_node(
                                    "Constant",
                                    inputs=[],
                                    outputs=["one"],
                                    value=oh.make_tensor(
                                        name="const_tensor_one",
                                        data_type=TINT64,
                                        dims=(),
                                        vals=[1],
                                    ),
                                ),
                                oh.make_node(
                                    "Constant",
                                    inputs=[],
                                    outputs=["slice_start"],
                                    value=oh.make_tensor(
                                        name="const_tensor_zero",
                                        data_type=TINT64,
                                        dims=(1,),
                                        vals=[0],
                                    ),
                                ),
                                oh.make_node(
                                    "Add", inputs=["iter_count", "one"], outputs=["end"]
                                ),
                                oh.make_node(
                                    "Constant",
                                    inputs=[],
                                    outputs=["axes"],
                                    value=oh.make_tensor(
                                        name="const_tensor_axes",
                                        data_type=TINT64,
                                        dims=(1,),
                                        vals=[0],
                                    ),
                                ),
                                oh.make_node(
                                    "Unsqueeze", inputs=["end", "axes"], outputs=["slice_end"]
                                ),
                                oh.make_node(
                                    "Slice",
                                    inputs=["x", "slice_start", "slice_end"],
                                    outputs=["slice_out"],
                                ),
                                oh.make_node(
                                    "SequenceInsert",
                                    inputs=["seq_in", "slice_out"],
                                    outputs=["seq_out"],
                                ),
                            ],
                            "loop_body",
                            [
                                oh.make_tensor_value_info("iter_count", TINT64, []),
                                oh.make_tensor_value_info(
                                    "cond_in", onnx.TensorProto.BOOL, []
                                ),
                                oh.make_tensor_sequence_value_info("seq_in", TFLOAT, None),
                            ],
                            [
                                oh.make_tensor_value_info(
                                    "cond_out", onnx.TensorProto.BOOL, []
                                ),
                                oh.make_tensor_sequence_value_info("seq_out", TFLOAT, None),
                            ],
                        ),
                    ),
                    oh.make_node(
                        "ConcatFromSequence",
                        inputs=["seq_res"],
                        outputs=["res"],
                        axis=0,
                        new_axis=0,
                    ),
                ],
            )
        )
        self._finalize_test(
            model, torch.tensor(5, dtype=torch.int64), torch.tensor(1, dtype=torch.bool)
        )

    def test_conv(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node(
                        "Conv",
                        ["X", "W", "B"],
                        ["Y"],
                        pads=[1, 1, 1, 1],
                        dilations=[1, 1],
                        strides=[2, 2],
                    )
                ],
                "g",
                [
                    oh.make_tensor_value_info("X", TFLOAT, [None, None, None, None]),
                    oh.make_tensor_value_info("W", TFLOAT, [None, None, None, None]),
                    oh.make_tensor_value_info("B", TFLOAT, [None, None, None, None]),
                ],
                [oh.make_tensor_value_info("Y", TFLOAT, [None, None, None, None])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )
        sH, sW = 5, 6
        i = sH // 2
        j = sW // 2
        X = torch.zeros((1, 1, sH, sW), dtype=torch.float32)
        X[0, 0, i, j] = 1.0
        W = torch.zeros((1, 1, 3, 3), dtype=torch.float32)
        W[0, 0, :, :] = torch.minimum(
            2 ** torch.arange(9).reshape((3, -1)), torch.tensor([256])
        )
        B = torch.tensor([[[[0]]]], dtype=torch.float32)
        self._finalize_test(model, X, W, B, use_ort=True)

    def test_conv_autopad_valid(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node(
                        "Conv",
                        ["X", "W", "B"],
                        ["Y"],
                        dilations=[1, 1],
                        strides=[2, 2],
                        auto_pad="VALID",
                    )
                ],
                "g",
                [
                    oh.make_tensor_value_info("X", TFLOAT, [None, None, None, None]),
                    oh.make_tensor_value_info("W", TFLOAT, [None, None, None, None]),
                    oh.make_tensor_value_info("B", TFLOAT, [None, None, None, None]),
                ],
                [oh.make_tensor_value_info("Y", TFLOAT, [None, None, None, None])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )
        sH, sW = 5, 5
        i = sH // 2
        j = sW // 2
        X = torch.zeros((1, 1, sH, sW), dtype=torch.float32)
        X[0, 0, i, j] = 1.0
        W = torch.zeros((1, 1, 3, 3), dtype=torch.float32)
        W[0, 0, :, :] = torch.minimum(
            2 ** torch.arange(9).reshape((3, -1)), torch.tensor([256])
        )
        B = torch.tensor([[[[0]]]], dtype=torch.float32)
        self._finalize_test(model, X, W, B, use_ort=True)

    def test_conv_autopad_upper(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node(
                        "Conv",
                        ["X", "W", "B"],
                        ["Y"],
                        dilations=[1, 1],
                        strides=[2, 2],
                        auto_pad="SAME_UPPER",
                    )
                ],
                "g",
                [
                    oh.make_tensor_value_info("X", TFLOAT, [None, None, None, None]),
                    oh.make_tensor_value_info("W", TFLOAT, [None, None, None, None]),
                    oh.make_tensor_value_info("B", TFLOAT, [None, None, None, None]),
                ],
                [oh.make_tensor_value_info("Y", TFLOAT, [None, None, None, None])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )
        sH, sW = 5, 5
        i = sH // 2
        j = sW // 2
        X = torch.zeros((1, 1, sH, sW), dtype=torch.float32)
        X[0, 0, i, j] = 1.0
        W = torch.zeros((1, 1, 3, 3), dtype=torch.float32)
        W[0, 0, :, :] = torch.minimum(
            2 ** torch.arange(9).reshape((3, -1)), torch.tensor([256])
        )
        B = torch.tensor([[[[0]]]], dtype=torch.float32)
        self._finalize_test(model, X, W, B, use_ort=True)

    def test_nonzero(self):
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("NonZero", ["X"], ["Y"])],
                "g",
                [oh.make_tensor_value_info("X", TFLOAT, [None, None])],
                [oh.make_tensor_value_info("Y", TINT64, [None, None])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )

        self._finalize_test(
            model, torch.tensor([[1, 0], [1, 1]], dtype=torch.float32), use_ort=True
        )

    def test_scatternd_2d(self):
        for reduction in ["none", "add", "min", "max", "mul"]:
            with self.subTest(reduction=reduction):
                model = oh.make_model(
                    oh.make_graph(
                        [
                            oh.make_node(
                                "ScatterND",
                                ["data", "indices", "updates"],
                                ["Y"],
                                reduction=reduction,
                            )
                        ],
                        "g",
                        [
                            oh.make_tensor_value_info("data", TFLOAT, [None, None, None]),
                            oh.make_tensor_value_info("indices", TINT64, [None, None]),
                            oh.make_tensor_value_info("updates", TFLOAT, [None, None, None]),
                        ],
                        [oh.make_tensor_value_info("Y", TFLOAT, [None, None, None])],
                    ),
                    opset_imports=[oh.make_opsetid("", 18)],
                    ir_version=10,
                )

                self._finalize_test(
                    model,
                    torch.tensor(
                        [
                            [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
                            [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
                            [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]],
                            [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]],
                        ],
                        dtype=torch.float32,
                    ),
                    torch.tensor([[0], [0]], dtype=torch.int64),
                    torch.tensor(
                        [
                            [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
                            [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]],
                        ],
                        dtype=torch.float32,
                    ),
                    use_ort=True,
                )

    def test_averagepool_1d(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node(
                        "AveragePool",
                        inputs=["x"],
                        outputs=["y"],
                        kernel_shape=[2],
                    )
                ],
                "ut",
                [oh.make_tensor_value_info("x", TFLOAT, [None, None, None])],
                [oh.make_tensor_value_info("y", TFLOAT, [None, None, None])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )
        self._finalize_test(model, torch.rand((1, 3, 32), dtype=torch.float32))

    def test_averagepool_2d(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node(
                        "AveragePool",
                        inputs=["x"],
                        outputs=["y"],
                        kernel_shape=[5, 5],
                        pads=[2, 2, 2, 2],
                    )
                ],
                "ut",
                [oh.make_tensor_value_info("x", TFLOAT, [None, None, None, None])],
                [oh.make_tensor_value_info("y", TFLOAT, [None, None, None, None])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )
        self._finalize_test(
            model,
            torch.tensor(
                [
                    [
                        [
                            [1, 2, 3, 4, 5],
                            [6, 7, 8, 9, 10],
                            [11, 12, 13, 14, 15],
                            [16, 17, 18, 19, 20],
                            [21, 22, 23, 24, 25],
                        ]
                    ]
                ],
                dtype=torch.float32,
            ),
        )

    def test_tile(self):
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("Tile", ["x", "repeat"], ["y"])],
                "ut",
                [
                    oh.make_tensor_value_info("x", TFLOAT, [None, None]),
                    oh.make_tensor_value_info("repeat", TFLOAT, [None]),
                ],
                [oh.make_tensor_value_info("y", TFLOAT, [None, None, None, None])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )
        self._finalize_test(
            model,
            torch.tensor([[1, 2], [3, 4]], dtype=torch.float32),
            torch.tensor([2, 2], dtype=torch.int64),
        )

    def test_custom_kernels(self):
        class LayerNormalizationOrt(OpRunKernel):
            "LayerNormalization"

            _shared = [0]

            def __init__(self, node: onnx.NodeProto, version=None, verbose=0):
                super().__init__(node, version, verbose=verbose)
                self.axis = self.get_attribute_int(node, "axis", -1)
                self.epsilon = self.get_attribute_float(node, "epsilon", 1e-5)
                self.stash_type = onnx_dtype_to_torch_dtype(
                    self.get_attribute_int(node, "stash_type", onnx.TensorProto.FLOAT)
                )
                self.compute_std = len(node.output) > 1
                assert not self.compute_std
                layer_model = oh.make_model(
                    oh.make_graph(
                        [
                            oh.make_node(
                                "LayerNormalization",
                                ["X", "W", "B"],
                                ["Z"],
                                axis=-1,
                                epsilon=9.999999974752427e-7,
                            )
                        ],
                        "dummy",
                        [
                            oh.make_tensor_value_info("X", TFLOAT16, ["b", "c", "d"]),
                            oh.make_tensor_value_info("W", TFLOAT16, ["d"]),
                            oh.make_tensor_value_info("B", TFLOAT16, ["d"]),
                        ],
                        [oh.make_tensor_value_info("Z", TFLOAT16, ["b", "c", "d"])],
                    ),
                    ir_version=9,
                    opset_imports=[oh.make_opsetid("", 17)],
                )
                import onnxruntime

                self.ort_sess = onnxruntime.InferenceSession(
                    layer_model.SerializeToString(), providers=["CUDAExecutionProvider"]
                )

            def run(self, x, scale, bias=None):
                self._shared[0] += 1
                feeds = dict(X=x, W=scale)
                if bias is not None:
                    feeds["B"] = bias
                feeds = {k: v.tensor.detach().cpu().numpy() for k, v in feeds.items()}
                got = self.ort_sess.run(None, feeds)[0]
                return OpRunTensor(torch.from_numpy(got).to(x.dtype).to(x.device))

        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node(
                        "LayerNormalization",
                        ["X", "W", "B"],
                        ["ln"],
                        axis=-1,
                        epsilon=9.999999974752427e-7,
                    ),
                    oh.make_node(
                        "Add", ["ln", "W"], ["Z"], axis=-1, epsilon=9.999999974752427e-7
                    ),
                ],
                "dummy",
                [
                    oh.make_tensor_value_info("X", TFLOAT16, ["b", "c", "d"]),
                    oh.make_tensor_value_info("W", TFLOAT16, ["d"]),
                    oh.make_tensor_value_info("B", TFLOAT16, ["d"]),
                ],
                [oh.make_tensor_value_info("Z", TFLOAT16, ["b", "c", "d"])],
            ),
            ir_version=9,
            opset_imports=[oh.make_opsetid("", 17)],
        )

        torch_sess = TorchOnnxEvaluator(model, verbose=0)
        torch_sess_custom = TorchOnnxEvaluator(
            model,
            verbose=0,
            custom_kernels={("", "LayerNormalization"): LayerNormalizationOrt},
        )
        feeds = dict(
            zip(
                torch_sess.input_names,
                [
                    torch.rand(3, 4, 5, dtype=torch.float16),
                    torch.abs(torch.rand(5, dtype=torch.float16)),
                    torch.rand(5, dtype=torch.float16),
                ],
            )
        )
        expected = torch_sess.run(None, feeds)
        got = torch_sess_custom.run(None, feeds)
        self.assertEqualAny(expected, got, atol=1e-3)
        self.assertEqual([1], LayerNormalizationOrt._shared)


if __name__ == "__main__":
    unittest.main(verbosity=2)
