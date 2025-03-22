import unittest
from typing import Optional
import numpy as np
import onnx
import onnx.helper as oh
from onnx import TensorProto
from onnx.helper import (
    make_node,
    make_tensor_value_info,
    make_opsetid,
)
from onnx_diagnostic.helpers import from_array_extended
from onnx_diagnostic.ext_test_case import ExtTestCase, has_cuda
from onnx_diagnostic.reference import ExtendedReferenceEvaluator


class TestReferenceOps(ExtTestCase):
    def _range(self, *shape, bias: Optional[float] = None):
        n = np.prod(shape)
        x = np.arange(n).astype(np.float32) / n
        if bias:
            x = x + bias
        return x.reshape(tuple(shape)).astype(np.float32)

    def test_fused_matmul(self):
        model = oh.make_model(
            oh.make_graph(
                [make_node("FusedMatMul", ["X", "Y"], ["Z"], domain="com.microsoft")],
                "name",
                [
                    oh.make_tensor_value_info("X", TensorProto.FLOAT, None),
                    oh.make_tensor_value_info("Y", TensorProto.FLOAT, None),
                ],
                [make_tensor_value_info("Z", TensorProto.FLOAT, None)],
            ),
            opset_imports=[make_opsetid("", 18), oh.make_opsetid("com.microsoft", 1)],
        )
        ref = ExtendedReferenceEvaluator(model)
        a = np.arange(4).reshape(-1, 2)
        got = ref.run(None, {"X": a, "Y": a})
        self.assertEqualArray(a @ a, got[0])

    def test_fused_matmul11(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node(
                        "FusedMatMul",
                        ["X", "Y"],
                        ["Z"],
                        transA=1,
                        transB=1,
                        domain="com.microsoft",
                    )
                ],
                "name",
                [
                    oh.make_tensor_value_info("X", TensorProto.FLOAT, None),
                    oh.make_tensor_value_info("Y", TensorProto.FLOAT, None),
                ],
                [make_tensor_value_info("Z", TensorProto.FLOAT, None)],
            ),
            opset_imports=[make_opsetid("", 18), oh.make_opsetid("com.microsoft", 1)],
        )
        ref = ExtendedReferenceEvaluator(model)
        a = np.arange(4).reshape(-1, 2)
        got = ref.run(None, {"X": a, "Y": a})
        self.assertEqualArray(a.T @ a.T, got[0])

    def test_memcpy(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("MemcpyToHost", ["X"], ["Z"]),
                    oh.make_node("MemcpyFromHost", ["X"], ["Z"]),
                ],
                "name",
                [make_tensor_value_info("X", TensorProto.FLOAT, None)],
                [make_tensor_value_info("Z", TensorProto.FLOAT, None)],
            ),
            opset_imports=[make_opsetid("", 18), oh.make_opsetid("com.microsoft", 1)],
            ir_version=9,
        )
        a = np.arange(4).reshape(-1, 2).astype(np.float32)
        ref = ExtendedReferenceEvaluator(model)
        got = ref.run(None, {"X": a})
        self.assertEqualArray(a, got[0])

    def test_quick_gelu(self):
        from onnxruntime import InferenceSession

        for alpha in [0.0, 2.0]:
            model = oh.make_model(
                oh.make_graph(
                    [
                        oh.make_node(
                            "QuickGelu",
                            ["X"],
                            ["Z"],
                            domain="com.microsoft",
                            alpha=alpha,
                        )
                    ],
                    "name",
                    [make_tensor_value_info("X", TensorProto.FLOAT, None)],
                    [make_tensor_value_info("Z", TensorProto.FLOAT, None)],
                ),
                opset_imports=[make_opsetid("", 18), oh.make_opsetid("com.microsoft", 1)],
                ir_version=9,
            )
            sess = InferenceSession(
                model.SerializeToString(), providers=["CPUExecutionProvider"]
            )
            a = np.arange(4).reshape(-1, 2).astype(np.float32)
            expected = sess.run(None, {"X": a})
            ref = ExtendedReferenceEvaluator(model)
            got = ref.run(None, {"X": a})
            self.assertEqualArray(expected[0], got[0])

    def test_scatter_elements(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node(
                        "ScatterElements",
                        ["data", "indices", "updates"],
                        ["Z"],
                        axis=3,
                        reduction="add",
                    )
                ],
                "name",
                [
                    oh.make_tensor_value_info("data", TensorProto.FLOAT, None),
                    oh.make_tensor_value_info("indices", TensorProto.INT64, None),
                    oh.make_tensor_value_info("updates", TensorProto.FLOAT, None),
                ],
                [make_tensor_value_info("Z", TensorProto.FLOAT, None)],
            ),
            opset_imports=[make_opsetid("", 18)],
        )
        data = np.zeros(2**4, dtype=np.float32).reshape((2, 2, 2, 2))
        indices = np.array([[[[0]]]], dtype=np.int64)
        updates = np.array([[[[1]]]], dtype=np.float32)
        y = np.array(
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32
        ).reshape((2, 2, 2, 2))
        ref = ExtendedReferenceEvaluator(model)
        got = ref.run(None, {"data": data, "indices": indices, "updates": updates})
        self.assertEqualArray(y, got[0])

    def test_skip_layer_normalization_nobias(self):
        import onnxruntime

        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node(
                        "SkipLayerNormalization",
                        ["x", "skip", "beta", "gamma"],
                        ["Z"],
                        epsilon=1.0e-5,
                        domain="com.microsoft",
                    )
                ],
                "name",
                [
                    oh.make_tensor_value_info("x", TensorProto.FLOAT, ["a", "b", "c"]),
                    oh.make_tensor_value_info("skip", TensorProto.FLOAT, ["a", "b", "c"]),
                    oh.make_tensor_value_info("beta", TensorProto.FLOAT, ["c"]),
                    oh.make_tensor_value_info("gamma", TensorProto.FLOAT, ["c"]),
                ],
                [make_tensor_value_info("Z", TensorProto.FLOAT, None)],
            ),
            opset_imports=[make_opsetid("", 18), oh.make_opsetid("com.microsoft", 1)],
            ir_version=10,
        )
        feeds = dict(
            x=self._range(2, 3, 8),
            skip=self._range(2, 3, 8, bias=3),
            beta=self._range(8, bias=1),
            gamma=self._range(8, bias=2),
        )
        ref = ExtendedReferenceEvaluator(model)
        got = ref.run(None, feeds)
        sess = onnxruntime.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        expected = sess.run(None, feeds)
        self.assertEqual(len(expected), len(got))
        self.assertEqualArrayAny(expected, got, atol=1e-3)

    def test_skip_layer_normalization_bias(self):
        import onnxruntime

        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node(
                        "SkipLayerNormalization",
                        ["x", "skip", "beta", "gamma", "bias"],
                        ["Z"],
                        epsilon=1.0e-5,
                        domain="com.microsoft",
                    )
                ],
                "name",
                [
                    oh.make_tensor_value_info("x", TensorProto.FLOAT, ["a", "b", "c"]),
                    oh.make_tensor_value_info("skip", TensorProto.FLOAT, ["a", "b", "c"]),
                    oh.make_tensor_value_info("beta", TensorProto.FLOAT, ["c"]),
                    oh.make_tensor_value_info("gamma", TensorProto.FLOAT, ["c"]),
                    oh.make_tensor_value_info("bias", TensorProto.FLOAT, ["c"]),
                ],
                [make_tensor_value_info("Z", TensorProto.FLOAT, None)],
            ),
            opset_imports=[make_opsetid("", 18), oh.make_opsetid("com.microsoft", 1)],
            ir_version=10,
        )
        feeds = dict(
            x=self._range(2, 3, 8),
            skip=self._range(2, 3, 8, bias=3),
            beta=self._range(8, bias=1),
            gamma=self._range(8, bias=2),
            bias=self._range(8, bias=0.1),
        )
        ref = ExtendedReferenceEvaluator(model)
        got = ref.run(None, feeds)
        sess = onnxruntime.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        expected = sess.run(None, feeds)
        self.assertEqual(len(expected), len(got))
        self.assertEqualArrayAny(expected, got, atol=1e-3)

    def _get_model_attention(self) -> onnx.ModelProto:
        # Obtained with:
        # python -m onnx_array_api translate -a onnx-short -m <model.onnx>
        opset_imports = [
            oh.make_opsetid("pkg.onnxscript.torch_lib.common", 1),
            oh.make_opsetid("", 18),
            oh.make_opsetid("pkg.onnxscript.torch_lib", 1),
            oh.make_opsetid("pkg.torch.__subgraph__", 1),
            oh.make_opsetid("com.microsoft", 1),
        ]
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        functions = []
        value = np.random.randn(1024, 1024).astype(np.float32)
        initializers.append(
            from_array_extended(
                np.array(value, dtype=np.float32),
                name="encoder.encoders.0.self_attn.linear_q.weight",
            )
        )
        value = np.random.randn(1024).astype(np.float32)
        initializers.append(
            from_array_extended(
                np.array(value, dtype=np.float32),
                name="encoder.encoders.0.self_attn.linear_q.bias",
            )
        )
        value = np.random.randn(1024, 1024).astype(np.float32)
        initializers.append(
            from_array_extended(
                np.array(value, dtype=np.float32),
                name="encoder.encoders.0.self_attn.linear_k.weight",
            )
        )
        value = np.random.randn(1024).astype(np.float32)
        initializers.append(
            from_array_extended(
                np.array(value, dtype=np.float32),
                name="encoder.encoders.0.self_attn.linear_k.bias",
            )
        )
        value = np.random.randn(1024, 1024).astype(np.float32)
        initializers.append(
            from_array_extended(
                np.array(value, dtype=np.float32),
                name="encoder.encoders.0.self_attn.linear_v.weight",
            )
        )
        value = np.random.randn(1024).astype(np.float32)
        initializers.append(
            from_array_extended(
                np.array(value, dtype=np.float32),
                name="encoder.encoders.0.self_attn.linear_v.bias",
            )
        )
        initializers.append(from_array_extended(np.array(1, dtype=np.int64), name="dim_0_7"))
        inputs.append(
            oh.make_tensor_value_info(
                "layer_norm_1", onnx.TensorProto.FLOAT, shape=("s0", "(s1-1)//8+1", 1024)
            )
        )
        inputs.append(
            oh.make_tensor_value_info(
                "expand_1", onnx.TensorProto.BOOL, shape=("s0", "(s1-1)//8+1", "(s1-1)//8+1")
            )
        )
        inputs.append(
            oh.make_tensor_value_info(
                "unsqueeze_9",
                onnx.TensorProto.FLOAT,
                shape=(1, 16, "(s1-1)//8+1", "(s1-1)//8+1"),
            )
        )
        inputs.append(oh.make_tensor_value_info("val_104", onnx.TensorProto.INT64, shape=(4,)))
        inputs.append(oh.make_tensor_value_info("val_112", onnx.TensorProto.INT64, shape=(4,)))
        inputs.append(oh.make_tensor_value_info("val_120", onnx.TensorProto.INT64, shape=(4,)))
        inputs.append(oh.make_tensor_value_info("val_132", onnx.TensorProto.INT64, shape=(3,)))
        nodes.append(oh.make_node("Unsqueeze", ["expand_1", "dim_0_7"], ["unsqueeze_6"]))
        nodes.append(
            oh.make_node("Cast", ["unsqueeze_6"], ["convert_element_type_default"], to=7)
        )
        nodes.append(
            oh.make_node(
                "Concat",
                [
                    "encoder.encoders.0.self_attn.linear_q.weight",
                    "encoder.encoders.0.self_attn.linear_k.weight",
                    "encoder.encoders.0.self_attn.linear_v.weight",
                ],
                ["encoder.encoders.0.self_attn.linear_q.weight_qkv"],
                axis=1,
            )
        )
        nodes.append(
            oh.make_node(
                "Concat",
                [
                    "encoder.encoders.0.self_attn.linear_q.bias",
                    "encoder.encoders.0.self_attn.linear_k.bias",
                    "encoder.encoders.0.self_attn.linear_v.bias",
                ],
                ["encoder.encoders.0.self_attn.linear_q.bias_bias"],
                axis=0,
            )
        )
        nodes.append(
            oh.make_node(
                "Cast",
                ["convert_element_type_default"],
                ["convert_element_type_default_int32"],
                to=6,
            )
        )
        nodes.append(
            oh.make_node(
                "Attention",
                [
                    "layer_norm_1",
                    "encoder.encoders.0.self_attn.linear_q.weight_qkv",
                    "encoder.encoders.0.self_attn.linear_q.bias_bias",
                    "convert_element_type_default_int32",
                    "",
                    "unsqueeze_9",
                ],
                ["view_3"],
                domain="com.microsoft",
                num_heads=16,
            )
        )
        outputs.append(
            oh.make_tensor_value_info(
                "view_3", onnx.TensorProto.FLOAT, shape=("s0", "(s1-1)//8+1", 1024)
            )
        )
        graph = oh.make_graph(
            nodes,
            "experiment",
            inputs,
            outputs,
            initializers,
            sparse_initializer=sparse_initializers,
        )
        model = oh.make_model(
            graph, functions=functions, opset_imports=opset_imports, ir_version=10
        )
        return model

    def test_attention(self):
        model = self._get_model_attention()
        path = self.dump_onnx("test_attention.onnx", model)
        ref = ExtendedReferenceEvaluator(path)
        feeds = {
            "layer_norm_1": self._range(2, 8, 1024),  # s0,(s1-1)//8+1,1024
            "expand_1": np.random.randint(0, 2, size=(2, 8, 8))
            > 0,  # s0,CeilToInt(IntTrueDiv(s1, 8)),CeilToInt(IntTrueDiv(s1, 8))
            "unsqueeze_9": self._range(1, 16, 8, 8),  # 1,16,(s1-1)//8+1,(s1-1)//8+1
            "val_104": np.array([2, 8, 16, 64], dtype=np.int64),  # s0,(s1-1)//8+1,16,6
            "val_112": np.array([2, 8, 16, 64], dtype=np.int64),  # s0,(s1-1)//8+1,16,6
            "val_120": np.array([2, 8, 16, 64], dtype=np.int64),  # s0,(s1-1)//8+1,16,6
            "val_132": np.array(
                [2, 8, 1024], dtype=np.int64
            ),  # s0,CeilToInt(IntTrueDiv(s1, 8)),1024
        }
        got = ref.run(None, feeds)

        if not has_cuda():
            return
        import onnxruntime

        sess = onnxruntime.InferenceSession(
            model.SerializeToString(),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        expected = sess.run(None, feeds)
        self.assertEqualArrayAny(expected, got, atol=1)

    def test_inline_1_function(self):
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
                oh.make_tensor_value_info("X", TensorProto.FLOAT, [None, None]),
                oh.make_tensor_value_info("A", TensorProto.FLOAT, [None, None]),
                oh.make_tensor_value_info("B", TensorProto.FLOAT, [None, None]),
            ],
            [oh.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
        )

        onnx_model = oh.make_model(
            graph,
            opset_imports=[oh.make_opsetid("", 14), oh.make_opsetid(new_domain, 1)],
            functions=[linear_regression],
        )
        ref = ExtendedReferenceEvaluator(onnx_model)
        feeds = dict(
            X=np.arange(9).reshape((3, 3)).astype(np.float32),
            A=np.arange(9).reshape((3, 3)).astype(np.float32),
            B=np.arange(9).reshape((3, 3)).astype(np.float32),
        )
        ref.run(None, feeds)[0]

    def test_inline_2_functions_recursive(self):
        new_domain = "custom"

        linear_add = oh.make_function(
            new_domain,
            "LinearAdd",
            ["x", "a"],
            ["y"],
            [
                oh.make_node("Add", ["x", "a"], ["y"]),
            ],
            [oh.make_opsetid("", 14)],
            [],
        )

        linear_regression = oh.make_function(
            new_domain,
            "LinearRegression",
            ["x", "a", "b"],
            ["y"],
            [
                oh.make_node("MatMul", ["x", "a"], ["xa"]),
                oh.make_node("LinearAdd", ["xa", "b"], ["y"], domain=new_domain),
            ],
            [oh.make_opsetid("", 14), oh.make_opsetid(new_domain, 1)],
            [],
        )

        graph = oh.make_graph(
            [
                oh.make_node("LinearRegression", ["X", "A", "B"], ["Y2"], domain=new_domain),
                oh.make_node("Abs", ["Y2"], ["Y"]),
            ],
            "example",
            [
                oh.make_tensor_value_info("X", TensorProto.FLOAT, [None, None]),
                oh.make_tensor_value_info("A", TensorProto.FLOAT, [None, None]),
                oh.make_tensor_value_info("B", TensorProto.FLOAT, [None, None]),
            ],
            [oh.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
        )

        onnx_model = oh.make_model(
            graph,
            opset_imports=[oh.make_opsetid("", 14), oh.make_opsetid(new_domain, 1)],
            functions=[linear_add, linear_regression],
        )
        ref = ExtendedReferenceEvaluator(onnx_model)
        feeds = dict(
            X=np.arange(9).reshape((3, 3)).astype(np.float32),
            A=np.arange(9).reshape((3, 3)).astype(np.float32),
            B=np.arange(9).reshape((3, 3)).astype(np.float32),
        )
        ref.run(None, feeds)[0]


if __name__ == "__main__":
    unittest.main(verbosity=2)
