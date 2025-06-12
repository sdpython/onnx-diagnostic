import unittest
import onnx
import onnx.helper as oh
import torch
from onnx_diagnostic.ext_test_case import ExtTestCase
from onnx_diagnostic.helpers.doc_helper import LayerNormalizationOrt, MatMulOrt
from onnx_diagnostic.reference import TorchOnnxEvaluator

TFLOAT = onnx.TensorProto.FLOAT
TFLOAT16 = onnx.TensorProto.FLOAT16


class TestDocHelper(ExtTestCase):
    def test_custom_doc_kernels_layer_normalization(self):
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
            opset_imports=[oh.make_opsetid("", 18)],
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

    def test_custom_doc_kernels_matmul(self):
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("MatMul", ["X", "Y"], ["Z"])],
                "dummy",
                [
                    oh.make_tensor_value_info("X", TFLOAT16, ["b", "c", "d"]),
                    oh.make_tensor_value_info("Y", TFLOAT16, ["b", "d", "e"]),
                ],
                [oh.make_tensor_value_info("Z", TFLOAT16, ["b", "c", "e"])],
            ),
            ir_version=9,
            opset_imports=[oh.make_opsetid("", 18)],
        )

        torch_sess = TorchOnnxEvaluator(model, verbose=0)
        torch_sess_custom = TorchOnnxEvaluator(
            model,
            verbose=0,
            custom_kernels={("", "MatMul"): MatMulOrt},
        )
        feeds = dict(
            zip(
                torch_sess.input_names,
                [
                    torch.rand(3, 4, 5, dtype=torch.float16),
                    torch.rand(3, 5, 7, dtype=torch.float16),
                ],
            )
        )
        expected = torch_sess.run(None, feeds)
        got = torch_sess_custom.run(None, feeds)
        self.assertEqualAny(expected, got)


if __name__ == "__main__":
    unittest.main(verbosity=2)
