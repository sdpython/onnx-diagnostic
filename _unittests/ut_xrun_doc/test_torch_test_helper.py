import unittest
import numpy as np
import ml_dtypes
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh
import torch
from onnx_diagnostic.ext_test_case import ExtTestCase
from onnx_diagnostic.torch_test_helper import (
    dummy_llm,
    check_model_ort,
    to_numpy,
    is_torchdynamo_exporting,
)

TFLOAT = onnx.TensorProto.FLOAT


class TestOrtSession(ExtTestCase):

    def test_is_torchdynamo_exporting(self):
        self.assertFalse(is_torchdynamo_exporting())

    def test_dummy_llm(self):
        for cls_name in ["AttentionBlock", "MultiAttentionBlock", "DecoderLayer", "LLM"]:
            model, inputs = dummy_llm(cls_name)
            model(*inputs)

    def test_dummy_llm_ds(self):
        for cls_name in ["AttentionBlock", "MultiAttentionBlock", "DecoderLayer", "LLM"]:
            model, inputs, ds = dummy_llm(cls_name, dynamic_shapes=True)
            model(*inputs)
            self.assertIsInstance(ds, dict)

    def test_dummy_llm_exc(self):
        self.assertRaise(lambda: dummy_llm("LLLLLL"), NotImplementedError)

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

    def test_to_numpy(self):
        t = torch.tensor([0, 1], dtype=torch.bfloat16)
        a = to_numpy(t)
        self.assertEqual(a.dtype, ml_dtypes.bfloat16)


if __name__ == "__main__":
    unittest.main(verbosity=2)
