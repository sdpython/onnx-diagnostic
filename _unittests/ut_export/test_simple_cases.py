import unittest
import torch
from onnx_diagnostic.ext_test_case import ExtTestCase
from onnx_diagnostic.reference import ExtendedReferenceEvaluator


class TestDynamicShapes(ExtTestCase):
    def test_getitem_index_put1(self):
        class Model(torch.nn.Module):
            def forward(self, x, value):
                x = x.clone()
                x[:, :, :, : value.shape[-1]] = value
                return x

        inputs = (torch.randn(2, 2, 3, 4), torch.randn(2, 2, 3, 3))
        model = Model()
        expected = model(*inputs)

        onx = self.to_onnx(model, inputs, dynamic_shapes=({3: "M"}, {3: "N"}))
        self.dump_onnx("test_getitem_index_put1.onnx", onx)
        feeds = dict(zip(["x", "value"], [x.detach().cpu().numpy() for x in inputs]))
        ref = ExtendedReferenceEvaluator(onx, verbose=0)
        got = ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-5)
        sess = self.ort().InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
