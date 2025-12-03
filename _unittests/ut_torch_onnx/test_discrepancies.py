import os
import unittest
import numpy as np
import onnx
from onnx_diagnostic.ext_test_case import ExtTestCase


class TestDiscrepancies(ExtTestCase):
    def test_attention_opset15_in_a_loop(self):
        model = onnx.load(
            os.path.join(os.path.dirname(__file__), "data", "attention_loopa24.onnx")
        )
        sess = self.check_ort(model)
        feeds = dict(
            c_lifted_tensor_0=np.array([0], dtype=np.int64),
            cat_2=np.array(
                [
                    0,
                    64,
                    128,
                    192,
                    256,
                    304,
                    368,
                    432,
                    496,
                    560,
                    608,
                    672,
                    736,
                    800,
                    864,
                    912,
                    976,
                    1040,
                    1104,
                    1168,
                    1216,
                    1232,
                    1248,
                    1264,
                    1280,
                    1292,
                ],
                dtype=np.int64,
            ),
            unsqueeze_4=np.random.randn(1, 16, 1292, 80).astype(np.float32),
            unsqueeze_5=np.random.randn(1, 16, 1292, 80).astype(np.float32),
            unsqueeze_6=np.random.randn(1, 16, 1292, 80).astype(np.float32),
        )
        got = sess.run(None, feeds)
        self.assertEqual(len(got), 1)
        self.assertEqual((1, 1292, 16, 80), got[0].shape)


if __name__ == "__main__":
    unittest.main(verbosity=2)
