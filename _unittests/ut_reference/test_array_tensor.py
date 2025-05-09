import unittest
import numpy as np
from onnx import TensorProto
from onnx.helper import make_graph, make_model, make_node, make_tensor_value_info
from onnx_diagnostic.ext_test_case import ExtTestCase, ignore_warnings
from onnx_diagnostic.helpers.onnx_helper import from_array_extended, to_array_extended
from onnx_diagnostic.reference import ExtendedReferenceEvaluator


class TestArrayTensor(ExtTestCase):
    @ignore_warnings(DeprecationWarning)
    def test_from_array(self):
        for dt in (np.float32, np.float16, np.uint16, np.uint8):
            with self.subTest(dtype=dt):
                a = np.array([0, 1, 2], dtype=dt)
                t = from_array_extended(a, "a")
                b = to_array_extended(t)
                self.assertEqualArray(a, b)
                t2 = from_array_extended(b, "a")
                self.assertEqual(t.SerializeToString(), t2.SerializeToString())

    @ignore_warnings(DeprecationWarning)
    def test_from_array_f8(self):
        def make_model_f8(fr, to):
            model = make_model(
                make_graph(
                    [make_node("Cast", ["X"], ["Y"], to=to)],
                    "cast",
                    [make_tensor_value_info("X", fr, None)],
                    [make_tensor_value_info("Y", to, None)],
                )
            )
            return model

        for dt in (np.float32, np.float16, np.uint16, np.uint8):
            with self.subTest(dtype=dt):
                a = np.array([0, 1, 2], dtype=dt)
                b = from_array_extended(a, "a")
                for to in [
                    TensorProto.FLOAT8E4M3FN,
                    TensorProto.FLOAT8E4M3FNUZ,
                    TensorProto.FLOAT8E5M2,
                    TensorProto.FLOAT8E5M2FNUZ,
                    TensorProto.BFLOAT16,
                ]:
                    with self.subTest(fr=b.data_type, to=to):
                        model = make_model_f8(b.data_type, to)
                        ref = ExtendedReferenceEvaluator(model)
                        got = ref.run(None, {"X": a})[0]
                        back = from_array_extended(got, "a")
                        self.assertEqual(to, back.data_type)


if __name__ == "__main__":
    unittest.main(verbosity=2)
