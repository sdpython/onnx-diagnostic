import unittest
import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh
from onnx_diagnostic.ext_test_case import ExtTestCase
from onnx_diagnostic.torch_onnx.compare import ObsCompare

TFLOAT = onnx.TensorProto.FLOAT
TINT64 = onnx.TensorProto.INT64


class TestCompare(ExtTestCase):
    def _get_model(self, cast=True):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Unsqueeze", ["X", "zero"], ["xu1"]),
                    oh.make_node("Unsqueeze", ["xu1", "un"], ["xu2"]),
                    oh.make_node("Reshape", ["xu2", "shape1"], ["xm1"]),
                    oh.make_node("Reshape", ["Y", "shape2"], ["xm2c"]),
                    (
                        oh.make_node("Cast", ["xm2c"], ["xm2"], to=1)
                        if cast
                        else oh.make_node("Identity", ["xmc2"], ["xm2"])
                    ),
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

    def test_edit_distance_0(self):
        model = self._get_model()
        seq = ObsCompare.obs_sequence_from_model(model)
        dist, path, pair_cmp = ObsCompare.distance_sequence(seq, seq)
        self.assertEqual(dist, 0)
        self.assertEqual(path, [(i, i) for i in range(len(path))])
        self.assertEqual(len(path), len(pair_cmp))
        uni = set()
        for o1, o2 in pair_cmp:
            self.assertIsInstance(o1, ObsCompare)
            self.assertIsInstance(o2, ObsCompare)
            self.assertEqual(o1, o2)
            row = f"{o1} | {o2}"
            uni.add(len(row))
        self.assertEqual(len(uni), 1)

    def test_edit_distance_1(self):
        model = self._get_model()
        model2 = self._get_model(cast=False)
        seq1 = ObsCompare.obs_sequence_from_model(model)
        seq2 = ObsCompare.obs_sequence_from_model(model2)
        dist, path, pair_cmp = ObsCompare.distance_sequence(seq1, seq2)
        self.assertGreater(dist, 2000)
        expected_path = [
            *[(i, i) for i in range(11)],
            *[(10, 11), (11, 11)],
            *[(i, i) for i in range(12, len(seq1))],
        ]
        self.assertEqual(expected_path, path)
        self.assertEqual(len(path), len(pair_cmp))
        n1, n2, n12 = 0, 0, 0
        for o1, o2 in pair_cmp:
            if o1:
                self.assertIsInstance(o1, ObsCompare)
            else:
                n1 += 1
            if o2:
                self.assertIsInstance(o2, ObsCompare)
            else:
                n2 += 1
            if o1 and o2:
                self.assertEqual(o1, o2)
            elif not o1 and not o2:
                n12 += 1
        self.assertEqual(n1, 1)
        self.assertEqual(n2, 1)
        self.assertEqual(n12, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
