import unittest
import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh
from onnx_diagnostic.ext_test_case import ExtTestCase, ignore_warnings
from onnx_diagnostic.export.api import to_onnx
from onnx_diagnostic.torch_onnx.compare import ObsCompare, ObsComparePair

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
        for pair in pair_cmp:
            o1, o2 = pair.side1, pair.side2
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
        self.assertGreaterOrEqual(dist, 900)
        self.assertEqual([(i, i) for i in range(len(path))], path)
        self.assertEqual(len(path), len(pair_cmp))
        n1, n2, n12 = 0, 0, 0
        for pair in pair_cmp:
            o1, o2 = pair.side1, pair.side2
            if o1:
                self.assertIsInstance(o1, ObsCompare)
            else:
                n1 += 1
            if o2:
                self.assertIsInstance(o2, ObsCompare)
            else:
                n2 += 1
            if o1 and o2:
                pass
            elif not o1 and not o2:
                n12 += 1
        self.assertEqual(n1, 0)
        self.assertEqual(n2, 0)
        self.assertEqual(n12, 0)

    @ignore_warnings(DeprecationWarning)
    def test_comp_model_gemm(self):
        import torch

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 16, 5)
                self.fc1 = torch.nn.Linear(144, 64)
                self.fc2 = torch.nn.Linear(64, 128)
                self.fc3 = torch.nn.Linear(128, 10)

            def forward(self, x):
                x = torch.nn.functional.max_pool2d(
                    torch.nn.functional.relu(self.conv1(x)), (4, 4)
                )
                # x = F.max_pool2d(F.relu(self.conv2(x)), 2)
                x = torch.flatten(x, 1)
                x = torch.nn.functional.relu(self.fc1(x))
                x = torch.nn.functional.relu(self.fc2(x))
                y = self.fc3(x)
                return y

        model = Model()
        x = torch.randn((2, 3, 16, 17), dtype=torch.float32)
        model(x)
        dynamic_shapes = ({0: "batch", 3: "dim"},)
        onx_file = self.get_dump_file("test_comp_model_gemm.onnx")
        to_onnx(
            model, (x,), dynamic_shapes=dynamic_shapes, exporter="custom", optimize=True
        ).save(onx_file)
        onx = onnx.load(onx_file)
        self.assert_onnx_disc("test_comp_model_gemm", onx, model, (x,), use_ort=True)
        seq1 = ObsCompare.obs_sequence_from_model(onx)
        seq2 = ObsCompare.obs_sequence_from_model(onx)
        dist, _path, pair_cmp = ObsCompare.distance_sequence(seq1, seq2)
        text = str(pair_cmp[0])
        self.assertIn("0000 INITIA", text)
        self.assertNotIn("(", text)
        text = ObsComparePair.to_str(pair_cmp)
        self.assertEqual(dist, 0)
        self.assertNotIn("?", text)
        self.assertIn("0013 NODE", text)
        onx_file0 = self.get_dump_file("test_comp_model_gemm0.onnx")
        to_onnx(
            model, (x,), dynamic_shapes=dynamic_shapes, exporter="custom", optimize=False
        ).save(onx_file0)
        onx0 = onnx.load(onx_file0)
        seq1 = ObsCompare.obs_sequence_from_model(onx0)
        seq2 = ObsCompare.obs_sequence_from_model(onx)
        _dist, _path, pair_cmp = ObsCompare.distance_sequence(seq1, seq2)
        text = ObsComparePair.to_str(pair_cmp)
        self.assertIn("Conv", text)
        for pair in pair_cmp:
            assert (
                pair.side1.op_type != "Conv" or pair.side2.op_type == "FusedConv"
            ), f"wrong pair {pair!r}"


if __name__ == "__main__":
    unittest.main(verbosity=2)
