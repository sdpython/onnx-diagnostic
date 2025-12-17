import unittest
import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh
from onnx_diagnostic.ext_test_case import ExtTestCase, hide_stdout
from onnx_diagnostic.helpers.optim_helper import optimize_model

TFLOAT = onnx.TensorProto.FLOAT


class TestOptimHelper(ExtTestCase):
    @hide_stdout()
    def test_optimize_model(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Shape", ["X"], ["D2"], start=2, end=3),
                    oh.make_node("Concat", ["I1", "D2"], ["d"], axis=0),
                    oh.make_node("Reshape", ["X", "d"], ["Y"]),
                ],
                "test",
                [oh.make_tensor_value_info("X", TFLOAT, [2, 3, "d"])],
                [oh.make_tensor_value_info("Y", TFLOAT, [6, "d"])],
                [onh.from_array(np.array([-1], dtype=np.int64), name="I1")],
            ),
            opset_imports=[oh.make_operatorsetid("", 18)],
            ir_version=10,
        )
        filename = self.dump_onnx("test_optimize_model.onnx", model)
        for algo in ["default", "default+onnxruntime", "ir", "os_ort", "slim"]:
            output = self.get_dump_file(f"test_optimize_model.{algo}.onnx")
            with self.subTest(algo=algo):
                optimize_model(algo, filename, output=output, verbose=1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
