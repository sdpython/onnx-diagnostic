import unittest
import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh
import torch
from onnx_diagnostic.ext_test_case import ExtTestCase
from onnx_diagnostic.torch_onnx.runtime_info import (
    first_used_last_used,
    RuntimeValue,
    RuntimeValueKind,
    RuntimeDevice,
)


class TestRuntimeInfo(ExtTestCase):
    def test_runtime_info(self):
        rt = RuntimeValue("e", is_shape=True, value=torch.Tensor([0]))
        r = repr(rt)
        self.assertEqual("RuntimeValue(name=e, is_shape=True, value=T1s1)", r)

    def test_runtime_kind(self):
        h = RuntimeValueKind.INPUT
        self.assertEqual(h.to_str(), "INPUT")

    def test_runtime_device(self):
        h = RuntimeDevice.CPU
        self.assertEqual(h.to_str(), "CPU")

    def test_runtime_values(self):
        def _mkv_(name):
            value_info_proto = onnx.ValueInfoProto()
            value_info_proto.name = name
            return value_info_proto

        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("ReduceSum", ["0X"], ["1Xred"]),
                    oh.make_node("Add", ["0X", "0two"], ["2X0"]),
                    oh.make_node("Add", ["2X0", "0zero"], ["3X00"]),
                    oh.make_node("CastLike", ["0one", "1Xred"], ["4one_c"]),
                    oh.make_node("Greater", ["1Xred", "4one_c"], ["5cond"]),
                    oh.make_node(
                        "If",
                        ["5cond"],
                        ["6Z_c"],
                        then_branch=oh.make_graph(
                            [
                                oh.make_node("Constant", [], ["0two"], value_floats=[2.1]),
                                oh.make_node("Add", ["3X00", "0two"], ["11Y"]),
                            ],
                            "then",
                            [],
                            [_mkv_("11Y")],
                        ),
                        else_branch=oh.make_graph(
                            [
                                oh.make_node("Constant", [], ["0two"], value_floats=[2.2]),
                                oh.make_node("Sub", ["2X0", "0two"], ["12Y"]),
                            ],
                            "else",
                            [],
                            [_mkv_("12Y")],
                        ),
                    ),
                    oh.make_node("CastLike", ["6Z_c", "0X"], ["7Z"]),
                ],
                "test",
                [
                    oh.make_tensor_value_info("0X", onnx.TensorProto.FLOAT, ["N"]),
                    oh.make_tensor_value_info("0one", onnx.TensorProto.FLOAT, ["N"]),
                ],
                [oh.make_tensor_value_info("7Z", onnx.TensorProto.UNDEFINED, ["N"])],
                [
                    onh.from_array(np.array([0], dtype=np.float32), name="0zero"),
                    onh.from_array(np.array([2], dtype=np.float32), name="0two"),
                ],
            ),
            opset_imports=[oh.make_operatorsetid("", 18)],
            ir_version=10,
        )
        rt_values = first_used_last_used(model)
        self.assertEqual(
            {
                "2X0",
                "0two",
                "5cond",
                "1Xred",
                "0zero",
                "0X",
                "4one_c",
                "7Z",
                "6Z_c",
                "0one",
                "3X00",
            },
            set(rt_values),
        )
        for name, node in rt_values.items():
            self.assertEqual(name, node.name)
            if name != "7Z":
                self.assertIsInstance(node.first_used, int)
            self.assertIsInstance(node.last_used, int)
            self.assertIsInstance(node.created, int, msg=f"{name!r} missing 'created'")
            self.assertIsInstance(node.kind, int)
            self.assertEqual(
                int(name[0]) - 1, node.created, msg=f"{name!r} created is wrong {node.created}"
            )
            if name != "7Z":
                self.assertGreater(node.first_used, node.created)
                self.assertGreaterOrEqual(node.last_used, node.first_used)


if __name__ == "__main__":
    unittest.main(verbosity=2)
