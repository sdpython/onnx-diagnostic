import textwrap
import unittest
import onnx
import onnx.helper as oh
from onnx_diagnostic.ext_test_case import ExtTestCase
from onnx_diagnostic.helpers.graph_helper import GraphRendering

TFLOAT = onnx.TensorProto.FLOAT


class TestGraphHelper(ExtTestCase):
    def test_computation_order(self):
        proto = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Sigmoid", ["Y"], ["sy"]),
                    oh.make_node("Mul", ["Y", "sy"], ["ysy"]),
                    oh.make_node("Mul", ["X", "ysy"], ["final"]),
                ],
                "-nd-",
                [
                    oh.make_tensor_value_info("X", TFLOAT, [1, "b", "c"]),
                    oh.make_tensor_value_info("Y", TFLOAT, ["a", "b", "c"]),
                ],
                [oh.make_tensor_value_info("final", TFLOAT, ["a", "b", "c"])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        order = GraphRendering.computation_order(
            proto.graph.node, [i.name for i in [*proto.graph.input, *proto.graph.initializer]]
        )
        self.assertEqual([1, 2, 3], order)

    def test_graph_positions1(self):
        proto = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Sigmoid", ["Y"], ["sy"]),
                    oh.make_node("Mul", ["Y", "sy"], ["ysy"]),
                    oh.make_node("Mul", ["X", "ysy"], ["final"]),
                ],
                "-nd-",
                [
                    oh.make_tensor_value_info("X", TFLOAT, [1, "b", "c"]),
                    oh.make_tensor_value_info("Y", TFLOAT, ["a", "b", "c"]),
                ],
                [oh.make_tensor_value_info("final", TFLOAT, ["a", "b", "c"])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        existing = [i.name for i in [*proto.graph.input, *proto.graph.initializer]]
        order = GraphRendering.computation_order(proto.graph.node, existing)
        positions = GraphRendering.graph_positions(proto.graph.node, order, existing)
        self.assertEqual([(1, 0), (2, 0), (3, 0)], positions)

    def test_graph_positions2(self):
        proto = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Add", ["X", "Y"], ["xy"]),
                    oh.make_node("Neg", ["Y"], ["ny"]),
                    oh.make_node("Mul", ["xy", "ny"], ["Z"]),
                ],
                "-nd-",
                [
                    oh.make_tensor_value_info("X", TFLOAT, ["a", "b", "c"]),
                    oh.make_tensor_value_info("Y", TFLOAT, ["a", "b", "c"]),
                ],
                [oh.make_tensor_value_info("Z", TFLOAT, ["a", "b", "c"])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        existing = [i.name for i in [*proto.graph.input, *proto.graph.initializer]]
        order = GraphRendering.computation_order(proto.graph.node, existing)
        positions = GraphRendering.graph_positions(proto.graph.node, order, existing)
        self.assertEqual([(1, 0), (1, 1), (2, 0)], positions)

    def test_text_positionss(self):
        proto = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Add", ["X", "Y"], ["xy"]),
                    oh.make_node("Neg", ["Y"], ["ny"]),
                    oh.make_node("Mul", ["xy", "ny"], ["a"]),
                    oh.make_node("Mul", ["a", "Y"], ["Z"]),
                ],
                "-nd-",
                [
                    oh.make_tensor_value_info("X", TFLOAT, ["a", "b", "c"]),
                    oh.make_tensor_value_info("Y", TFLOAT, ["a", "b", "c"]),
                ],
                [oh.make_tensor_value_info("Z", TFLOAT, ["a", "b", "c"])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        existing = [i.name for i in [*proto.graph.input, *proto.graph.initializer]]
        order = GraphRendering.computation_order(proto.graph.node, existing)
        self.assertEqual([1, 1, 2, 3], order)
        positions = GraphRendering.graph_positions(proto.graph.node, order, existing)
        self.assertEqual([(1, 0), (1, 1), (2, 0), (3, 0)], positions)
        text_pos = GraphRendering.text_positions(proto.graph.node, positions)
        self.assertEqual([(4, 4), (4, 20), (8, 12), (12, 20)], text_pos)

    def test_text_rendering(self):
        proto = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Add", ["X", "Y"], ["xy"]),
                    oh.make_node("Neg", ["Y"], ["ny"]),
                    oh.make_node("Mul", ["xy", "ny"], ["a"]),
                    oh.make_node("Mul", ["a", "Y"], ["Z"]),
                ],
                "-nd-",
                [
                    oh.make_tensor_value_info("X", TFLOAT, ["a", "b", "c"]),
                    oh.make_tensor_value_info("Y", TFLOAT, ["a", "b", "c"]),
                ],
                [oh.make_tensor_value_info("Z", TFLOAT, ["a", "b", "c"])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        graph = GraphRendering(proto)
        text = textwrap.dedent(graph.text_rendering(prefix="|")).strip("\n")
        expected = textwrap.dedent(
            """
            |
            |
            |
            |
            |   Add             Neg
            |    |               |
            |    +-------+-------+
            |            |
            |           Mul
            |            |
            |            +-------+
            |                    |
            |                   Mul
            |
            |
            """
        ).strip("\n")
        self.assertEqual(expected, text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
