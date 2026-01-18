import textwrap
import unittest
import onnx
import onnx.helper as oh
from onnx_diagnostic.ext_test_case import ExtTestCase, requires_transformers
from onnx_diagnostic.helpers.dot_helper import to_dot
from onnx_diagnostic.export.api import to_onnx
from onnx_diagnostic.torch_export_patches import torch_export_patches
from onnx_diagnostic.torch_models.hghub import get_untrained_model_with_inputs


class TestDotHelper(ExtTestCase):
    def test_custom_doc_kernels_layer_normalization(self):
        TFLOAT16 = onnx.TensorProto.FLOAT16
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
        dot = to_dot(model)
        expected = textwrap.dedent("""
            digraph {
              graph [rankdir=TB, splines=true, overlap=false, nodesep=0.2, ranksep=0.2, fontsize=8];
              node [style="rounded,filled", color="#888888", fontcolor="#222222", shape=box];
              edge [arrowhead=vee, fontsize=7, labeldistance=-5, labelangle=0];
              I_0 [label="X\\nFLOAT16(b,c,d)", fillcolor="#aaeeaa"];
              I_1 [label="W\\nFLOAT16(d)", fillcolor="#aaeeaa"];
              I_2 [label="B\\nFLOAT16(d)", fillcolor="#aaeeaa"];
              LayerNormalization_3 [label="LayerNormalization(., ., ., axis=-1)", fillcolor="#cccccc"];
              Add_4 [label="Add(., ., axis=-1)", fillcolor="#cccccc"];
              I_0 -> LayerNormalization_3 [label="FLOAT16(b,c,d)"];
              I_1 -> LayerNormalization_3 [label="FLOAT16(d)"];
              I_2 -> LayerNormalization_3 [label="FLOAT16(d)"];
              LayerNormalization_3 -> Add_4 [label="FLOAT16(b,c,d)"];
              I_1 -> Add_4 [label="FLOAT16(d)"];
              O_5 [label="Z\\nFLOAT16(b,c,d)", fillcolor="#aaaaee"];
              Add_4 -> O_5;
            }
            """)
        self.maxDiff = None
        self.assertEqual(expected.strip("\n "), dot.strip("\n "))

    def test_custom_doc_kernels_layer_normalization_constant(self):
        TFLOAT16 = onnx.TensorProto.FLOAT16
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
                    oh.make_node("Constant", [], ["cst"], value_float=[1]),
                    oh.make_node("Cast", ["cst"], ["cst16"], to=onnx.TensorProto.FLOAT16),
                    oh.make_node("Add", ["ln", "cst16"], ["Z"], axis=-1),
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
        dot = to_dot(model)
        expected = textwrap.dedent("""
                digraph {
                graph [rankdir=TB, splines=true, overlap=false, nodesep=0.2, ranksep=0.2, fontsize=8];
                node [style="rounded,filled", color="#888888", fontcolor="#222222", shape=box];
                edge [arrowhead=vee, fontsize=7, labeldistance=-5, labelangle=0];
                I_0 [label="X\\nFLOAT16(b,c,d)", fillcolor="#aaeeaa"];
                I_1 [label="W\\nFLOAT16(d)", fillcolor="#aaeeaa"];
                I_2 [label="B\\nFLOAT16(d)", fillcolor="#aaeeaa"];
                LayerNormalization_3 [label="LayerNormalization(., ., ., axis=-1)", fillcolor="#cccccc"];
                Cast_4 [label="Cast([1.0], to=FLOAT16)", fillcolor="#cccccc"];
                Add_5[label="Add(.,.,axis=-1)",fillcolor="#cccccc"];
                I_0 -> LayerNormalization_3 [label="FLOAT16(b,c,d)"];
                I_1 -> LayerNormalization_3 [label="FLOAT16(d)"];
                I_2 -> LayerNormalization_3 [label="FLOAT16(d)"];
                LayerNormalization_3 -> Add_5 [label="FLOAT16(b,c,d)"];
                Cast_4->Add_5[label="FLOAT16()"];
                O_6 [label="Z\\nFLOAT16(b,c,d)", fillcolor="#aaaaee"];
                Add_5 -> O_6;
                }
                """).strip("\n").replace(" ", "")
        self.maxDiff = None
        self.assertEqual(expected, dot.strip("\n").replace(" ", ""))

    @requires_transformers("4.57")
    def test_dot_plot_tiny(self):
        data = get_untrained_model_with_inputs("arnir0/Tiny-LLM")
        model, inputs, ds = data["model"], data["inputs"], data["dynamic_shapes"]
        with torch_export_patches(patch_transformers=True):
            em = to_onnx(model, inputs, dynamic_shapes=ds, exporter="custom")
        dot = to_dot(em.model_proto)
        name = self.get_dump_file("test_dot_plot_tiny.dot")
        with open(name, "w") as f:
            f.write(dot)
        # dot -Tpng dump_test/test_dot_plot_tiny.dot -o dump_test/test_dot_plot_tiny.png
        self.assertIn("-> Add", dot)


if __name__ == "__main__":
    unittest.main(verbosity=2)
