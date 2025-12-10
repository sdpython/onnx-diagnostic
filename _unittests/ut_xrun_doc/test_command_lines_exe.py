import os
import unittest
from contextlib import redirect_stdout
from io import StringIO
import pandas
import torch
from onnx_diagnostic.ext_test_case import ExtTestCase, ignore_warnings, requires_transformers
from onnx_diagnostic._command_lines_parser import main
from onnx_diagnostic.helpers.log_helper import enumerate_csv_files
from onnx_diagnostic.export.api import to_onnx


class TestCommandLines(ExtTestCase):
    @property
    def dummy_path(self):
        return os.path.abspath(
            os.path.join(os.path.dirname(__file__), "data", "two_nodes.onnx")
        )

    def test_a_parser_print(self):
        for fmt in ["raw", "text", "pretty", "printer", "shape", "dot"]:
            with self.subTest(format=fmt):
                st = StringIO()
                with redirect_stdout(st):
                    main(["print", fmt, self.dummy_path])
                text = st.getvalue()
                self.assertIn("Add", text)

    @unittest.skipIf(
        torch.__version__.startswith("2.9.0"), "Possibly one issue with matplotlib."
    )
    def test_b_parser_stats(self):
        output = self.get_dump_file("test_parser_stats.xlsx")
        st = StringIO()
        with redirect_stdout(st):
            main(["stats", "-i", self.dummy_path, "-o", output, "-r", ".*"])
        text = st.getvalue()
        self.assertIn("processing", text)
        self.assertExists(output)

    def test_c_parser_find(self):
        st = StringIO()
        with redirect_stdout(st):
            main(["find", "-i", self.dummy_path, "-n", "node_Add_188"])
        text = st.getvalue()
        self.assertIsInstance(text, str)

    def test_d_parser_find_v2(self):
        st = StringIO()
        with redirect_stdout(st):
            main(["find", "-i", self.dummy_path, "-n", "node_Add_188", "--v2"])
        text = st.getvalue()
        self.assertIsInstance(text, str)

    def test_e_parser_config(self):
        st = StringIO()
        with redirect_stdout(st):
            main(["config", "-m", "arnir0/Tiny-LLM"])
        text = st.getvalue()
        self.assertIn("LlamaForCausalLM", text)

    def test_f_parser_validate(self):
        st = StringIO()
        with redirect_stdout(st):
            main(["validate"])
            main(["validate", "-t", "text-generation"])
        text = st.getvalue()
        self.assertIn("dynamic_shapes", text)
        st = StringIO()
        with redirect_stdout(st):
            main(["validate"])
            main(["validate", "-m", "arnir0/Tiny-LLM", "--run", "-v", "1"])
        text = st.getvalue()
        self.assertIn("model_clas", text)

    @ignore_warnings(UserWarning)
    @unittest.skipIf(
        torch.__version__.startswith("2.9.0"), "Possibly one issue with matplotlib."
    )
    def test_g_parser_agg(self):
        path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "ut_helpers", "data")
        )
        assert list(enumerate_csv_files([f"{path}/*.zip"]))
        output = self.get_dump_file("test_parser_agg.xlsx")
        st = StringIO()
        with redirect_stdout(st):
            main(["agg", output, f"{path}/*.zip", "--filter", ".*.csv", "-v", "1"])
        text = st.getvalue()
        self.assertIn("[CubeLogs.to_excel] plots 1 plots", text)
        self.assertExists(output)

    @ignore_warnings(UserWarning)
    @requires_transformers("4.53")
    def test_h_parser_sbs(self):
        import torch

        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.fc1 = torch.nn.Linear(10, 32)  # input size 10 → hidden size 32
                self.relu = torch.nn.ReLU()
                self.fc2 = torch.nn.Linear(32, 1)  # hidden → output

            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.fc2(x)
                return x

        inputs = dict(x=torch.randn((5, 10)))
        ds = dict(x={0: "batch"})
        input_file = self.get_dump_file("test_h_parser_sbs.inputs.pt")
        ep_file = self.get_dump_file("test_h_parser_sbs.ep")
        onnx_file = self.get_dump_file("test_h_parser_sbs.model.onnx")
        replay_foler = self.get_dump_folder("test_h_parser_sbs.replay")
        torch.save(inputs, input_file)
        to_onnx(
            Model(),
            kwargs=inputs,
            dynamic_shapes=ds,
            exporter="custom",
            save_ep=(ep_file, 2**30),
            filename=onnx_file,
        )

        output = self.get_dump_file("test_h_parser_sbs.xlsx")
        st = StringIO()
        with redirect_stdout(st):
            main(
                [
                    "sbs",
                    "-v",
                    "2",
                    "--first",
                    "-i",
                    input_file,
                    "-e",
                    f"{ep_file}.ep.pt2",
                    "-o",
                    output,
                    "-m",
                    onnx_file,
                    "-t",
                    "Gemm",
                    "-f",
                    replay_foler,
                ]
            )
        text = st.getvalue()
        self.assertIn("[run_aligned", text)
        self.assertExists(output)
        df = pandas.read_excel(output).apply(
            lambda col: col.fillna("") if col.dtype == "object" else col
        )
        self.assertLess(df["err_abs"].max(), 1e-5)
        self.assertEqual(df["err_h01"].max(), 0)
        self.assertIn("p_fc1_weight", set(df["ep_name"]))
        self.assertIn("fc1.bias", set(df["onnx_name"]))
        self.assertNotIn("NaN", set(df["ep_name"]))
        # print(f"{df}\n{st.getvalue()}")
        self.assertIn("[run_aligned] done", st.getvalue())
        sdf = df[(df.ep_target == "placeholder") & (df.onnx_op_type == "initializer")]
        self.assertEqual(sdf.shape[0], 4)

    @ignore_warnings(UserWarning)
    @requires_transformers("4.53")
    def test_i_parser_dot(self):
        import torch

        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.fc1 = torch.nn.Linear(10, 32)  # input size 10 → hidden size 32
                self.relu = torch.nn.ReLU()
                self.fc2 = torch.nn.Linear(32, 1)  # hidden → output

            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.fc2(x)
                return x

        inputs = dict(x=torch.randn((5, 10)))
        ds = dict(x={0: "batch"})
        onnx_file = self.get_dump_file("test_i_parser_dot.model.onnx")
        to_onnx(
            Model(),
            kwargs=inputs,
            dynamic_shapes=ds,
            exporter="custom",
            filename=onnx_file,
        )

        output = self.get_dump_file("test_i_parser_dot.dot")
        args = ["dot", onnx_file, "-v", "1", "-o", output]
        if not self.unit_test_going():
            args.extend(["--run", "svg"])

        st = StringIO()
        with redirect_stdout(st):
            main(args)
        text = st.getvalue()
        if text:
            # text is empty is dot is not installed
            self.assertIn("converts into dot", text)

    def test_j_parser_compare(self):
        st = StringIO()
        with redirect_stdout(st):
            main(["compare", self.dummy_path, self.dummy_path])
        text = st.getvalue()
        print(text)
        self.assertIn("done with distance 0", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
