import os
import unittest
from contextlib import redirect_stdout
from io import StringIO
import torch
from onnx_diagnostic.ext_test_case import ExtTestCase, ignore_warnings
from onnx_diagnostic._command_lines_parser import main
from onnx_diagnostic.helpers.log_helper import enumerate_csv_files


class TestCommandLines(ExtTestCase):
    @property
    def dummy_path(self):
        return os.path.abspath(
            os.path.join(os.path.dirname(__file__), "data", "two_nodes.onnx")
        )

    def test_a_parser_print(self):
        for fmt in ["raw", "text", "pretty", "printer"]:
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
