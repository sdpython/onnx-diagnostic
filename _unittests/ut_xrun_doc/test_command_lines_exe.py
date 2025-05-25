import os
import unittest
from contextlib import redirect_stdout
from io import StringIO
from onnx_diagnostic.ext_test_case import ExtTestCase
from onnx_diagnostic._command_lines_parser import main


class TestCommandLines(ExtTestCase):
    @property
    def dummy_path(self):
        return os.path.abspath(
            os.path.join(os.path.dirname(__file__), "data", "two_nodes.onnx")
        )

    def test_parser_print(self):
        for fmt in ["raw", "text", "pretty", "printer"]:
            with self.subTest(format=fmt):
                st = StringIO()
                with redirect_stdout(st):
                    main(["print", fmt, self.dummy_path])
                text = st.getvalue()
                self.assertIn("Add", text)

    def test_parser_stats(self):
        output = self.get_dump_file("test_parser_stats.xlsx")
        st = StringIO()
        with redirect_stdout(st):
            main(["stats", "-i", self.dummy_path, "-o", output, "-r", ".*"])
        text = st.getvalue()
        self.assertIn("processing", text)
        self.assertExists(output)

    def test_parser_find(self):
        st = StringIO()
        with redirect_stdout(st):
            main(["find", "-i", self.dummy_path, "-n", "node_Add_188"])
        text = st.getvalue()
        self.assertIsInstance(text, str)

    def test_parser_config(self):
        st = StringIO()
        with redirect_stdout(st):
            main(["config", "-m", "arnir0/Tiny-LLM"])
        text = st.getvalue()
        self.assertIn("LlamaForCausalLM", text)

    def test_parser_validate(self):
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
