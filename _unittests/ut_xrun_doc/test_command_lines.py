import unittest
from contextlib import redirect_stdout
from io import StringIO
from onnx_diagnostic.ext_test_case import ExtTestCase
from onnx_diagnostic._command_lines_parser import (
    get_main_parser,
    get_parser_agg,
    get_parser_compare,
    get_parser_config,
    get_parser_dot,
    get_parser_find,
    get_parser_lighten,
    get_parser_optimize,
    get_parser_partition,
    get_parser_print,
    get_parser_sbs,
    get_parser_stats,
    get_parser_unlighten,
    get_parser_validate,
)


class TestCommandLines(ExtTestCase):
    def test_main_parser(self):
        st = StringIO()
        with redirect_stdout(st):
            get_main_parser().print_help()
        text = st.getvalue()
        self.assertIn("lighten", text)
        self.assertIn("dot", text)

    def test_parser_lighten(self):
        st = StringIO()
        with redirect_stdout(st):
            get_parser_lighten().print_help()
        text = st.getvalue()
        self.assertIn("model", text)

    def test_parser_unlighten(self):
        st = StringIO()
        with redirect_stdout(st):
            get_parser_unlighten().print_help()
        text = st.getvalue()
        self.assertIn("model", text)

    def test_parser_print(self):
        st = StringIO()
        with redirect_stdout(st):
            get_parser_print().print_help()
        text = st.getvalue()
        self.assertIn("pretty", text)

    def test_parser_find(self):
        st = StringIO()
        with redirect_stdout(st):
            get_parser_find().print_help()
        text = st.getvalue()
        self.assertIn("names", text)

    def test_parser_config(self):
        st = StringIO()
        with redirect_stdout(st):
            get_parser_config().print_help()
        text = st.getvalue()
        self.assertIn("mid", text)

    def test_parser_validate(self):
        st = StringIO()
        with redirect_stdout(st):
            get_parser_validate().print_help()
        text = st.getvalue()
        self.assertIn("mid", text)

    def test_parser_validate_cmd(self):
        parser = get_parser_validate()
        args = parser.parse_args(
            [
                "-m",
                "arnir0/Tiny-LLM",
                "--run",
                "-v",
                "1",
                "--mop",
                "cache_implementation=static",
                "--iop",
                "cls_cache=StaticCache",
                "--patch",
            ]
        )
        self.assertEqual(args.mid, "arnir0/Tiny-LLM")
        self.assertEqual(args.run, True)
        self.assertEqual(args.patch, True)
        self.assertEqual(args.verbose, 1)
        self.assertEqual(args.mop, {"cache_implementation": "static"})
        self.assertEqual(args.iop, {"cls_cache": "StaticCache"})
        args = parser.parse_args(
            [
                "-m",
                "arnir0/Tiny-LLM",
                "--run",
                "-v",
                "1",
                "--mop",
                "cache_implementation=static",
                "--iop",
                "cls_cache=StaticCache",
                "--patch",
                "patch_sympy=False",
                "--patch",
                "patch_torch=False",
            ]
        )
        self.assertEqual(args.mid, "arnir0/Tiny-LLM")
        self.assertEqual(args.run, True)
        self.assertEqual(
            args.patch,
            {
                "patch_diffusers": True,
                "patch_sympy": False,
                "patch_torch": False,
                "patch_transformers": True,
            },
        )
        self.assertEqual(args.verbose, 1)
        self.assertEqual(args.mop, {"cache_implementation": "static"})
        self.assertEqual(args.iop, {"cls_cache": "StaticCache"})

    def test_parser_stats(self):
        st = StringIO()
        with redirect_stdout(st):
            get_parser_stats().print_help()
        text = st.getvalue()
        self.assertIn("input", text)

    def test_parser_agg(self):
        st = StringIO()
        with redirect_stdout(st):
            get_parser_agg().print_help()
        text = st.getvalue()
        self.assertIn("--recent", text)

    def test_parser_agg_cmd(self):
        parser = get_parser_agg()
        args = parser.parse_args(
            [
                "o.xlsx",
                "*.zip",
                "--sbs",
                "dynamo:exporter=onnx-dynamo,opt=ir,attn_impl=eager",
                "--sbs",
                "custom:exporter=custom,opt=default,attn_impl=eager",
            ]
        )
        self.assertEqual(
            args.sbs,
            {
                "custom": {"attn_impl": "eager", "exporter": "custom", "opt": "default"},
                "dynamo": {"attn_impl": "eager", "exporter": "onnx-dynamo", "opt": "ir"},
            },
        )

    def test_parser_sbs(self):
        st = StringIO()
        with redirect_stdout(st):
            get_parser_sbs().print_help()
        text = st.getvalue()
        self.assertIn("--onnx", text)

    def test_parser_dot(self):
        st = StringIO()
        with redirect_stdout(st):
            get_parser_dot().print_help()
        text = st.getvalue()
        self.assertIn("--run", text)

    def test_parser_compare(self):
        st = StringIO()
        with redirect_stdout(st):
            get_parser_compare().print_help()
        text = st.getvalue()
        self.assertIn("compare", text)

    def test_parser_optimize(self):
        st = StringIO()
        with redirect_stdout(st):
            get_parser_optimize().print_help()
        text = st.getvalue()
        self.assertIn("default", text)

    def test_parser_partition(self):
        st = StringIO()
        with redirect_stdout(st):
            get_parser_partition().print_help()
        text = st.getvalue()
        self.assertIn("regex", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
