import unittest
import os
import sys
import importlib.util
import subprocess
import time
import torch
from onnx_diagnostic import __file__ as onnx_diagnostic_file
from onnx_diagnostic.ext_test_case import (
    ExtTestCase,
    is_windows,
    has_transformers,
    has_torch,
    ignore_errors,
)


VERBOSE = 0
ROOT = os.path.realpath(os.path.abspath(os.path.join(onnx_diagnostic_file, "..", "..")))


def import_source(module_file_path, module_name):
    if not os.path.exists(module_file_path):
        raise FileNotFoundError(module_file_path)
    module_spec = importlib.util.spec_from_file_location(module_name, module_file_path)
    if module_spec is None:
        raise FileNotFoundError(
            "Unable to find '{}' in '{}'.".format(module_name, module_file_path)
        )
    module = importlib.util.module_from_spec(module_spec)
    return module_spec.loader.exec_module(module)


class TestDocumentationExamples(ExtTestCase):
    def run_test(self, fold: str, name: str, verbose=0) -> int:
        ppath = os.environ.get("PYTHONPATH", "")
        if not ppath:
            os.environ["PYTHONPATH"] = ROOT
        elif ROOT not in ppath:
            sep = ";" if is_windows() else ":"
            os.environ["PYTHONPATH"] = ppath + sep + ROOT
        perf = time.perf_counter()
        try:
            mod = import_source(fold, os.path.splitext(name)[0])
            assert mod is not None
        except FileNotFoundError:
            # try another way
            cmds = [sys.executable, "-u", os.path.join(fold, name)]
            p = subprocess.Popen(cmds, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            res = p.communicate()
            _out, err = res
            st = err.decode("ascii", errors="ignore")
            if st and "Traceback" in st:
                if '"dot" not found in path.' in st:
                    # dot not installed, this part
                    # is tested in onnx framework
                    raise unittest.SkipTest(f"failed: {name!r} due to missing dot.")
                if (
                    "We couldn't connect to 'https://huggingface.co'" in st
                    or "Cannot access content at: https://huggingface.co/" in st
                ):
                    raise unittest.SkipTest(f"Connectivity issues due to\n{err}")
                raise AssertionError(  # noqa: B904
                    "Example '{}' (cmd: {} - exec_prefix='{}') "
                    "failed due to\n{}"
                    "".format(name, cmds, sys.exec_prefix, st)
                )
        dt = time.perf_counter() - perf
        if verbose:
            print(f"{dt:.3f}: run {name!r}")
        return 1

    @classmethod
    def add_test_methods(cls):
        this = os.path.abspath(os.path.dirname(__file__))
        fold = os.path.normpath(os.path.join(this, "..", "..", "_doc", "examples"))
        found = os.listdir(fold)
        has_dot = int(os.environ.get("UNITTEST_DOT", "0"))
        for name in found:
            if not name.endswith(".py") or not name.startswith("plot_"):
                continue
            reason = None

            if not reason and not has_dot and name in {"plot_dump_intermediate_results.py"}:
                reason = "dot not installed"

            if (
                not reason
                and name in {"plot_export_tiny_llm.py"}
                and not has_transformers("4.51")
            ):
                reason = "transformers<4.51"

            if (
                not reason
                and name in {"plot_export_hub_codellama.py"}
                and not has_transformers("4.52")
            ):
                reason = "transformers<4.52"

            if (
                not reason
                and name
                in {
                    "plot_export_locate_issue.py",
                    "plot_export_with_auto.py",
                    "plot_export_hub_codellama.py",
                }
                and not has_torch("2.8")
            ):
                reason = "torch<2.8"

            if (
                not reason
                and name in {"plot_dump_intermediate_results.py"}
                and not has_torch("2.9.1")
            ):
                reason = "unstable, let's wait for the next version"

            if (
                not reason
                and name in {"plot_export_tiny_phi2.py"}
                and not has_transformers("4.55")
            ):
                reason = "unstable, let's wait for the next version"

            if not reason and name in {
                "plot_export_tiny_llm_dim01.py",
                "plot_export_tiny_llm_dim01_onnx.py",
                "plot_export_tiny_llm_dim01_onnx_custom.py",
            }:
                reason = "too long"

            if not reason and torch.__version__.startswith("2.9.0"):
                reason = "examples are failing for on CI for 2.9.0"

            if reason:

                @unittest.skip(reason)
                def _test_(self, name=name):
                    res = self.run_test(fold, name, verbose=VERBOSE)
                    self.assertTrue(res)

            else:

                @ignore_errors(OSError)  # connectivity issues
                def _test_(self, name=name):
                    res = self.run_test(fold, name, verbose=VERBOSE)
                    self.assertTrue(res)

            short_name = os.path.split(os.path.splitext(name)[0])[-1]
            setattr(cls, f"test_{short_name}", _test_)


TestDocumentationExamples.add_test_methods()

if __name__ == "__main__":
    unittest.main(verbosity=2)
