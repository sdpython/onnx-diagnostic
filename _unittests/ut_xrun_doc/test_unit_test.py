import math
import os
import unittest
import pandas
import onnx_diagnostic
from onnx_diagnostic.ext_test_case import (
    ExtTestCase,
    statistics_on_file,
    statistics_on_folder,
    is_apple,
    is_windows,
    is_azure,
    is_linux,
    unit_test_going,
    measure_time,
)


class TestUnitTest(ExtTestCase):
    def test_statistics_on_file(self):
        stat = statistics_on_file(__file__)
        self.assertEqual(stat["ext"], ".py")
        self.assertGreater(stat["lines"], 8)
        self.assertGreater(stat["chars"], stat["lines"])

    def test_statistics_on_folder(self):
        stat = statistics_on_folder(
            os.path.join(os.path.dirname(__file__), ".."), aggregation=1
        )
        self.assertGreater(len(stat), 1)

        df = pandas.DataFrame(stat)
        gr = df.drop("name", axis=1).groupby(["dir", "ext"]).sum()
        self.assertEqual(len(gr.columns), 2)

    def test_statistics_on_folders(self):
        stat = statistics_on_folder(
            [
                os.path.join(os.path.dirname(onnx_diagnostic.__file__)),
                os.path.join(os.path.dirname(onnx_diagnostic.__file__), "..", "_doc"),
                os.path.join(
                    os.path.dirname(onnx_diagnostic.__file__),
                    "..",
                    "_unittests",
                ),
            ],
            aggregation=2,
        )
        self.assertGreater(len(stat), 1)

        df = pandas.DataFrame(stat)
        gr = df.drop("name", axis=1).groupby(["ext", "dir"]).sum().reset_index()
        gr = gr[(gr["dir"] != "_doc/auto_examples") & (gr["dir"] != "_doc/auto_recipes")]
        total = (
            gr[gr["dir"].str.contains("onnx_diagnostic/")]
            .drop(["ext", "dir"], axis=1)
            .sum(axis=0)
        )
        self.assertEqual(len(gr.columns), 4)
        self.assertEqual(total.shape, (2,))

    def test_is(self):
        is_apple()
        is_windows()
        is_azure()
        is_linux()
        unit_test_going()

    def test_measure_time(self):
        res = measure_time(lambda: math.cos(0.5))
        self.assertIsInstance(res, dict)
        self.assertEqual(
            set(res),
            {
                "min_exec",
                "max_exec",
                "average",
                "warmup_time",
                "context_size",
                "deviation",
                "repeat",
                "ttime",
                "number",
            },
        )

    def test_measure_time_max(self):
        res = measure_time(lambda: math.cos(0.5), max_time=0.1)
        self.assertIsInstance(res, dict)
        self.assertEqual(
            set(res),
            {
                "min_exec",
                "max_exec",
                "average",
                "warmup_time",
                "context_size",
                "deviation",
                "repeat",
                "ttime",
                "number",
            },
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
