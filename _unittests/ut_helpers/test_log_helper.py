import io
import textwrap
import unittest
import pandas
from onnx_diagnostic.ext_test_case import ExtTestCase, hide_stdout
from onnx_diagnostic.helpers.log_helper import CubeLogs, CubeViewDef


class TestLogHelper(ExtTestCase):
    @classmethod
    def df1(cls):
        return pandas.read_csv(
            io.StringIO(
                textwrap.dedent(
                    """
                    date,version_python,version_transformers,model_name,model_exporter,time_load,time_latency,time_baseline,disc_ort,disc_ort2
                    2025/01/01,3.13.3,4.52.4,phi3,export,0.5,0.1,0.1,1e-5,1e-5
                    2025/01/02,3.13.3,4.52.4,phi3,export,0.6,0.11,0.1,1e-5,1e-5
                    2025/01/01,3.13.3,4.52.4,phi4,export,0.5,0.1,0.105,1e-5,1e-5
                    2025/01/01,3.12.3,4.52.4,phi4,onnx-dynamo,0.5,0.1,0.999,1e-5,1e-5
                    """
                )
            )
        )

    @classmethod
    def cube1(cls, verbose=0):
        cube = CubeLogs(
            cls.df1(),
            recent=True,
            formulas={"speedup": lambda df: df["time_baseline"] / df["time_baseline"]},
        )
        return cube.load(verbose=verbose)

    @hide_stdout()
    def test_cube_logs_load_df(self):
        df = self.df1()
        cube = CubeLogs(df)
        text = str(cube)
        self.assertIsInstance(text, str)
        self.assertRaise(lambda: cube.load(verbose=1), AssertionError)
        cube = CubeLogs(
            self.df1(),
            recent=True,
            formulas={"speedup": lambda df: df["time_baseline"] / df["time_baseline"]},
        )
        cube.load(verbose=1)
        text = str(cube)
        self.assertIsInstance(text, str)
        self.assertEqual((3, df.shape[1] + 1), cube.shape)
        self.assertEqual(set(cube.columns), {*df.columns, "speedup"})

    @hide_stdout()
    def test_cube_logs_load_list(self):
        cube = CubeLogs(
            [
                dict(
                    date="1/1/2001",
                    version_python="3.13",
                    model_exporter="A",
                    time_latency=5.6,
                ),
                dict(
                    date="1/1/2001",
                    version_python="3.13",
                    model_exporter="B",
                    time_latency=5.7,
                ),
            ]
        )
        cube.load(verbose=1)
        self.assertEqual((2, 4), cube.shape)

    def test_cube_logs_view_repr(self):
        v = CubeViewDef(["version.*", "model_name"], ["time_latency", "time_baseline"])
        r = repr(v)
        self.assertEqual(
            "CubeViewDef(key_index=['version.*', 'model_name'], "
            "values=['time_latency', 'time_baseline'])",
            r,
        )

    @hide_stdout()
    def test_cube_logs_view(self):
        cube = self.cube1(verbose=1)
        view = cube.view(
            CubeViewDef(["version.*", "model_name"], ["time_latency", "time_baseline"])
        )
        self.assertEqual((3, 4), view.shape)
        self.assertEqual(
            [
                ("time_baseline", "export"),
                ("time_baseline", "onnx-dynamo"),
                ("time_latency", "export"),
                ("time_latency", "onnx-dynamo"),
            ],
            list(view.columns),
        )
        self.assertEqual(
            [("3.12.3", "phi4"), ("3.13.3", "phi3"), ("3.13.3", "phi4")], list(view.index)
        )

        view = cube.view(
            CubeViewDef(
                ["version.*"], ["time_latency", "time_baseline"], order=["model_exporter"]
            )
        )
        self.assertEqual((2, 6), view.shape)
        self.assertEqual(
            [
                ("time_baseline", "export", "phi3"),
                ("time_baseline", "export", "phi4"),
                ("time_baseline", "onnx-dynamo", "phi4"),
                ("time_latency", "export", "phi3"),
                ("time_latency", "export", "phi4"),
                ("time_latency", "onnx-dynamo", "phi4"),
            ],
            list(view.columns),
        )
        self.assertEqual(["3.12.3", "3.13.3"], list(view.index))

    def test_cube_logs_view_agg(self):
        cube = self.cube1(verbose=0)
        view = cube.view(
            CubeViewDef(
                ["version.*", "model.*"],
                ["time_latency", "time_baseline"],
                key_agg=["model_name"],
            )
        )
        self.assertEqual((2, 2), view.shape)
        self.assertEqual(["time_baseline", "time_latency"], list(view.columns))
        self.assertEqual([("3.13.3", "export"), ("3.12.3", "onnx-dynamo")], list(view.index))

    @hide_stdout()
    def test_cube_logs_excel(self):
        output = self.get_dump_file("test_cube_logs_excel.xlsx")
        cube = self.cube1(verbose=0)
        cube.to_excel(
            output,
            {
                "example": CubeViewDef(
                    ["version.*", "model_name"], ["time_latency", "time_baseline"]
                ),
                "agg": CubeViewDef(
                    ["version.*", "model.*"],
                    ["time_latency", "time_baseline"],
                    key_agg=["model_name"],
                ),
            },
            verbose=1,
        )
        self.assertExists(output)


if __name__ == "__main__":
    unittest.main(verbosity=2)
