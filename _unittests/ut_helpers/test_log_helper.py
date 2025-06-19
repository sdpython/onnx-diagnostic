import io
import os
import textwrap
import unittest
import zipfile
import numpy as np
import pandas
from onnx_diagnostic.ext_test_case import ExtTestCase, hide_stdout
from onnx_diagnostic.helpers.log_helper import (
    CubeLogs,
    CubeLogsPerformance,
    CubeViewDef,
    enumerate_csv_files,
    open_dataframe,
)


class TestLogHelper(ExtTestCase):
    @classmethod
    def df1(cls):
        return pandas.read_csv(
            io.StringIO(
                textwrap.dedent(
                    """
                    date,version_python,version_transformers,model_name,model_exporter,time_load,time_latency,time_baseline,disc_ort,disc_ort2
                    2025/01/01,3.13.3,4.52.4,phi3,export,0.51,0.1,0.1,1e-5,1e-5
                    2025/01/02,3.13.3,4.52.4,phi3,export,0.62,0.11,0.11,1e-5,1e-5
                    2025/01/01,3.13.3,4.52.4,phi4,export,0.53,0.1,0.105,1e-5,1e-5
                    2025/01/01,3.12.3,4.52.4,phi4,onnx-dynamo,0.54,0.14,0.999,1e-5,1e-5
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
    def test_cube_logs_load_dfdf(self):
        df = self.df1()
        cube = CubeLogs([df, df], recent=True)
        cube.load(verbose=1)
        self.assertEqual((3, 10), cube.shape)

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
            CubeViewDef(
                ["version.*", "model_name"],
                ["time_latency", "time_baseline"],
                ignore_columns=["date"],
            )
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
                ["version.*"],
                ["time_latency", "time_baseline"],
                order=["model_exporter"],
                ignore_columns=["date"],
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
                key_agg=["model_name", "date"],
                ignore_columns=["version_python"],
            )
        )
        self.assertEqual((4, 1), view.shape)
        self.assertEqual(["VALUE"], list(view.columns))
        self.assertEqual(
            [
                ("export", "time_baseline"),
                ("export", "time_latency"),
                ("onnx-dynamo", "time_baseline"),
                ("onnx-dynamo", "time_latency"),
            ],
            list(view.index),
        )

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

    @hide_stdout()
    def test_enumerate_csv_files(self):
        df = self.df1()
        filename = self.get_dump_file("test_enumerate_csv_files.csv")
        df.to_csv(filename, index=False)
        zip_file = self.get_dump_file("test_enumerate_csv_files.zip")
        with zipfile.ZipFile(zip_file, "w", zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(filename)

        dirname = os.path.dirname(filename)
        data = [os.path.join(dirname, "*.csv"), os.path.join(dirname, "*.zip")]
        dfs = list(enumerate_csv_files(data, verbose=1))
        self.assertNotEmpty(dfs)
        for df in dfs:
            open_dataframe(df)

        cube = CubeLogs(data, recent=True)
        cube.load(verbose=1)
        self.assertEqual((3, 11), cube.shape)
        self.assertIn("RAWFILENAME", cube.data.columns)

    def test_cube_logs_performance(self):
        output = self.get_dump_file("test_cube_logs_performance.xlsx")
        filename = os.path.join(os.path.dirname(__file__), "data", "data-agg.zip")
        assert list(enumerate_csv_files(filename))
        dfs = [open_dataframe(df) for df in enumerate_csv_files(filename)]
        assert dfs, f"{filename!r} empty"
        cube = CubeLogsPerformance(dfs)
        cube.load()
        cube.to_excel(
            output,
            views=[
                "agg-suite",
                "disc",
                "speedup",
                "counts",
                "time",
                "time_export",
                "err",
                # "cmd",
                "bucket-speedup",
                "raw-short",
            ],
        )
        self.assertExists(output)

    def test_duplicate(self):
        df = pandas.DataFrame(
            [
                dict(date="2025/01/01", time_engine=0.5, model_name="A", version_engine="0.5"),
                dict(date="2025/01/01", time_engine=0.5, model_name="A", version_engine="0.5"),
            ]
        )
        cube = CubeLogs(df)
        self.assertRaise(lambda: cube.load(), AssertionError)
        CubeLogs(df, recent=True).load()

    def test_historical(self):
        # case 1
        df = pandas.DataFrame(
            [
                dict(date="2025/01/01", time_p=0.51, exporter="E1", m_name="A", m_cls="CA"),
                dict(date="2025/01/02", time_p=0.62, exporter="E1", m_name="A", m_cls="CA"),
                dict(date="2025/01/01", time_p=0.53, exporter="E2", m_name="A", m_cls="CA"),
                dict(date="2025/01/02", time_p=0.64, exporter="E2", m_name="A", m_cls="CA"),
                dict(date="2025/01/01", time_p=0.55, exporter="E2", m_name="B", m_cls="CA"),
                dict(date="2025/01/02", time_p=0.66, exporter="E2", m_name="B", m_cls="CA"),
            ]
        )
        cube = CubeLogs(df, keys=["^m_*", "exporter"]).load()
        view, view_def = cube.view(CubeViewDef(["^m_.*"], ["^time_.*"]), return_view_def=True)
        self.assertEqual(
            "CubeViewDef(key_index=['^m_.*'], values=['^time_.*'])", repr(view_def)
        )
        self.assertEqual(["METRICS", "exporter", "date"], view.columns.names)
        got = view.values.ravel()
        self.assertEqual(
            sorted([0.51, 0.62, 0.53, 0.64, -1, -1, 0.55, 0.66]),
            sorted(np.where(np.isnan(got), -1, got).tolist()),
        )

        # case 2
        df = pandas.DataFrame(
            [
                dict(date="2025/01/02", time_p=0.62, exporter="E1", m_name="A", m_cls="CA"),
                dict(date="2025/01/02", time_p=0.64, exporter="E2", m_name="A", m_cls="CA"),
                dict(date="2025/01/01", time_p=0.51, exporter="E1", m_name="B", m_cls="CA"),
                dict(date="2025/01/02", time_p=0.66, exporter="E2", m_name="B", m_cls="CA"),
            ]
        )
        cube = CubeLogs(df, keys=["^m_*", "exporter"]).load()
        view, view_def = cube.view(CubeViewDef(["^m_.*"], ["^time_.*"]), return_view_def=True)
        self.assertEqual((2, 3), view.shape)
        self.assertEqual(["METRICS", "exporter", "date"], view.columns.names)


if __name__ == "__main__":
    unittest.main(verbosity=2)
