import io
import textwrap
import unittest
import pandas
from onnx_diagnostic.ext_test_case import ExtTestCase, hide_stdout
from onnx_diagnostic.helpers.log_helper import CubeLogs


class TestLogHelper(ExtTestCase):
    @hide_stdout()
    def test_cube_logs(self):
        df = pandas.read_csv(
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
        cube = CubeLogs(df)
        text = str(cube)
        self.assertIsInstance(text, str)
        self.assertRaise(lambda: cube.load(verbose=1), AssertionError)
        cube = CubeLogs(
            df,
            recent=True,
            formulas={"speedup": lambda df: df["time_baseline"] / df["time_baseline"]},
        )
        cube.load(verbose=1)
        text = str(cube)
        self.assertIsInstance(text, str)
        self.assertEqual((3, df.shape[1] + 1), cube.shape)
        self.assertEqual(set(cube.columns), {*df.columns, "speedup"})
        view = cube.view(["version.*", "model_name"], ["time_latency", "time_baseline"])
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
            ["version.*"], ["time_latency", "time_baseline"], order=["model_exporter"]
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
