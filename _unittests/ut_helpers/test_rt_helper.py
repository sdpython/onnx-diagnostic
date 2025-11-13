import os
import unittest
import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh
import torch
from onnx_diagnostic.ext_test_case import (
    ExtTestCase,
    has_onnxruntime_genai,
    hide_stdout,
    ignore_warnings,
    requires_torch,
    requires_transformers,
    skipif_ci_windows,
)
from onnx_diagnostic.helpers.rt_helper import (
    onnx_generate,
    generate_and_validate,
    onnx_generate_with_genai,
    name_type_to_onnx_dtype,
    js_profile_to_dataframe,
    plot_ort_profile_timeline,
    plot_ort_profile,
    _process_shape,
)
from onnx_diagnostic.torch_models.hghub import get_untrained_model_with_inputs
from onnx_diagnostic.torch_export_patches import torch_export_patches
from onnx_diagnostic.export.api import to_onnx


class TestRtSession(ExtTestCase):
    @requires_transformers("4.55")
    @requires_torch("2.9")
    @hide_stdout()
    @ignore_warnings(FutureWarning)
    def test_onnx_generate(self):
        mid = "arnir0/Tiny-LLM"
        print("-- test_onnx_generate: get model")
        data = get_untrained_model_with_inputs(mid)
        model, inputs, ds = data["model"], data["inputs"], data["dynamic_shapes"]
        configuration = data["configuration"]
        del inputs["position_ids"]
        del ds["position_ids"]
        input_ids = inputs["input_ids"]
        print("----", input_ids.shape)
        folder = self.get_dump_folder("test_onnx_generate")
        model_name = os.path.join(folder, "model.onnx")
        print("-- test_onnx_generate: export model")
        with torch_export_patches(patch_transformers=True, patch_torch=False):
            to_onnx(
                model,
                (),
                kwargs=inputs,
                dynamic_shapes=ds,
                filename=model_name,
                exporter="custom",
            )

        print("-- test_onnx_generate: generate")
        res, session, _feeds = onnx_generate(
            model_name, input_ids[:1], 2, max_new_tokens=10, return_session=True
        )
        n_inputs = input_ids.shape[1]
        self.assertEqualArray(input_ids[:1], res[:, :n_inputs])
        self.assertEqual(res.dtype, torch.int64)
        self.assertEqual(res.shape, (1, 13))
        print("-- test_onnx_generate: done")
        # expected = model.generate(input_ids[:1], max_new_tokens=10)
        expected, _ = generate_and_validate(
            model, input_ids[:1], 2, max_new_tokens=10, session=session
        )
        self.assertEqualArray(input_ids[:1], expected[:, :n_inputs])
        self.assertEqual(expected.dtype, torch.int64)
        self.assertEqual(expected.shape, (1, 13))
        self.assertEqualArray(expected, res)

        if not has_onnxruntime_genai():
            raise unittest.SkipTest("onnxruntime_genai is missing")

        res, session = onnx_generate_with_genai(
            model_name,
            input_ids[:1],
            max_new_tokens=10,
            return_session=True,
            transformers_config=configuration,
        )
        self.assertNotEmpty(session)
        self.assertEqualArray(expected, res)

    def test_name_type_to_onnx_dtype(self):
        for name in ["int64", "int32", "int64", "float16", "float", "double", "bfloat16"]:
            look = f"tensor({name})"
            expected = getattr(onnx.TensorProto, name.upper())
            self.assertEqual(expected, name_type_to_onnx_dtype(look))

    def test_shapes(self):
        tests = [
            (
                "U8[1x128x768]+F+U8",
                [{"uint8": [1, 128, 768]}, {"float": []}, {"uint8": []}],
            ),
            ("F[1x128x768]", [{"float": [1, 128, 768]}]),
        ]
        for expected, shapes in tests:
            with self.subTest(shapes=shapes):
                out = _process_shape(shapes)
                self.assertEqual(expected, out)

    def _get_model(self):
        model_def0 = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Add", ["X", "init1"], ["X1"]),
                    oh.make_node("Abs", ["X"], ["X2"]),
                    oh.make_node("Add", ["X", "init3"], ["inter"]),
                    oh.make_node("Mul", ["X1", "inter"], ["Xm"]),
                    oh.make_node("Sub", ["X2", "Xm"], ["final"]),
                ],
                "test",
                [oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [None])],
                [oh.make_tensor_value_info("final", onnx.TensorProto.FLOAT, [None])],
                [
                    onh.from_array(np.array([1], dtype=np.float32), name="init1"),
                    onh.from_array(np.array([3], dtype=np.float32), name="init3"),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        return model_def0

    def test_js_profile_to_dataframe(self):
        import onnxruntime

        sess_options = onnxruntime.SessionOptions()
        sess_options.enable_profiling = True
        sess = onnxruntime.InferenceSession(
            self._get_model().SerializeToString(),
            sess_options,
            providers=["CPUExecutionProvider"],
        )
        for _ in range(11):
            sess.run(None, dict(X=np.arange(10).astype(np.float32)))
        prof = sess.end_profiling()

        df = js_profile_to_dataframe(prof, first_it_out=True)
        self.assertEqual(df.shape, (79, 18))
        self.assertEqual(
            set(df.columns),
            {
                "cat",
                "pid",
                "tid",
                "dur",
                "ts",
                "ph",
                "name",
                "args_op_name",
                "op_name",
                "args_thread_scheduling_stats",
                "args_output_size",
                "args_parameter_size",
                "args_activation_size",
                "args_node_index",
                "args_provider",
                "event_name",
                "iteration",
                "it==0",
            },
        )

        df = js_profile_to_dataframe(prof, agg=True)
        self.assertEqual(df.shape, (9, 1))
        self.assertEqual(list(df.columns), ["dur"])

        df = js_profile_to_dataframe(prof, agg_op_name=True)
        self.assertEqual(df.shape, (79, 17))
        self.assertEqual(
            set(df.columns),
            {
                "cat",
                "pid",
                "tid",
                "dur",
                "ts",
                "ph",
                "name",
                "args_op_name",
                "op_name",
                "args_thread_scheduling_stats",
                "args_output_size",
                "args_parameter_size",
                "args_activation_size",
                "args_node_index",
                "args_provider",
                "event_name",
                "iteration",
            },
        )

        os.remove(prof)

    @ignore_warnings(UserWarning)
    @skipif_ci_windows("failing because of tkinter?")
    def test_plot_profile_2(self):
        import matplotlib.pyplot as plt
        import onnxruntime

        sess_options = onnxruntime.SessionOptions()
        sess_options.enable_profiling = True
        sess = onnxruntime.InferenceSession(
            self._get_model().SerializeToString(),
            sess_options,
            providers=["CPUExecutionProvider"],
        )
        for _ in range(11):
            sess.run(None, dict(X=np.arange(10).astype(np.float32)))
        prof = sess.end_profiling()

        df = js_profile_to_dataframe(prof, first_it_out=True)

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        plot_ort_profile(df, ax[0], ax[1], "test_title")
        # fig.savefig("graph1.png")
        self.assertNotEmpty(fig)

        os.remove(prof)

    @ignore_warnings(UserWarning)
    @skipif_ci_windows("failing because of tkinter?")
    def test_plot_profile_2_shape(self):
        import matplotlib.pyplot as plt
        import onnxruntime

        sess_options = onnxruntime.SessionOptions()
        sess_options.enable_profiling = True
        sess = onnxruntime.InferenceSession(
            self._get_model().SerializeToString(),
            sess_options,
            providers=["CPUExecutionProvider"],
        )
        for _ in range(11):
            sess.run(None, dict(X=np.arange(10).astype(np.float32)))
        prof = sess.end_profiling()

        df = js_profile_to_dataframe(prof, first_it_out=True, with_shape=True)

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        plot_ort_profile(df, ax[0], ax[1], "test_title")
        # fig.savefig("graph1.png")
        self.assertNotEmpty(fig)

        os.remove(prof)

    @ignore_warnings(UserWarning)
    @skipif_ci_windows("failing because of tkinter?")
    def test_plot_profile_agg(self):
        import matplotlib.pyplot as plt
        import onnxruntime

        sess_options = onnxruntime.SessionOptions()
        sess_options.enable_profiling = True
        sess = onnxruntime.InferenceSession(
            self._get_model().SerializeToString(),
            sess_options,
            providers=["CPUExecutionProvider"],
        )
        for _ in range(11):
            sess.run(None, dict(X=np.arange(10).astype(np.float32)))
        prof = sess.end_profiling()

        df = js_profile_to_dataframe(prof, first_it_out=True, agg=True)

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        plot_ort_profile(df, ax, title="test_title")
        fig.tight_layout()
        # fig.savefig("graph2.png")
        self.assertNotEmpty(fig)

        os.remove(prof)

    def _get_model2(self):
        model_def0 = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Add", ["X", "init1"], ["X1"]),
                    oh.make_node("Abs", ["X"], ["X2"]),
                    oh.make_node("Add", ["X", "init3"], ["inter"]),
                    oh.make_node("Mul", ["X1", "inter"], ["Xm"]),
                    oh.make_node("MatMul", ["X1", "Xm"], ["Xm2"]),
                    oh.make_node("Sub", ["X2", "Xm2"], ["final"]),
                ],
                "test",
                [oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [None, None])],
                [oh.make_tensor_value_info("final", onnx.TensorProto.FLOAT, [None, None])],
                [
                    onh.from_array(np.array([1], dtype=np.float32), name="init1"),
                    onh.from_array(np.array([3], dtype=np.float32), name="init3"),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        return model_def0

    @ignore_warnings(UserWarning)
    @skipif_ci_windows("failing because of tkinter?")
    def test_plot_profile_timeline(self):
        import matplotlib.pyplot as plt
        import onnxruntime

        sess_options = onnxruntime.SessionOptions()
        sess_options.enable_profiling = True
        sess = onnxruntime.InferenceSession(
            self._get_model2().SerializeToString(),
            sess_options,
            providers=["CPUExecutionProvider"],
        )
        for _ in range(11):
            sess.run(None, dict(X=np.random.rand(2**10, 2**10).astype(np.float32)))
        prof = sess.end_profiling()

        df = js_profile_to_dataframe(prof, first_it_out=True)

        fig, ax = plt.subplots(1, 1, figsize=(5, 10))
        plot_ort_profile_timeline(df, ax, title="test_timeline", quantile=0.5)
        fig.tight_layout()
        fig.savefig("test_plot_profile_timeline.png")
        self.assertNotEmpty(fig)

        os.remove(prof)


if __name__ == "__main__":
    import logging

    for name in [
        "matplotlib.font_manager",
        "PIL.PngImagePlugin",
        "matplotlib",
        "matplotlib.pyplot",
    ]:
        log = logging.getLogger(name)
        log.setLevel(logging.ERROR)
    unittest.main(verbosity=2)
