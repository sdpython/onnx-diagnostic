import copy
import unittest
import packaging.version as pv
import onnx
import torch
from onnx_diagnostic.ext_test_case import (
    ExtTestCase,
    hide_stdout,
    ignore_warnings,
    requires_torch,
    requires_experimental,
    requires_onnxscript,
    requires_transformers,
)
from onnx_diagnostic.torch_models.validate import (
    get_inputs_for_task,
    validate_model,
    filter_inputs,
    run_ort_fusion,
    empty,
)
from onnx_diagnostic.tasks import supported_tasks


class TestValidateWholeModels(ExtTestCase):
    def test_a_get_inputs_for_task(self):
        fcts = supported_tasks()
        for task in self.subloop(sorted(fcts)):
            try:
                data = get_inputs_for_task(task)
            except NotImplementedError:
                continue
            self.assertIsInstance(data, dict)
            self.assertIn("inputs", data)
            self.assertIn("dynamic_shapes", data)
            copy.deepcopy(data["inputs"])

    def test_b_empty(self):
        self.assertFalse(empty("float16"))

    @hide_stdout()
    def test_c_validate_model(self):
        mid = "arnir0/Tiny-LLM"
        summary, data = validate_model(mid, do_run=True, verbose=2)
        self.assertIsInstance(summary, dict)
        self.assertIsInstance(data, dict)
        validate_model(mid, do_run=True, verbose=2, quiet=True)

    @hide_stdout()
    def test_d_validate_model_dtype(self):
        mid = "arnir0/Tiny-LLM"
        summary, data = validate_model(
            mid, do_run=True, verbose=2, dtype="float32", device="cpu"
        )
        self.assertIsInstance(summary, dict)
        self.assertIsInstance(data, dict)
        validate_model(mid, do_run=True, verbose=2, quiet=True)

    @hide_stdout()
    def test_e_validate_model_export(self):
        mid = "arnir0/Tiny-LLM"
        summary, data = validate_model(
            mid,
            do_run=True,
            verbose=10,
            exporter="export-nostrict",
            dump_folder="dump_test/validate_model_export",
            patch=True,
        )
        self.assertIsInstance(summary, dict)
        self.assertIsInstance(data, dict)

    @requires_torch("2.10.99")
    @requires_transformers("4.54")
    @hide_stdout()
    @ignore_warnings(FutureWarning)
    def test_f_validate_model_onnx_dynamo_ir(self):
        mid = "arnir0/Tiny-LLM"
        summary, data = validate_model(
            mid,
            do_run=True,
            verbose=10,
            exporter="onnx-dynamo",
            dump_folder="dump_test/validate_model_onnx_dynamo_ir",
            patch=True,
            stop_if_static=0,
            optimization="ir",
        )
        self.assertIsInstance(summary, dict)
        self.assertIsInstance(data, dict)
        self.assertLess(summary["disc_onnx_ort_run_abs"], 1e-4)
        onnx_filename = data["onnx_filename"]
        output_path = f"{onnx_filename}.ortopt.onnx"
        run_ort_fusion(
            onnx_filename, output_path, num_attention_heads=2, hidden_size=192, verbose=10
        )

    @requires_torch("2.7")
    @requires_onnxscript("0.7")
    @hide_stdout()
    @ignore_warnings(FutureWarning)
    def test_g_validate_model_onnx_dynamo_os_ort(self):
        mid = "arnir0/Tiny-LLM"
        summary, data = validate_model(
            mid,
            do_run=True,
            verbose=10,
            exporter="onnx-dynamo",
            dump_folder="dump_test/validate_model_onnx_dynamo_os_ort",
            patch=True,
            stop_if_static=2 if pv.Version(torch.__version__) > pv.Version("2.6.1") else 0,
            optimization="os_ort",
        )
        self.assertIsInstance(summary, dict)
        self.assertIsInstance(data, dict)
        self.assertLess(summary["disc_onnx_ort_run_abs"], 1e-4)
        onnx_filename = data["onnx_filename"]
        self.assertExists(onnx_filename)

    @requires_torch("2.7")
    @hide_stdout()
    @ignore_warnings(FutureWarning)
    @requires_experimental()
    def test_i_validate_model_custom(self):
        mid = "arnir0/Tiny-LLM"
        summary, data = validate_model(
            mid,
            do_run=True,
            verbose=10,
            exporter="custom",
            dump_folder="dump_test/validate_model_custom_tiny_llm",
            patch=True,
            stop_if_static=2 if pv.Version(torch.__version__) > pv.Version("2.6.1") else 0,
            optimization="default",
            quiet=False,
            repeat=2,
            warmup=1,
        )
        self.assertIsInstance(summary, dict)
        self.assertIsInstance(data, dict)
        self.assertIn("disc_onnx_ort_run_abs", summary)
        self.assertLess(summary["disc_onnx_ort_run_abs"], 1e-4)
        onnx_filename = data["onnx_filename"]
        output_path = f"{onnx_filename}.ortopt.onnx"
        run_ort_fusion(
            onnx_filename, output_path, num_attention_heads=2, hidden_size=192, verbose=10
        )

    @requires_torch("2.7")
    @hide_stdout()
    @ignore_warnings(FutureWarning)
    @requires_experimental()
    def test_j_validate_model_custom_torch(self):
        mid = "arnir0/Tiny-LLM"
        summary, data = validate_model(
            mid,
            do_run=True,
            verbose=10,
            exporter="custom-noinline",
            dump_folder="dump_test/validate_model_custom_torch",
            patch=True,
            stop_if_static=2 if pv.Version(torch.__version__) > pv.Version("2.6.1") else 0,
            optimization="default",
            quiet=False,
            runtime="torch",
        )
        self.assertIsInstance(summary, dict)
        self.assertIsInstance(data, dict)
        self.assertIn("disc_onnx_ort_run_abs", summary)
        self.assertLess(summary["disc_onnx_ort_run_abs"], 1e-4)

    def test_k_filter_inputs(self):
        inputs, ds = {"a": 1, "b": 2}, {"a": 20, "b": 30}
        ni, nd = filter_inputs(inputs, dynamic_shapes=ds, drop_names=["a"])
        self.assertEqual((ni, nd), ({"b": 2}, {"b": 30}))

        inputs, ds = (1, 2), {"a": 20, "b": 30}
        ni, nd = filter_inputs(inputs, dynamic_shapes=ds, drop_names=["b"], model=["a", "b"])
        self.assertEqual((ni, nd), ((1, None), {"a": 20}))

        inputs, ds = (1, 2), (20, 30)
        ni, nd = filter_inputs(inputs, dynamic_shapes=ds, drop_names=["b"], model=["a", "b"])
        self.assertEqual((ni, nd), ((1, None), (20, None)))

        inputs, ds = ((1,), {"b": 4}), {"a": 20, "b": 30}
        ni, nd = filter_inputs(inputs, dynamic_shapes=ds, drop_names=["b"], model=["a", "b"])
        self.assertEqual((ni, nd), ((1,), {"a": 20}))

        inputs, ds = ((1,), {"b": 4}), {"a": 20, "b": 30}
        ni, nd = filter_inputs(inputs, dynamic_shapes=ds, drop_names=["a"], model=["a", "b"])
        self.assertEqual((ni, nd), (((None,), {"b": 4}), {"b": 30}))

    @requires_torch("2.7")
    @hide_stdout()
    @ignore_warnings(FutureWarning)
    @requires_transformers("4.51")
    def test_l_validate_model_modelbuilder(self):
        mid = "microsoft/phi-2"
        summary, data = validate_model(
            mid,
            do_run=True,
            verbose=10,
            exporter="modelbuilder",
            dump_folder="dump_test/validate_model_modelbuilder",
            patch=True,
        )
        self.assertIsInstance(summary, dict)
        self.assertIsInstance(data, dict)
        self.assertLess(summary["disc_onnx_ort_run_abs"], 3e-2)
        onnx_filename = data["onnx_filename"]
        self.assertExists(onnx_filename)

    @requires_torch("2.7")
    @hide_stdout()
    @ignore_warnings(FutureWarning)
    @requires_transformers("4.51")
    def test_m_validate_model_vit_model(self):
        mid = "ydshieh/tiny-random-ViTForImageClassification"
        summary, data = validate_model(
            mid,
            do_run=True,
            verbose=10,
            exporter="onnx-dynamo",
            dump_folder="dump_test/validate_model_vit_model",
            inputs2=True,
        )
        self.assertIsInstance(summary, dict)
        self.assertIsInstance(data, dict)
        self.assertLess(summary["disc_onnx_ort_run_abs"], 1e-3)
        self.assertLess(summary["disc_onnx_ort_run22_abs"], 1e-3)
        self.assertEqual("dict(pixel_values:A1s2x3x30x30)", summary["run_feeds_inputs"])
        self.assertEqual("dict(pixel_values:A1s3x3x31x31)", summary["run_feeds_inputs2"])
        self.assertEqual("#1[A1s2x2]", summary["run_output_inputs"])
        self.assertEqual("#1[A1s3x2]", summary["run_output_inputs2"])
        onnx_filename = data["onnx_filename"]
        self.assertExists(onnx_filename)

    @requires_torch("2.9")
    @hide_stdout()
    @ignore_warnings(FutureWarning)
    @requires_transformers("4.55")
    def test_n_validate_phi35_mini_instruct(self):
        mid = "microsoft/Phi-3.5-mini-instruct"
        summary, data = validate_model(
            mid,
            do_run=True,
            verbose=10,
            exporter="custom",
            dump_folder="dump_test/validate_phi35_mini_instruct",
            inputs2=False,
            patch=True,
            rewrite=True,
            optimization="default",
        )
        self.assertIsInstance(summary, dict)
        self.assertIsInstance(data, dict)
        onnx_filename = data["onnx_filename"]
        onx = onnx.load(onnx_filename)
        op_types = set(n.op_type for n in onx.graph.node)
        self.assertIn("If", op_types)

    @requires_torch("2.9")
    @hide_stdout()
    @ignore_warnings(FutureWarning)
    @requires_transformers("4.55")
    def test_o_validate_phi35_4k_mini_instruct(self):
        mid = "microsoft/Phi-3-mini-4k-instruct"
        summary, data = validate_model(
            mid,
            do_run=True,
            verbose=10,
            exporter="custom",
            dump_folder="dump_test/validate_phi35_mini_instruct",
            inputs2=True,
            patch=True,
            rewrite=True,
            model_options={"rope_scaling": {"rope_type": "dynamic", "factor": 10.0}},
        )
        self.assertIsInstance(summary, dict)
        self.assertIsInstance(data, dict)
        onnx_filename = data["onnx_filename"]
        onx = onnx.load(onnx_filename)
        op_types = set(n.op_type for n in onx.graph.node)
        self.assertIn("If", op_types)


if __name__ == "__main__":
    unittest.main(verbosity=2)
