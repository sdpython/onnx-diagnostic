import unittest
from onnx_diagnostic.ext_test_case import (
    ExtTestCase,
    ignore_errors,
    requires_torch,
    requires_transformers,
    hide_stdout,
)
from onnx_diagnostic.helpers.model_builder_helper import (
    download_model_builder_to_cache,
    import_model_builder,
    create_model_builder,
    save_model_builder,
)
from onnx_diagnostic.torch_models.hghub import (
    get_untrained_model_with_inputs,
)
from onnx_diagnostic.helpers.rt_helper import make_feeds


class TestModelBuilderHelper(ExtTestCase):
    # This is to limit impact on CI.
    @requires_transformers("4.52")
    @requires_torch("2.7.99")
    @ignore_errors(OSError)  # connectivity issues
    def test_download_model_builder(self):
        path = download_model_builder_to_cache()
        self.assertExists(path)
        builder = import_model_builder()
        self.assertHasAttr(builder, "create_model")

    # This is to limit impact on CI.
    @requires_transformers("4.52")
    @requires_torch("2.7.99")
    @hide_stdout()
    @ignore_errors(OSError)  # connectivity issues
    def test_model_builder_id(self):
        # clear&&python ~/.cache/onnx-diagnostic/builder.py
        # --model arnir0/Tiny-LLM -p fp16 -c dump_cache -e cpu -o dump_model
        folder = self.get_dump_folder("test_model_builder_id")
        data = get_untrained_model_with_inputs("arnir0/Tiny-LLM")
        onnx_model = create_model_builder(
            data["configuration"],
            data["model"],
            precision="fp32",
            execution_provider="cpu",
            cache_dir=folder,
            verbose=1,
        )
        self.assertGreater(onnx_model.model.graph.num_nodes(), 5)
        model_name = save_model_builder(onnx_model, folder, verbose=1)
        self.assertExists(model_name)

        import onnxruntime

        sess = onnxruntime.InferenceSession(model_name, providers=["CPUExecutionProvider"])
        del data["inputs"]["position_ids"]
        feeds = make_feeds([i.name for i in sess.get_inputs()], data["inputs"], use_numpy=True)
        expected = data["model"](**data["inputs"])

        try:
            got = sess.run(None, feeds)
        except onnxruntime.capi.onnxruntime_pybind11_state.InvalidArgument as e:
            if "batch_size must be 1 when sequence_length > 1" in str(e):
                raise unittest.SkipTest("batch_size must be 1 when sequence_length > 1")
        self.assertEqualAny(expected, got)


if __name__ == "__main__":
    unittest.main(verbosity=2)
