import unittest
import torch
from onnx_diagnostic.ext_test_case import ExtTestCase, hide_stdout
from onnx_diagnostic.helpers import max_diff
from onnx_diagnostic.helpers.torch_helper import torch_deepcopy
from onnx_diagnostic.helpers.rt_helper import make_feeds
from onnx_diagnostic.torch_models.hghub import get_untrained_model_with_inputs
from onnx_diagnostic.torch_export_patches import torch_export_patches
from onnx_diagnostic.export.api import to_onnx


class TestValidate(ExtTestCase):
    @hide_stdout()
    def test_to_onnx(self):
        class Model(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        x = torch.randn((5, 6))
        y = torch.randn((1, 6))
        ds = ({0: "a", 1: "b"}, {1: "b"})
        to_onnx(
            Model(),
            (x, y),
            dynamic_shapes=ds,
            exporter="custom",
            filename=self.get_dump_file("to_onnx_custom.onnx"),
        )
        to_onnx(
            Model(),
            (x, y),
            dynamic_shapes=ds,
            exporter="onnx-dynamo",
            filename=self.get_dump_file("to_onnx_onnx-dynamo.onnx"),
        )

    @hide_stdout()
    def test_tiny_llm_to_onnx(self):
        import onnxruntime

        data = get_untrained_model_with_inputs("arnir0/Tiny-LLM")
        model, inputs, ds = data["model"], data["inputs"], data["dynamic_shapes"]
        b1 = data["inputs_batch1"]
        filenames = {
            "custom": self.get_dump_file("test_tiny_llm_to_onnx-custom.onnx"),
            "onnx-dynamo": self.get_dump_file("test_tiny_llm_to_onnx-dynamo.onnx"),
            "modelbuilder": self.get_dump_file("model.onnx"),
        }
        del inputs["position_ids"]
        del ds["position_ids"]
        del b1["position_ids"]

        expected = model(**torch_deepcopy(b1))

        with torch_export_patches(patch_transformers=True):
            for exporter, filename in filenames.items():
                with self.subTest(exporter=exporter):
                    to_onnx(
                        model,
                        kwargs=inputs,
                        dynamic_shapes=ds,
                        exporter=exporter,
                        filename=filename,
                    )
        for exporter, filename in filenames.items():
            with self.subTest(exporter=f"validate-{exporter}"):
                sess = onnxruntime.InferenceSession(
                    filename, providers=["CPUExecutionProvider"]
                )
                feeds = make_feeds(sess, b1, use_numpy=True)
                got = sess.run(None, feeds)
                diff = max_diff(expected, got)
                assert diff["abs"] <= 1e-5, f"diff={diff}"

        b1["attention_mask"][:, :] = 1
        expected = model(**torch_deepcopy(b1))
        for exporter, filename in filenames.items():
            with self.subTest(exporter=f"full-mask-{exporter}"):
                sess = onnxruntime.InferenceSession(
                    filename, providers=["CPUExecutionProvider"]
                )
                feeds = make_feeds(sess, b1, use_numpy=True)
                got = sess.run(None, feeds)
                diff = max_diff(expected, got)
                assert diff["abs"] <= 1e-5, f"diff={diff}"


if __name__ == "__main__":
    unittest.main(verbosity=2)
