import unittest
from onnx_diagnostic.ext_test_case import ExtTestCase, hide_stdout
from onnx_diagnostic.ci_models.export_qwen25_vl import main as main_qwen25


class TestCiExport(ExtTestCase):
    @hide_stdout()
    def test_main_qwen25(self):
        main_qwen25(
            model_id="arnir0/Tiny-LLM",
            device="cpu",
            dtype="float32",
            exporter="custom",
            pretrained=False,
            part="",
            output_folder=self.get_dump_folder("test_main_qwen25"),
            make_zip=True,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
