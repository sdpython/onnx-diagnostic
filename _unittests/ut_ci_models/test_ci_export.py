import unittest
from onnx_diagnostic.ext_test_case import (
    ExtTestCase,
    hide_stdout,
    requires_transformers,
    never_test,
)
from onnx_diagnostic.ci_models.export_qwen25_vl import main as main_qwen25


class TestCiExport(ExtTestCase):
    @hide_stdout()
    @requires_transformers("4.55")
    def test_main_qwen25_tiny_llm(self):
        main_qwen25(
            model_id="arnir0/Tiny-LLM",
            device="cpu",
            dtype="float32",
            exporter="custom",
            pretrained=False,
            part="",
            output_folder=self.get_dump_folder("test_main_qwen25_tiny_llm"),
            opset=24,
        )
        self.clean_dump()

    @never_test()
    @hide_stdout()
    @requires_transformers("4.57")
    def test_main_qwen25_embedding(self):
        main_qwen25(
            model_id="Qwen/Qwen2.5-VL-7B-Instruct",
            device="cpu",
            dtype="float32",
            exporter="custom",
            pretrained=False,
            output_folder=self.get_dump_folder("test_main_qwen25_embedding"),
            make_zip=True,
            part="embedding",
            second_input=True,
        )
        self.clean_dump()


if __name__ == "__main__":
    unittest.main(verbosity=2)
