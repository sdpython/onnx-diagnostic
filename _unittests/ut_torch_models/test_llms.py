import unittest
import torch
from onnx_diagnostic.ext_test_case import ExtTestCase, ignore_warnings
from onnx_diagnostic.torch_models.llms import get_tiny_llm
from onnx_diagnostic.helpers import string_type
from onnx_diagnostic.torch_export_patches import bypass_export_some_errors


class TestLlms(ExtTestCase):
    def test_get_tiny_llm(self):
        data = get_tiny_llm()
        model, inputs = data["model"], data["inputs"]
        self.assertIn("DynamicCache", string_type(inputs))
        model(**inputs)

    @ignore_warnings(UserWarning)
    def test_export_tiny_llm_1(self):
        data = get_tiny_llm()
        model, inputs = data["model"], data["inputs"]
        ep = torch.export.export(
            model, (), kwargs=inputs, dynamic_shapes=data["dynamic_shapes"]
        )
        assert ep
        print(ep)

    @ignore_warnings(UserWarning)
    def test_export_tiny_llm_2_bypassed(self):
        data = get_tiny_llm()
        model, inputs = data["model"], data["inputs"]
        with bypass_export_some_errors():
            ep = torch.export.export(
                model, (), kwargs=inputs, dynamic_shapes=data["dynamic_shapes"]
            )
            assert ep


if __name__ == "__main__":
    unittest.main(verbosity=2)
