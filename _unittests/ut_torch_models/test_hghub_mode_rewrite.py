import unittest
import torch
from onnx_diagnostic.ext_test_case import (
    ExtTestCase,
    hide_stdout,
    ignore_errors,
    requires_torch,
)
from onnx_diagnostic.torch_export_patches.patch_module_helper import code_needing_rewriting
from onnx_diagnostic.torch_models.hghub.model_inputs import get_untrained_model_with_inputs
from onnx_diagnostic.torch_export_patches import torch_export_patches
from onnx_diagnostic.torch_export_patches.patch_inputs import use_dyn_not_str


class TestHuggingFaceHubModelRewrite(ExtTestCase):
    def test_code_needing_rewriting(self):
        self.assertEqual(2, len(code_needing_rewriting("BartForConditionalGeneration")))

    @hide_stdout()
    @ignore_errors(OSError)
    @requires_torch("2.8")
    def test_export_rewriting_bart(self):
        mid = "hf-tiny-model-private/tiny-random-PLBartForConditionalGeneration"
        data = get_untrained_model_with_inputs(mid, verbose=1)
        model, inputs, ds = data["model"], data["inputs"], data["dynamic_shapes"]
        dump_folder = self.get_dump_file("test_export_rewritin_bart")
        print(self.string_type(inputs))
        print(self.string_type(ds))
        with torch_export_patches(
            patch_transformers=True, rewrite=model, dump_rewriting=dump_folder
        ):
            model(**inputs)
            torch.export.export(model, (), kwargs=inputs, dynamic_shapes=use_dyn_not_str(ds))


if __name__ == "__main__":
    unittest.main(verbosity=2)
