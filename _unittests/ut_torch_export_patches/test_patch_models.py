import unittest
import torch
from onnx_diagnostic.ext_test_case import ExtTestCase, hide_stdout, requires_transformers
from onnx_diagnostic.helpers.torch_helper import torch_deepcopy
from onnx_diagnostic.torch_models.hghub.model_inputs import get_untrained_model_with_inputs
from onnx_diagnostic.torch_export_patches import torch_export_patches
from onnx_diagnostic.torch_export_patches.patch_inputs import use_dyn_not_str


class TestHuggingFaceHubModel(ExtTestCase):
    @hide_stdout()
    @requires_transformers("4.51")
    def test_patch_eager_mask_open_whisper_tiny(self):
        mid = "openai/whisper-tiny"
        data = get_untrained_model_with_inputs(mid, verbose=1)
        model, inputs, ds = data["model"], data["inputs"], data["dynamic_shapes"]
        model(**torch_deepcopy(inputs))
        with torch_export_patches(patch_transformers=True, verbose=1):
            torch.export.export(model, (), kwargs=inputs, dynamic_shapes=use_dyn_not_str(ds))


if __name__ == "__main__":
    unittest.main(verbosity=2)
