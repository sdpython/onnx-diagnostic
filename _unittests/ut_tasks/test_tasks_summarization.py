import unittest
from onnx_diagnostic.ext_test_case import ExtTestCase, hide_stdout, requires_transformers
from onnx_diagnostic.torch_models.hghub.model_inputs import get_untrained_model_with_inputs


class TestTasksSummarization(ExtTestCase):
    @requires_transformers("4.51.999")
    @hide_stdout()
    def test_summarization(self):
        mid = "facebook/bart-large-cnn"
        data = get_untrained_model_with_inputs(mid, verbose=1, add_second_input=True)
        self.assertEqual(data["task"], "summarization")
        self.assertIn((data["size"], data["n_weights"]), [(1427701760, 356925440)])
        model, inputs, _ds = data["model"], data["inputs"], data["dynamic_shapes"]
        print(f"-- {mid}: {self.string_type(inputs, with_shape=True)}")
        model(**inputs)
        model(**data["inputs2"])
        # with torch_export_patches(patch_transformers=True, verbose=10):
        #    torch.export.export(
        #        model, (), kwargs=inputs, dynamic_shapes=use_dyn_not_str(ds), strict=False
        #    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
