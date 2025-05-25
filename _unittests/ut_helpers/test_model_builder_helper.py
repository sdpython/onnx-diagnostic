import os
import unittest
from onnx_diagnostic.ext_test_case import (
    ExtTestCase,
    requires_torch,
    requires_transformers,
)
from onnx_diagnostic.helpers.model_builder_helper import (
    download_model_builder_to_cache,
    import_model_builder,
    create_model,
)
from onnx_diagnostic.torch_models.hghub import (
    get_untrained_model_with_inputs,
)


class TestModelBuilderHelper(ExtTestCase):
    # This is to limit impact on CI.
    @requires_transformers("4.52")
    @requires_torch("2.7.99")
    def test_download_model_builder(self):
        path = download_model_builder_to_cache()
        self.assertExists(path)
        builder = import_model_builder()
        self.assertHasAttr(builder, "create_model")

    # This is to limit impact on CI.
    @requires_transformers("4.52")
    @requires_torch("2.7.99")
    def test_model_builder_id(self):
        folder = self.get_dump_folder("test_model_builder_id")
        data = get_untrained_model_with_inputs("arnir0/Tiny-LLM")
        model = create_model(
            data["configuration"], precision="fp32", execution_provider="cpu", cache_dir=folder
        )
        self.assertGreater(len(model.nodes), 5)
        model.save_model(folder)
        self.assertExists(os.path.join(folder, "model.onnx"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
