import unittest
import pandas
from onnx_diagnostic.ext_test_case import ExtTestCase, requires_transformers
from onnx_diagnostic.investigate.input_observer import InputObserver
from onnx_diagnostic.torch_export_patches import (
    register_additional_serialization_functions,
    torch_export_patches,
)
from onnx_diagnostic.export.api import to_onnx
from onnx_diagnostic.torch_models.hghub import get_untrained_model_with_inputs
from onnx_diagnostic.helpers.rt_helper import onnx_generate


class TestInputObserverTransformers(ExtTestCase):
    @requires_transformers("4.57")
    def test_input_observer_onnx_generate_tiny_llm(self):
        mid = "arnir0/Tiny-LLM"
        print("-- test_onnx_generate: get model")
        data = get_untrained_model_with_inputs(mid)
        model, inputs, _ds = data["model"], data["inputs"], data["dynamic_shapes"]
        input_ids = inputs["input_ids"][:1]
        attention_mask = inputs["attention_mask"][:1]

        observer = InputObserver()
        with (
            register_additional_serialization_functions(patch_transformers=True),
            observer(model),
        ):
            outputs = model.generate(
                input_ids=input_ids, attention_mask=attention_mask, do_sample=False
            )

        filenamec = self.get_dump_file("test_input_observer_onnx_generate_tiny_llm.onnx")
        with torch_export_patches(patch_transformers=True):
            to_onnx(
                model,
                (),
                kwargs=observer.infer_arguments(),
                dynamic_shapes=observer.infer_dynamic_shapes(set_batch_dimension_for=True),
                filename=filenamec,
                exporter="custom",
            )

        data = observer.check_discrepancies(filenamec, progress_bar=False)
        df = pandas.DataFrame(data)
        self.assertLess(df["abs"].max(), 1e-4)

        onnx_tokens = onnx_generate(
            filenamec,
            input_ids=input_ids,
            attention_mask=attention_mask,
            eos_token_id=model.config.eos_token_id,
            max_new_tokens=20,
        )
        self.assertEqualArray(outputs, onnx_tokens)


if __name__ == "__main__":
    unittest.main(verbosity=2)
