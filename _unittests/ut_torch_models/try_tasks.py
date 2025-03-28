import unittest
from onnx_diagnostic.ext_test_case import ExtTestCase, never_test
from onnx_diagnostic.helpers import string_type
from onnx_diagnostic.torch_test_helper import steel_forward


class TestHuggingFaceHubModel(ExtTestCase):
    @never_test()
    def test_image_classiciation(self):
        # clear&&NEVERTEST=1 python _unittests/ut_torch_models/try_tasks.py -k image_c

        from transformers import ViTImageProcessor, ViTModel
        from PIL import Image
        import requests

        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)

        processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        inputs = processor(images=image, return_tensors="pt")
        print()
        print("-- inputs", string_type(inputs, with_shape=True, with_min_max=True))

        outputs = model(**inputs)
        print("-- outputs", string_type(outputs, with_shape=True, with_min_max=True))

    @never_test()
    def test_text2text_generation(self):
        # clear&&NEVERTEST=1 python _unittests/ut_torch_models/try_tasks.py -k text2t

        from transformers import RobertaTokenizer, T5ForConditionalGeneration

        tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-small")
        model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-small")

        text = "def greet(user): print(f'hello <extra_id_0>!')"
        input_ids = tokenizer(text, return_tensors="pt").input_ids

        # simply generate a single sequence
        print()
        print("-- inputs", string_type(input_ids, with_shape=True, with_min_max=True))
        with steel_forward(model):
            generated_ids = model.generate(input_ids, max_length=100)
        print("-- outputs", string_type(generated_ids, with_shape=True, with_min_max=True))
        print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))


if __name__ == "__main__":
    unittest.main(verbosity=2)
