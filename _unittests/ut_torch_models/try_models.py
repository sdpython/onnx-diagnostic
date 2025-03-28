import unittest
from onnx_diagnostic.ext_test_case import ExtTestCase, never_test
from onnx_diagnostic.helpers import string_type


class TestHuggingFaceHubModel(ExtTestCase):
    @never_test()
    def test_image_classiciation(self):

        from transformers import ViTImageProcessor, ViTModel
        from PIL import Image
        import requests

        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)

        processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        inputs = processor(images=image, return_tensors="pt")
        print("inputs", string_type(inputs, with_shape=True, with_min_max=True))

        outputs = model(**inputs)
        print("outputs", string_type(outputs, with_shape=True, with_min_max=True))


if __name__ == "__main__":
    unittest.main(verbosity=2)
