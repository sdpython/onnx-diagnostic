import unittest
from onnx_diagnostic.ext_test_case import ExtTestCase, never_test
from onnx_diagnostic.helpers import string_type
from onnx_diagnostic.helpers.torch_test_helper import steel_forward


class TestHuggingFaceHubModel(ExtTestCase):
    @never_test()
    def test_image_classification(self):
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

        import torch
        from transformers import RobertaTokenizer, T5ForConditionalGeneration

        tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-small")
        model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-small")

        text = "def greet(user): print(f'hello <extra_id_0>!')"
        input_ids = tokenizer(text, return_tensors="pt").input_ids
        mask = (
            torch.tensor([1 for i in range(input_ids.shape[1])])
            .to(torch.int64)
            .reshape((1, -1))
        )

        # simply generate a single sequence
        print()
        with steel_forward(model):
            generated_ids = model.generate(
                decoder_input_ids=input_ids, attention_mask=mask, max_length=100
            )
        print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))

    @never_test()
    def test_imagetext2text_generation(self):
        # clear&&NEVERTEST=1 python _unittests/ut_torch_models/try_tasks.py -k etext2t
        # https://huggingface.co/docs/transformers/main/en/tasks/idefics

        import torch
        from transformers import IdeficsForVisionText2Text, AutoProcessor

        mid = "HuggingFaceM4/tiny-random-idefics"
        processor = AutoProcessor.from_pretrained(mid)
        model = IdeficsForVisionText2Text.from_pretrained(
            mid, torch_dtype=torch.bfloat16, device_map="auto"
        )

        prompt = [
            "https://images.unsplash.com/photo-1583160247711-2191776b4b91?ixlib=rb-4.0.3"
            "&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=3542&q=80",
        ]
        inputs = processor(text=prompt, return_tensors="pt").to("cuda")
        bad_words_ids = processor.tokenizer(
            ["<image>", "<fake_token_around_image>"], add_special_tokens=False
        ).input_ids
        print()
        with steel_forward(model):
            generated_ids = model.generate(
                **inputs, max_new_tokens=10, bad_words_ids=bad_words_ids
            )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)

        print(generated_text[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
