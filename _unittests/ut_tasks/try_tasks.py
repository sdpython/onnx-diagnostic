import unittest
from onnx_diagnostic.ext_test_case import ExtTestCase, never_test
from onnx_diagnostic.helpers import string_type
from onnx_diagnostic.helpers.torch_test_helper import steal_forward


class TestHuggingFaceHubModel(ExtTestCase):
    @never_test()
    def test_image_classification(self):
        # clear&&NEVERTEST=1 python _unittests/ut_tasks/try_tasks.py -k image_c

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
    def test_image_classification_resnet(self):
        # clear&&NEVERTEST=1 python _unittests/ut_tasks/try_tasks.py -k resnet

        from transformers import ViTImageProcessor, ViTModel
        from PIL import Image
        import requests

        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)

        processor = ViTImageProcessor.from_pretrained("microsoft/resnet-50")
        model = ViTModel.from_pretrained("microsoft/resnet-50")
        inputs = processor(images=image, return_tensors="pt")
        print()
        print("-- inputs", string_type(inputs, with_shape=True, with_min_max=True))

        outputs = model(**inputs)
        print("-- outputs", string_type(outputs, with_shape=True, with_min_max=True))

    @never_test()
    def test_zero_shot_image_classification(self):
        # clear&&NEVERTEST=1 python _unittests/ut_tasks/try_tasks.py -k zero
        from PIL import Image
        import requests
        from transformers import CLIPProcessor, CLIPModel

        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        inputs = processor(
            text=["a photo of a cat", "a photo of a dog"],
            images=[image, image],
            return_tensors="pt",
            padding=True,
        )
        print()
        print("-- inputs", string_type(inputs, with_shape=True, with_min_max=True))
        outputs = model(**inputs)
        print("-- outputs", string_type(outputs, with_shape=True, with_min_max=True))
        logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        probs = logits_per_image.softmax(
            dim=1
        )  # we can take the softmax to get the label probabilities
        assert probs is not None

    @never_test()
    def test_text2text_generation(self):
        # clear&&NEVERTEST=1 python _unittests/ut_tasks/try_tasks.py -k text2t

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
        with steal_forward(model):
            generated_ids = model.generate(
                decoder_input_ids=input_ids, attention_mask=mask, max_length=100
            )
        print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))

    @never_test()
    def test_text_generation_phi4(self):
        # clear&&NEVERTEST=1 python _unittests/ut_tasks/try_tasks.py -k phi4

        import torch
        from transformers import RobertaTokenizer, T5ForConditionalGeneration

        tokenizer = RobertaTokenizer.from_pretrained("microsoft/Phi-4-mini-instruct")
        model = T5ForConditionalGeneration.from_pretrained("microsoft/Phi-4-mini-instruct")

        text = "def greet(user): print(f'hello <extra_id_0>!')"
        input_ids = tokenizer(text, return_tensors="pt").input_ids
        mask = (
            torch.tensor([1 for i in range(input_ids.shape[1])])
            .to(torch.int64)
            .reshape((1, -1))
        )

        # simply generate a single sequence
        print()
        with steal_forward(model):
            generated_ids = model.generate(
                decoder_input_ids=input_ids, attention_mask=mask, max_length=100
            )
        print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))

    @never_test()
    def test_imagetext2text_generation(self):
        # clear&&NEVERTEST=1 python _unittests/ut_tasks/try_tasks.py -k etext2t
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
        with steal_forward(model):
            generated_ids = model.generate(
                **inputs, max_new_tokens=10, bad_words_ids=bad_words_ids
            )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)

        print(generated_text[0])

    @never_test()
    def test_automatic_speech_recognition(self):
        # clear&&NEVERTEST=1 python _unittests/ut_tasks/try_tasks.py -k automatic_speech
        # https://huggingface.co/openai/whisper-tiny

        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        from datasets import load_dataset

        """
        kwargs=dict(
            cache_position:T7s4,
            past_key_values:EncoderDecoderCache(
                self_attention_cache=DynamicCache[serialized](#2[#0[],#0[]]),
                cross_attention_cache=DynamicCache[serialized](#2[#0[],#0[]])
            ),
            decoder_input_ids:T7s1x4,
            encoder_outputs:dict(last_hidden_state:T1s1x1500x384),
            use_cache:bool,return_dict:bool
        )
        kwargs=dict(
            cache_position:T7s1,
            past_key_values:EncoderDecoderCache(
                self_attention_cache=DynamicCache[serialized](#2[
                    #4[T1s1x6x4x64,T1s1x6x4x64,T1s1x6x4x64,T1s1x6x4x64],
                    #4[T1s1x6x4x64,T1s1x6x4x64,T1s1x6x4x64,T1s1x6x4x64]
                ]),
                cross_attention_cache=DynamicCache[serialized](#2[
                    #4[T1s1x6x1500x64,T1s1x6x1500x64,T1s1x6x1500x64,T1s1x6x1500x64],
                    #4[T1s1x6x1500x64,T1s1x6x1500x64,T1s1x6x1500x64,T1s1x6x1500x64]
                ]),
            ),
            decoder_input_ids:T7s1x1,
            encoder_outputs:dict(last_hidden_state:T1s1x1500x384),
            use_cache:bool,return_dict:bool
        )
        """

        # load model and processor
        processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
        forced_decoder_ids = processor.get_decoder_prompt_ids(
            language="english", task="transcribe"
        )

        # load streaming dataset and read first audio sample
        ds = load_dataset(
            "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
        )
        sample = ds[0]["audio"]
        input_features = processor(
            sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt"
        ).input_features

        # generate token ids
        print()
        with steal_forward(model.model.decoder):
            predicted_ids = model.generate(
                input_features, forced_decoder_ids=forced_decoder_ids
            )

        # decode token ids to text
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=False)
        print("--", transcription)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        print("--", transcription)

    @never_test()
    def test_fill_mask(self):
        # clear&&NEVERTEST=1 python _unittests/ut_tasks/try_tasks.py -k fill
        # https://huggingface.co/google-bert/bert-base-multilingual-cased

        from transformers import BertTokenizer, BertModel

        tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
        model = BertModel.from_pretrained("bert-base-multilingual-cased")
        text = "Replace me by any text you'd like."
        encoded_input = tokenizer(text, return_tensors="pt")
        print()
        print("-- inputs", string_type(encoded_input, with_shape=True, with_min_max=True))
        output = model(**encoded_input)
        print("-- outputs", string_type(output, with_shape=True, with_min_max=True))

    @never_test()
    def test_text_classification(self):
        # clear&&NEVERTEST=1 python _unittests/ut_tasks/try_tasks.py -k text_cl
        # https://huggingface.co/Intel/bert-base-uncased-mrpc

        from transformers import BertTokenizer, BertModel

        tokenizer = BertTokenizer.from_pretrained("Intel/bert-base-uncased-mrpc")
        model = BertModel.from_pretrained("Intel/bert-base-uncased-mrpc")
        text = "The inspector analyzed the soundness in the building."
        encoded_input = tokenizer(text, return_tensors="pt")
        print()
        print("-- inputs", string_type(encoded_input, with_shape=True, with_min_max=True))
        output = model(**encoded_input)
        print("-- outputs", string_type(output, with_shape=True, with_min_max=True))
        # print BaseModelOutputWithPoolingAndCrossAttentions and  pooler_output

        # Print tokens * ids in of inmput string below
        print("Tokenized Text: ", tokenizer.tokenize(text), "\n")
        print("Token IDs: ", tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text)))

        # Print tokens in text
        encoded_input["input_ids"][0]
        tokenizer.convert_ids_to_tokens(encoded_input["input_ids"][0])

    @never_test()
    def test_sentence_similary(self):
        # clear&&NEVERTEST=1 python _unittests/ut_tasks/try_tasks.py -k ce_sim
        # https://huggingface.co/sentence-transformers/all-MiniLM-L6-v1

        from transformers import AutoTokenizer, AutoModel
        import torch
        import torch.nn.functional as F

        # Mean Pooling - Take attention mask into account for correct averaging
        def mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[
                0
            ]  # First element of model_output contains all token embeddings
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
                input_mask_expanded.sum(1), min=1e-9
            )

        # Sentences we want sentence embeddings for
        sentences = ["This is an example sentence", "Each sentence is converted"]

        # Load model from HuggingFace Hub
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v1")
        model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v1")

        # Tokenize sentences
        encoded_input = tokenizer(
            sentences, padding=True, truncation=True, return_tensors="pt"
        )

        # Compute token embeddings
        with torch.no_grad():
            print()
            print("-- inputs", string_type(encoded_input, with_shape=True, with_min_max=True))
            model_output = model(**encoded_input)
            print("-- outputs", string_type(model_output, with_shape=True, with_min_max=True))

        # Perform pooling
        sentence_embeddings = mean_pooling(model_output, encoded_input["attention_mask"])

        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        print("Sentence embeddings:")
        print(sentence_embeddings)

    @never_test()
    def test_falcon_mamba_dev(self):
        # clear&&NEVERTEST=1 python _unittests/ut_tasks/try_tasks.py -k falcon_mamba_dev
        # https://huggingface.co/tiiuae/falcon-mamba-tiny-dev

        from transformers import AutoTokenizer
        import transformers
        import torch

        model = "tiiuae/falcon-mamba-tiny-dev"

        tokenizer = AutoTokenizer.from_pretrained(model)
        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )
        print()
        with steal_forward(pipeline.model):
            sequences = pipeline(
                "Girafatron is obsessed with giraffes, "
                "the most glorious animal on the face of this Earth. "
                "Giraftron believes all other animals are irrelevant "
                "when compared to the glorious majesty of the giraffe."
                "\nDaniel: Hello, Girafatron!\nGirafatron:",
                max_length=200,
                do_sample=True,
                top_k=10,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
            )
        for seq in sequences:
            print(f"Result: {seq['generated_text']}")

    @never_test()
    def test_falcon_mamba_7b(self):
        # clear&&NEVERTEST=1 python _unittests/ut_tasks/try_tasks.py -k falcon_mamba_7b
        # https://huggingface.co/tiiuae/falcon-mamba-7b

        from transformers import AutoTokenizer
        import transformers
        import torch

        model = "tiiuae/falcon-mamba-7b"

        tokenizer = AutoTokenizer.from_pretrained(model)
        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )
        print()
        with steal_forward(pipeline.model):
            sequences = pipeline(
                "Girafatron is obsessed with giraffes, "
                "the most glorious animal on the face of this Earth. "
                "Giraftron believes all other animals are irrelevant "
                "when compared to the glorious majesty of the giraffe."
                "\nDaniel: Hello, Girafatron!\nGirafatron:",
                max_length=200,
                do_sample=True,
                top_k=10,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
            )
        for seq in sequences:
            print(f"Result: {seq['generated_text']}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
