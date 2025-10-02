import os
import unittest
import torch
from onnx_diagnostic.ext_test_case import ExtTestCase, never_test
from onnx_diagnostic.helpers import string_type
from onnx_diagnostic.helpers.cache_helper import make_dynamic_cache, make_encoder_decoder_cache
from onnx_diagnostic.helpers.torch_helper import steal_forward
from onnx_diagnostic.torch_models.hghub.model_inputs import get_untrained_model_with_inputs
from onnx_diagnostic.torch_export_patches import torch_export_patches


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
    def test_text2text_generation_static(self):
        # clear&&NEVERTEST=1 python _unittests/ut_tasks/try_tasks.py -k text2t

        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        tokenizer = AutoTokenizer.from_pretrained("arnir0/Tiny-LLM")
        model = AutoModelForCausalLM.from_pretrained("arnir0/Tiny-LLM")

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
                input_ids=input_ids,
                attention_mask=mask,
                max_new_tokens=117,
                cache_implementation="static",
            )
        print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))

    @never_test()
    def test_text_generation_phi4_mini(self):
        # clear&&NEVERTEST=1 python _unittests/ut_tasks/try_tasks.py -k phi4_mini

        import torch
        from transformers import RobertaTokenizer, T5ForConditionalGeneration

        model_id = "microsoft/Phi-4-mini-instruct"
        tokenizer = RobertaTokenizer.from_pretrained(model_id)
        model = T5ForConditionalGeneration.from_pretrained(model_id)

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
    def test_text_generation_phi3_mini(self):
        # clear&&NEVERTEST=1 python _unittests/ut_tasks/try_tasks.py -k phi3_mini

        from transformers import Phi3ForCausalLM, AutoTokenizer

        model_id = "microsoft/Phi-3-mini-4k-instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = Phi3ForCausalLM.from_pretrained(model_id)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful digital assistant. Please provide safe, "
                    "ethical and accurate information to the user."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Can you provide ways to eat combinations of bananas and dragonfruits?"
                ),
            },
        ]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )

        # simply generate a single sequence
        print()
        with steal_forward(model):
            generated_ids = model.generate(inputs, max_length=100)
        print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))

    @never_test()
    @unittest.skip(
        reason="AttributeError: 'Phi4MMModel' object has no attribute "
        "'prepare_inputs_for_generation'"
    )
    def test_text_generation_phi4_moe(self):
        # clear&&NEVERTEST=1 python _unittests/ut_tasks/try_tasks.py -k phi4_moe

        import requests
        import io
        from PIL import Image
        import soundfile as sf
        from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
        from urllib.request import urlopen

        # Define model path
        model_path = "microsoft/Phi-4-multimodal-instruct"

        # Load model and processor
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="cuda",
            torch_dtype="auto",
            trust_remote_code=True,
            # if you do not use Ampere or later GPUs, change attention to "eager"
            # _attn_implementation='flash_attention_2',
            _attn_implementation="eager",
        ).cuda()

        # Load generation config
        generation_config = GenerationConfig.from_pretrained(model_path)

        # Define prompt structure
        user_prompt = "<|user|>"
        assistant_prompt = "<|assistant|>"
        prompt_suffix = "<|end|>"

        # Part 1: Image Processing
        print("\n--- IMAGE PROCESSING ---")
        image_url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        prompt = (
            f"{user_prompt}<|image_1|>What is shown in this image"
            f"?{prompt_suffix}{assistant_prompt}"
        )
        print(f">>> Prompt\n{prompt}")

        # Download and open image
        image = Image.open(requests.get(image_url, stream=True).raw)
        inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda:0")

        # Generate response
        print("--------- IMAGE PROCESSING ----------")
        print()
        with steal_forward(model):
            generate_ids = model.generate(
                **inputs,
                max_new_tokens=1000,
                generation_config=generation_config,
            )
        generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
        response = processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        print(f">>> Response\n{response}")

        # Part 2: Audio Processing
        print("\n--- AUDIO PROCESSING ---")
        audio_url = (
            "https://upload.wikimedia.org/wikipedia/commons/b/b0/"
            "Barbara_Sahakian_BBC_Radio4_The_Life_Scientific_29_May_2012_b01j5j24.flac"
        )
        speech_prompt = (
            "Transcribe the audio to text, and then translate the audio to French. "
            "Use <sep> as a separator between the original transcript and the translation."
        )
        prompt = f"{user_prompt}<|audio_1|>{speech_prompt}{prompt_suffix}{assistant_prompt}"
        print(f">>> Prompt\n{prompt}")

        # Download and open audio file
        audio, samplerate = sf.read(io.BytesIO(urlopen(audio_url).read()))

        # Process with the model
        inputs = processor(text=prompt, audios=[(audio, samplerate)], return_tensors="pt").to(
            "cuda:0"
        )

        print("--------- AUDIO PROCESSING ----------")
        print()
        with steal_forward(model):
            generate_ids = model.generate(
                **inputs,
                max_new_tokens=1000,
                generation_config=generation_config,
            )
        generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
        response = processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        print(f">>> Response\n{response}")

    @never_test()
    def test_imagetext2text_generation_idefics(self):
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
    def test_imagetext2text_generation_gemma3(self):
        """
        ::

            dict(input_ids:T7s1x281,
                pixel_values:T16s1x3x896x896,
                attention_mask:dict(full_attention:T9s1x1x281x380,sliding_attention:T9s1x1x281x380),
                position_ids:T7s1x281,
                past_key_values:HybridCache(
                    key_cache=#34[T1s1x4x380x256,...],
                    value_cache=#34[T1s1x4x380x256,...]),
                token_type_ids:T7s1x281,
                cache_position:T7s281,
                logits_to_keep:1)
            dict(input_ids:T7s1x1,
                pixel_values:None,
                attention_mask:dict(full_attention:T9s1x1x1x380,sliding_attention:T9s1x1x1x380),
                position_ids:T7s1x1,
                past_key_values:HybridCache(
                    key_cache=#34[T1s1x4x380x256,...],
                    value_cache=#34[T1s1x4x380x256,...]),
                token_type_ids:T7s1x1,
                cache_position:T7s1,
                logits_to_keep:1)
        """
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration
        import torch

        # model_id = "tiny-random/gemma-3"
        model_id = "google/gemma-3-4b-it"

        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id, device_map="auto"
        ).eval()

        processor = AutoProcessor.from_pretrained(model_id, use_fast=True)

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg",
                    },
                    {"type": "text", "text": "Describe this image in detail."},
                ],
            },
        ]

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]

        print()
        print(f"-- input_len={input_len}")
        # steal forward creates a bug...
        # with steal_forward(model), torch.inference_mode():
        with torch.inference_mode():
            generation = model.generate(**inputs, max_new_tokens=100, do_sample=False)
            generation = generation[0][input_len:]
        decoded = processor.decode(generation, skip_special_tokens=True)
        print(decoded)

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
    def test_feature_extraction(self):
        # clear&&NEVERTEST=1 python _unittests/ut_tasks/try_tasks.py -k feature_ex
        # https://huggingface.co/google-bert/bert-base-multilingual-cased

        from transformers import BartTokenizer, BartModel

        tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        model = BartModel.from_pretrained("facebook/bart-base")
        text = "Replace me by any text you'd like."
        encoded_input = tokenizer(text, return_tensors="pt")
        sequence_length, sequence_length2 = 30, 4
        sequence_length = 3
        batch_size, encoder_attention_heads, encoder_ffn_dim = 1, 12, 64
        batch_size, decoder_attention_heads, decoder_ffn_dim = 1, 12, 64
        num_hidden_layers = 6
        encoded_input["past_key_values"] = make_encoder_decoder_cache(
            make_dynamic_cache(
                [
                    (
                        torch.randn(
                            batch_size,
                            encoder_attention_heads,
                            sequence_length,
                            encoder_ffn_dim,
                        ),
                        torch.randn(
                            batch_size,
                            encoder_attention_heads,
                            sequence_length,
                            encoder_ffn_dim,
                        ),
                    )
                    for i in range(num_hidden_layers)
                ]
            ),
            make_dynamic_cache(
                [
                    (
                        torch.randn(
                            batch_size,
                            decoder_attention_heads,
                            sequence_length2,
                            decoder_ffn_dim,
                        ),
                        torch.randn(
                            batch_size,
                            decoder_attention_heads,
                            sequence_length2,
                            decoder_ffn_dim,
                        ),
                    )
                    for i in range(num_hidden_layers)
                ]
            ),
        )
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

    @never_test()
    def test_object_detection(self):
        # clear&&NEVERTEST=1 python _unittests/ut_tasks/try_tasks.py -k object_
        # https://huggingface.co/hustvl/yolos-tiny

        from transformers import YolosImageProcessor, YolosForObjectDetection
        from PIL import Image
        import torch
        import requests

        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)

        model = YolosForObjectDetection.from_pretrained("hustvl/yolos-tiny")
        image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")

        inputs = image_processor(images=image, return_tensors="pt")
        print()
        print("-- inputs", string_type(inputs, with_shape=True, with_min_max=True))
        outputs = model(**inputs)
        print("-- outputs", string_type(outputs, with_shape=True, with_min_max=True))

        # model predicts bounding boxes and corresponding COCO classes
        # logits = outputs.logits
        # bboxes = outputs.pred_boxes

        # print results
        target_sizes = torch.tensor([image.size[::-1]])
        results = image_processor.post_process_object_detection(
            outputs, threshold=0.9, target_sizes=target_sizes
        )[0]
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            print(
                f"Detected {model.config.id2label[label.item()]} with confidence "
                f"{round(score.item(), 3)} at location {box}"
            )

    @never_test()
    def test_text_to_image(self):
        # clear&&NEVERTEST=1 python _unittests/ut_tasks/try_tasks.py -k test_text_to_image
        import torch
        from diffusers import StableDiffusionPipeline

        model_id = "diffusers/tiny-torch-full-checker"  # "stabilityai/stable-diffusion-2"
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(
            "cuda"
        )

        prompt = "a photo of an astronaut riding a horse on mars and on jupyter"
        print()
        with steal_forward(pipe.unet, with_min_max=True):
            image = pipe(prompt).images[0]
        print("-- output", self.string_type(image, with_shape=True, with_min_max=True))
        # stolen forward for class UNet2DConditionModel -- iteration 44
        # sample=T10s2x4x96x96[-3.7734375,4.359375:A-0.043463995395642184]
        # time_step=T7s=101
        # encoder_hidden_states:T10s2x77x1024[-6.58203125,13.0234375:A-0.16780663634440257]

    @never_test()
    def test_imagetext2text_generation_zai_glm(self):
        """
        clear&&NEVERTEST=1 python _unittests/ut_tasks/try_tasks.py -k zai_glm
        """
        from transformers import AutoProcessor

        model_id = "zai-org/GLM-4.5V"
        data = get_untrained_model_with_inputs(model_id, verbose=1, add_second_input=True)
        model = data["model"]
        processor = AutoProcessor.from_pretrained(model_id, use_fast=True)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "url": "http://images.cocodataset.org/val2017/000000039769.jpg",
                    },
                    {"type": "text", "text": "describe this image"},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        inputs.pop("token_type_ids", None)

        print()
        # steal forward creates a bug...
        with steal_forward(model):  # , torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=8192)
        output_text = processor.decode(
            generated_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=False
        )
        print(output_text)

    @never_test()
    def test_sentence_similary_alibaba_nlp_gte(self):
        """
        clear&&NEVERTEST=1 python _unittests/ut_tasks/try_tasks.py -k alibaba
        """
        import torch.nn.functional as F
        from transformers import AutoModel, AutoTokenizer

        input_texts = [
            "what is the capital of China?",
            "how to implement quick sort in python?",
            "Beijing",
            "sorting algorithms",
        ]

        model_path = "Alibaba-NLP/gte-large-en-v1.5"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True)

        # Tokenize the input texts
        batch_dict = tokenizer(
            input_texts, max_length=8192, padding=True, truncation=True, return_tensors="pt"
        )

        print("-- type:", type(model))
        print("-- subclasses:", type(model).__subclasses__())
        print("-- inputs:", self.string_type(batch_dict, with_shape=True))
        outputs = model(**batch_dict)
        print("-- outputs:", self.string_type(outputs, with_shape=True))
        embeddings = outputs.last_hidden_state[:, 0]

        # (Optionally) normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        scores = (embeddings[:1] @ embeddings[1:].T) * 100
        print(scores.tolist())

    @never_test()
    def test_imagetext2text_generation_gemma3_4b_it(self):
        """
        clear&&NEVERTEST=1 python _unittests/ut_tasks/try_tasks.py -k gemma3_4b_it
        """
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration

        model_id = "google/gemma-3-4b-it"
        if os.environ.get("PRETRAINED", ""):
            model = Gemma3ForConditionalGeneration.from_pretrained(
                model_id, device_map="cpu"
            ).eval()
        else:
            data = get_untrained_model_with_inputs(
                model_id,
                verbose=1,
                add_second_input=False,
                # same_as_pretrained=True, #use_pretrained=True
                inputs_kwargs={
                    "sequence_length": 281,
                    "batch_size": 1,
                    "max_sequence_length": 580,
                    "n_images": 1,
                },
            )
            model = data["model"]

        print(f"-- model.device={model.device}")
        processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
        print(f"-- processor={type(processor)}")

        messages = messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg",
                    },
                    {"type": "text", "text": "Describe this image in detail."},
                ],
            },
        ]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device, dtype=torch.bfloat16)
        # if "token_type_ids" in inputs:
        #    print(
        #       f"-- remove token_type_ids: "
        #       f"{self.string_type(inputs['token_type_ids'], with_shape=True)}"
        #    )
        # inputs.pop("token_type_ids", None)
        print(f"-- inputs={self.string_type(inputs)}")

        # iteration merge = sequence > 1, cache not empty
        # iteration 1 = sequence > 1, no cache
        #   cache_position:T7s281,
        #   past_key_values:StaticCache(key_cache=#0[], value_cache=#0[]),
        #   input_ids:T7s1x281,
        #   inputs_embeds:None,
        #   token_type_ids:T7s1x281,
        #   attention_mask:dict(sliding_attention:T9s1x1x281x580,
        #                       full_attention:T9s1x1x281x580),
        #   position_ids:None,
        #   use_cache:bool,
        #   logits_to_keep:None,
        #   pixel_values:T16s1x3x896x896,
        #   return_dict:bool)
        # iteration 2 = sequence = 1, cache not empty
        #   cache_position:T7s1,
        #   past_key_values:StaticCache(key_cache=#34[T1s1x4x580x256,...],
        #                               value_cache=#34[T1s1x4x580x256,...]),
        #   input_ids:T7s1x1,
        #   inputs_embeds:None,
        #   token_type_ids:T7s1x1,
        #   attention_mask:dict(sliding_attention:T9s1x1x1x580,full_attention:T9s1x1x1x580),
        #   position_ids:None,
        #   use_cache:bool,logits_to_keep:None,return_dict:bool)

        print()
        with torch_export_patches(
            patch_torch=False, patch_sympy=False, patch_transformers=True
        ), steal_forward(
            model,
            dump_file=self.get_dump_file("test_imagetext2text_generation_gemma3_4b_it.onnx"),
            dump_drop={"attention_mask", "past_key_values", "pixel_values"},
            save_as_external_data=False,
        ):
            generated_ids = model.generate(
                **inputs,
                # 282 = value high enough to trigger multiple iterations of the model
                max_new_tokens=282,
                do_sample=False,
                cache_implementation="static",
            )
        output_text = processor.decode(
            generated_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=False
        )
        print(output_text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
