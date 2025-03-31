-m onnx_diagnostic config ... prints the config for a model id
==============================================================

Description
+++++++++++

The command lines prints out the configuration file a model id
available on :epkg:`HuggingFace`.

.. runpython::

    from onnx_diagnostic._command_lines_parser import get_parser_config

    get_parser_config().print_help()

Example
+++++++

.. code-block:: bash

    python -m onnx_diagnostic config HuggingFaceM4/tiny-random-idefics

.. code-block:: text

    IdeficsConfig {
    "additional_vocab_size": 2,
    "alpha_initializer": "ones",
    "alpha_type": "vector",
    "alphas_initializer_range": 0.0,
    "architectures": [
        "IdeficsForVisionText2Text"
    ],
    "bos_token_id": 1,
    "cross_layer_activation_function": "swiglu",
    "cross_layer_interval": 1,
    "dropout": 0.0,
    "eos_token_id": 2,
    "ffn_dim": 64,
    "freeze_lm_head": false,
    "freeze_text_layers": false,
    "freeze_text_module_exceptions": [],
    "freeze_vision_layers": false,
    "freeze_vision_module_exceptions": [],
    "hidden_act": "silu",
    "hidden_size": 16,
    "initializer_range": 0.02,
    "intermediate_size": 11008,
    "max_new_tokens": 128,
    "max_position_embeddings": 128,
    "model_type": "idefics",
    "num_attention_heads": 4,
    "num_hidden_layers": 2,
    "pad_token_id": 0,
    "perceiver_config": {
        "model_type": "idefics_perciever",
        "qk_layer_norms_perceiver": false,
        "resampler_depth": 2,
        "resampler_head_dim": 8,
        "resampler_n_heads": 2,
        "resampler_n_latents": 16,
        "use_resampler": false
    },
    "qk_layer_norms": false,
    "rms_norm_eps": 1e-06,
    "tie_word_embeddings": false,
    "torch_dtype": "float16",
    "transformers_version": "4.51.0.dev0",
    "use_cache": true,
    "use_resampler": true,
    "vision_config": {
        "attention_dropout": 0.0,
        "embed_dim": 32,
        "hidden_act": "gelu",
        "image_size": 30,
        "initializer_factor": 1.0,
        "initializer_range": 0.02,
        "intermediate_size": 37,
        "layer_norm_eps": 1e-05,
        "model_type": "idefics_vision",
        "num_attention_heads": 4,
        "num_channels": 3,
        "num_hidden_layers": 5,
        "patch_size": 2,
        "vision_model_name": "hf-internal-testing/tiny-random-clip"
    },
    "vocab_size": 32000,
    "word_embed_proj_dim": 16
    }
