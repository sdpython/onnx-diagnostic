from typing import Any, Dict
import transformers


def get_tiny_llm(
    batch_size: int = 2,
    sequence_length: int = 30,
    sequence_length2: int = 3,
    dynamic_rope: bool = False,
    use_static_cache: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """
    Gets a non initialized model similar to :epkg:`arnir0/Tiny-LLM`.

    :param batch_size: batch size
    :param sequence_length: sequence length
    :param sequence_length2: new sequence length
    :param dynamic_rope: use dynamic rope (see :class:`transformers.LlamaConfig`)
    :param use_static_cache: use StaticCache instead of DynamicCache
    :param kwargs: to overwrite the configuration, example ``num_hidden_layers=1``
    :return: dictionary

    See :ref:`l-plot-tiny-llm-export` or :ref:`l-plot-tiny-llm-export-patched` for examples.
    """
    from ...tasks.text_generation import get_inputs

    config = {
        "architectures": ["LlamaForCausalLM"],
        "bos_token_id": 1,
        "eos_token_id": 2,
        "hidden_act": "silu",
        "hidden_size": 192,
        "initializer_range": 0.02,
        "intermediate_size": 1024,
        "max_position_embeddings": 1024,
        "model_type": "llama",
        "num_attention_heads": 2,
        "num_hidden_layers": 1,
        "num_key_value_heads": 1,
        "pretraining_tp": 1,
        "rms_norm_eps": 1e-05,
        "rope_scaling": {"rope_type": "dynamic", "factor": 10.0} if dynamic_rope else None,
        "tie_word_embeddings": False,
        "torch_dtype": "float32",
        "transformers_version": "4.31.0.dev0",
        "use_cache": True,
        "vocab_size": 32000,
    }

    config.update(**kwargs)
    conf = transformers.LlamaConfig(**config)
    conf.cache_implementation = "static"
    model = transformers.LlamaForCausalLM(conf)
    model.eval()

    res = get_inputs(
        model,
        conf,
        dummy_max_token_id=config["vocab_size"],  # type: ignore[arg-type]
        num_hidden_layers=config["num_hidden_layers"],  # type: ignore[arg-type]
        batch_size=batch_size,
        sequence_length=sequence_length,
        sequence_length2=sequence_length2,
        dynamic_rope=dynamic_rope,
        num_key_value_heads=config["num_key_value_heads"],  # type: ignore[arg-type]
        cls_cache="StaticCache" if use_static_cache else "DynamicCache",
    )

    return dict(
        inputs=res["inputs"],
        model=model,
        dynamic_shapes=res["dynamic_shapes"],
        configuration=conf,
    )
