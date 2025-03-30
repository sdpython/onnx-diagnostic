from typing import Any, Dict
import torch
import transformers
from ...helpers.cache_helper import make_dynamic_cache


def get_phi2(
    batch_size: int = 1,
    sequence_length: int = 30,
    sequence_length2: int = 3,
    dynamic_rope: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """
    Gets a non initialized model similar to :epkg:`microsoft/phi-2`.

    :param batch_size: batch size
    :param sequence_length: sequence length
    :param sequence_length2: new sequence length
    :param dynamic_rope: use dynamic rope (see :class:`transformers.LlamaConfig`)
    :param kwargs: to overwrite the configuration, example ``num_hidden_layers=1``
    :return: dictionary

    See :ref:`l-plot-tiny-llm-export-patched` for an example with a similar model.
    """
    config = {
        "_name_or_path": "microsoft/phi-2",
        "architectures": ["PhiForCausalLM"],
        "attention_dropout": 0.0,
        "bos_token_id": 50256,
        "embd_pdrop": 0.0,
        "eos_token_id": 50256,
        "hidden_act": "gelu_new",
        "hidden_size": 2560,
        "initializer_range": 0.02,
        "intermediate_size": 10240,
        "layer_norm_eps": 1e-05,
        "max_position_embeddings": 2048,
        "model_type": "phi",
        "num_attention_heads": 32,
        "num_hidden_layers": 32,
        "num_key_value_heads": 32,
        "partial_rotary_factor": 0.4,
        "qk_layernorm": False,
        "resid_pdrop": 0.1,
        "rope_scaling": {"rope_type": "dynamic", "factor": 10.0} if dynamic_rope else None,
        "rope_theta": 10000.0,
        "tie_word_embeddings": False,
        "torch_dtype": "float16",
        "transformers_version": "4.37.0",
        "use_cache": True,
        "vocab_size": 51200,
    }
    config.update(**kwargs)
    conf = transformers.PhiConfig(**config)
    model = transformers.PhiForCausalLM(conf)
    model.eval()

    # now the inputs
    cache_last_dim = 80
    max_token_id = config["vocab_size"] - 1
    n_layers = config["num_hidden_layers"]
    num_key_value_heads = config["num_key_value_heads"]

    batch = torch.export.Dim("batch", min=1, max=1024)
    seq_length = torch.export.Dim("seq_length", min=1, max=4096)
    cache_length = torch.export.Dim("cache_length", min=1, max=4096)

    shapes = {
        "input_ids": {0: batch, 1: seq_length},
        "position_ids": {
            0: batch,
            1: torch.export.Dim.DYNAMIC,  # cache_length + seq_length
        },
        "attention_mask": {
            0: batch,
            1: torch.export.Dim.DYNAMIC,  # cache_length + seq_length
        },
        "past_key_values": [
            [{0: batch, 2: cache_length} for _ in range(n_layers)],
            [{0: batch, 2: cache_length} for _ in range(n_layers)],
        ],
    }
    inputs = dict(
        input_ids=torch.randint(0, max_token_id, (batch_size, sequence_length2)).to(
            torch.int64
        ),
        attention_mask=torch.ones((batch_size, sequence_length + sequence_length2)).to(
            torch.int64
        ),
        position_ids=torch.arange(sequence_length, sequence_length + sequence_length2)
        .to(torch.int64)
        .expand((batch_size, -1)),
        past_key_values=make_dynamic_cache(
            [
                (
                    torch.randn(
                        batch_size, num_key_value_heads, sequence_length, cache_last_dim
                    ),
                    torch.randn(
                        batch_size, num_key_value_heads, sequence_length, cache_last_dim
                    ),
                )
                for i in range(n_layers)
            ]
        ),
    )
    return dict(inputs=inputs, model=model, dynamic_shapes=shapes, configuration=conf)
