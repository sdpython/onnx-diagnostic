from typing import Any, Callable, Dict, Optional, Tuple
import torch
from ..helpers.cache_helper import make_dynamic_cache
from ..helpers.config_helper import update_config, check_hasattr, _pick

__TASK__ = "text-generation"


def reduce_model_config(config: Any, task: str) -> Dict[str, Any]:
    """Reduces a model size."""
    check_hasattr(
        config,
        ("head_dim", ("hidden_size", "num_attention_heads")),
        "num_hidden_layers",
        ("num_key_value_heads", "num_attention_heads"),
        "intermediate_size",
        "hidden_size",
    )
    kwargs = dict(
        head_dim=getattr(config, "head_dim", config.hidden_size // config.num_attention_heads),
        num_hidden_layers=min(config.num_hidden_layers, 2),
        num_key_value_heads=(
            config.num_key_value_heads
            if hasattr(config, "num_key_value_heads")
            else config.num_attention_heads
        ),
        intermediate_size=(
            min(config.intermediate_size, 24576 // 4)
            if config.intermediate_size % 4 == 0
            else config.intermediate_size
        ),
        hidden_size=(
            min(config.hidden_size, 3072 // 4)
            if config.hidden_size % 4 == 0
            else config.hidden_size
        ),
    )
    update_config(config, kwargs)
    return kwargs


def get_inputs(
    model: torch.nn.Module,
    config: Optional[Any],
    dummy_max_token_id: int,
    num_key_value_heads: int,
    num_hidden_layers: int,
    head_dim: int,
    batch_size: int = 2,
    sequence_length: int = 30,
    sequence_length2: int = 3,
    dynamic_rope: bool = False,
    **kwargs,  # unused
):
    """
    Generates input for task ``text-generation``.

    :param model: model to get the missing information
    :param config: configuration used to generate the model
    :param head_dim: last dimension of the cache
    :param dummy_max_token_id: dummy max token id
    :param batch_size: batch size
    :param sequence_length: sequence length
    :param sequence_length2: new sequence length
    :param dynamic_rope: use dynamic rope (see :class:`transformers.LlamaConfig`)
    :return: dictionary
    """
    if head_dim is None:
        assert config, "head_dim is None, the value cannot be set without a configuration"
        head_dim = config.hidden_size // config.num_attention_heads
    batch = torch.export.Dim("batch", min=1, max=1024)
    seq_length = torch.export.Dim("seq_length", min=1, max=4096)
    cache_length = torch.export.Dim("cache_length", min=1, max=4096)

    shapes = {
        "input_ids": {0: batch, 1: seq_length},
        "attention_mask": {
            0: batch,
            1: torch.export.Dim.DYNAMIC,  # cache_length + seq_length
        },
        "position_ids": {
            0: batch,
            1: torch.export.Dim.DYNAMIC,  # cache_length + seq_length
        },
        "past_key_values": [
            [{0: batch, 2: cache_length} for _ in range(num_hidden_layers)],
            [{0: batch, 2: cache_length} for _ in range(num_hidden_layers)],
        ],
    }
    inputs = dict(
        input_ids=torch.randint(0, dummy_max_token_id, (batch_size, sequence_length2)).to(
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
                    torch.randn(batch_size, num_key_value_heads, sequence_length, head_dim),
                    torch.randn(batch_size, num_key_value_heads, sequence_length, head_dim),
                )
                for i in range(num_hidden_layers)
            ]
        ),
    )
    return dict(inputs=inputs, dynamic_shapes=shapes)


def random_input_kwargs(config: Any, task: str) -> Tuple[Dict[str, Any], Callable]:
    """
    Inputs kwargs.

    If the configuration is None, the function selects typical dimensions.
    """
    if config is not None:
        check_hasattr(
            config,
            "vocab_size",
            "hidden_size",
            "num_attention_heads",
            ("num_key_value_heads", "num_attention_heads"),
            "intermediate_size",
            "hidden_size",
        )
    kwargs = dict(
        batch_size=2,
        sequence_length=30,
        sequence_length2=3,
        head_dim=(
            16
            if config is None
            else getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        ),
        dummy_max_token_id=31999 if config is None else (config.vocab_size - 1),
        num_hidden_layers=4 if config is None else config.num_hidden_layers,
        num_key_value_heads=(
            24
            if config is None
            else _pick(config, "num_key_value_heads", "num_attention_heads")
        ),
        intermediate_size=1024 if config is None else config.intermediate_size,
        hidden_size=512 if config is None else config.hidden_size,
    )
    return kwargs, get_inputs
