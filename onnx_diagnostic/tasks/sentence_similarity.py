from typing import Any, Callable, Dict, Optional, Tuple
import torch
from ..helpers.config_helper import update_config, check_hasattr

__TASK__ = "sentence-similarity"


def reduce_model_config(config: Any) -> Dict[str, Any]:
    """Reduces a model size."""
    check_hasattr(config, "num_attention_heads", "num_hidden_layers")
    kwargs = dict(
        num_hidden_layers=min(config.num_hidden_layers, 2),
        num_attention_heads=min(config.num_attention_heads, 4),
    )
    update_config(config, kwargs)
    return kwargs


def get_inputs(
    model: torch.nn.Module,
    config: Optional[Any],
    batch_size: int,
    sequence_length: int,
    dummy_max_token_id: int,
    add_second_input: bool = False,
    **kwargs,  # unused
):
    """
    Generates inputs for task ``sentence-similarity``.
    Example:

    ::

        input_ids:T7s1x13[101,72654:A16789.23076923077],
        token_type_ids:T7s1x13[0,0:A0.0],
        attention_mask:T7s1x13[1,1:A1.0])
    """
    batch = torch.export.Dim("batch", min=1, max=1024)
    seq_length = "seq_length"
    shapes = {
        "input_ids": {0: batch, 1: seq_length},
        "token_type_ids": {0: batch, 1: seq_length},
        "attention_mask": {0: batch, 1: seq_length},
    }
    inputs = dict(
        input_ids=torch.randint(0, dummy_max_token_id, (batch_size, sequence_length)).to(
            torch.int64
        ),
        token_type_ids=torch.zeros((batch_size, sequence_length)).to(torch.int64),
        attention_mask=torch.ones((batch_size, sequence_length)).to(torch.int64),
    )
    res = dict(inputs=inputs, dynamic_shapes=shapes)
    if add_second_input:
        res["inputs2"] = get_inputs(
            model=model,
            config=config,
            batch_size=batch_size + 1,
            sequence_length=sequence_length + 1,
            dummy_max_token_id=dummy_max_token_id,
            **kwargs,
        )["inputs"]
    return res


def random_input_kwargs(config: Any) -> Tuple[Dict[str, Any], Callable]:
    """
    Inputs kwargs.

    If the configuration is None, the function selects typical dimensions.
    """
    if config is not None:
        check_hasattr(config, "vocab_size")
    kwargs = dict(
        batch_size=2,
        sequence_length=30,
        dummy_max_token_id=31999 if config is None else (config.vocab_size - 1),
    )
    return kwargs, get_inputs
