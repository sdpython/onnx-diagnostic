from typing import Any, Callable, Dict, Optional, Tuple
import torch
from ..helpers.config_helper import update_config, check_hasattr, pick

__TASK__ = "text-to-image"


def reduce_model_config(config: Any) -> Dict[str, Any]:
    """Reduces a model size."""
    check_hasattr(config, "sample_size", "cross_attention_dim")
    kwargs = dict(
        sample_size=min(config["sample_size"], 32),
        cross_attention_dim=min(config["cross_attention_dim"], 64),
    )
    update_config(config, kwargs)
    return kwargs


def get_inputs(
    model: torch.nn.Module,
    config: Optional[Any],
    batch_size: int,
    sequence_length: int,
    cache_length: int,
    in_channels: int,
    sample_size: int,
    cross_attention_dim: int,
    add_second_input: bool = False,
    **kwargs,  # unused
):
    """
    Generates inputs for task ``text-to-image``.
    Example:

    ::

        sample:T10s2x4x96x96[-3.7734375,4.359375:A-0.043463995395642184]
        timestep:T7s=101
        encoder_hidden_states:T10s2x77x1024[-6.58203125,13.0234375:A-0.16780663634440257]
    """
    assert (
        "cls_cache" not in kwargs
    ), f"Not yet implemented for cls_cache={kwargs['cls_cache']!r}."
    batch = "batch"
    shapes = {
        "sample": {0: batch},
        "timestep": {},
        "encoder_hidden_states": {0: batch, 1: "encoder_length"},
    }
    inputs = dict(
        sample=torch.randn((batch_size, sequence_length, sample_size, sample_size)).to(
            torch.float32
        ),
        timestep=torch.tensor([101], dtype=torch.int64),
        encoder_hidden_states=torch.randn(
            (batch_size, sequence_length, cross_attention_dim)
        ).to(torch.float32),
    )
    res = dict(inputs=inputs, dynamic_shapes=shapes)
    if add_second_input:
        res["inputs2"] = get_inputs(
            model=model,
            config=config,
            batch_size=batch_size + 1,
            sequence_length=sequence_length,
            cache_length=cache_length + 1,
            in_channels=in_channels,
            sample_size=sample_size,
            cross_attention_dim=cross_attention_dim,
            **kwargs,
        )["inputs"]
    return res


def random_input_kwargs(config: Any) -> Tuple[Dict[str, Any], Callable]:
    """
    Inputs kwargs.

    If the configuration is None, the function selects typical dimensions.
    """
    if config is not None:
        check_hasattr(config, "sample_size", "cross_attention_dim", "in_channels")
    kwargs = dict(
        batch_size=2,
        sequence_length=pick(config, "in_channels", 4),
        cache_length=77,
        in_channels=pick(config, "in_channels", 4),
        sample_size=pick(config, "sample_size", 32),
        cross_attention_dim=pick(config, "cross_attention_dim", 64),
    )
    return kwargs, get_inputs
