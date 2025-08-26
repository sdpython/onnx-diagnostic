from typing import Any, Callable, Dict, Optional, Tuple
import torch
from ..helpers.config_helper import update_config, check_hasattr
from ..helpers.cache_helper import make_dynamic_cache, make_encoder_decoder_cache

__TASK__ = "feature-extraction"


def reduce_model_config(config: Any) -> Dict[str, Any]:
    """Reduces a model size."""
    check_hasattr(config, "num_hidden_layers")
    kwargs = dict(num_hidden_layers=min(config.num_hidden_layers, 4))
    update_config(config, kwargs)
    return kwargs


def get_inputs(
    model: torch.nn.Module,
    config: Optional[Any],
    batch_size: int,
    sequence_length: int,
    dummy_max_token_id: int,
    sequence_length2: int = 3,
    decoder_attention_heads: Optional[int] = None,
    encoder_attention_heads: Optional[int] = None,
    encoder_ffn_dim: Optional[int] = None,
    decoder_ffn_dim: Optional[int] = None,
    num_hidden_layers: Optional[int] = None,
    add_second_input: int = 1,
    **kwargs,  # unused
):
    """
    Generates inputs for task ``feature-extraction``.
    Example:

    ::

        input_ids:T7s1x13[101,72654:A16789.23076923077],
        token_type_ids:T7s1x13[0,0:A0.0],
        attention_mask:T7s1x13[1,1:A1.0])
    """
    assert (
        "cls_cache" not in kwargs
    ), f"Not yet implemented for cls_cache={kwargs['cls_cache']!r}."
    batch = torch.export.Dim("batch", min=1, max=1024)
    seq_length = "sequence_length"
    shapes = {
        "input_ids": {0: batch, 1: seq_length},
        "attention_mask": {0: batch, 1: seq_length},
    }
    inputs = dict(
        input_ids=torch.randint(0, dummy_max_token_id, (batch_size, sequence_length)).to(
            torch.int64
        ),
        attention_mask=torch.ones((batch_size, sequence_length)).to(torch.int64),
    )
    if (
        encoder_attention_heads
        and decoder_attention_heads
        and encoder_ffn_dim
        and decoder_ffn_dim
        and num_hidden_layers
    ):
        inputs["past_key_values"] = make_encoder_decoder_cache(
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
        cache_length = "cache_length_key"
        cache_length2 = "cache_length_val"
        shapes["past_key_values"] = [  # type: ignore[assignment]
            [
                [{0: batch, 2: cache_length} for _ in range(num_hidden_layers)],
                [{0: batch, 2: cache_length} for _ in range(num_hidden_layers)],
            ],
            [
                [{0: batch, 2: cache_length2} for _ in range(num_hidden_layers)],
                [{0: batch, 2: cache_length2} for _ in range(num_hidden_layers)],
            ],
        ]

    res = dict(inputs=inputs, dynamic_shapes=shapes)
    if add_second_input:
        assert (
            add_second_input > 0
        ), f"Not implemented for add_second_input={add_second_input}."
        res["inputs2"] = get_inputs(
            model=model,
            config=config,
            batch_size=batch_size + 1,
            sequence_length=sequence_length + add_second_input,
            dummy_max_token_id=dummy_max_token_id,
            sequence_length2=sequence_length2,
            decoder_attention_heads=decoder_attention_heads,
            encoder_attention_heads=encoder_attention_heads,
            encoder_ffn_dim=encoder_ffn_dim,
            decoder_ffn_dim=decoder_ffn_dim,
            num_hidden_layers=num_hidden_layers,
            add_second_input=0,
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
    for att in [
        "decoder_attention_heads",
        "encoder_attention_heads",
        "encoder_ffn_dim",
        "decoder_ffn_dim",
        "num_hidden_layers",
    ]:
        if hasattr(config, att):
            kwargs[att] = getattr(config, att)
    kwargs["decoder_ffn_dim"] = kwargs["encoder_ffn_dim"] = 64
    return kwargs, get_inputs
