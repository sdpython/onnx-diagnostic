from typing import Any, Callable, Dict, Optional, Tuple
import torch
from ..helpers.cache_helper import make_dynamic_cache, make_encoder_decoder_cache
from ..helpers.config_helper import (
    update_config,
    check_hasattr,
    _pick,
    default_num_hidden_layers as nhl,
)

__TASK__ = "text2text-generation"


def reduce_model_config(config: Any) -> Dict[str, Any]:
    """Reduces a model size."""
    kwargs: Dict[str, Any] = {}
    if hasattr(config, "num_decoder_layers"):
        config.num_decoder_layers = min(config.num_decoder_layers, 2)
    if hasattr(config, "num_hidden_layers"):
        config.num_hidden_layers = min(config.num_hidden_layers, nhl())
    update_config(config, kwargs)
    return kwargs


def get_inputs(
    model: torch.nn.Module,
    config: Optional[Any],
    dummy_max_token_id: int,
    num_key_value_heads_encoder: int,
    num_key_value_heads_decoder: int,
    num_hidden_layers: int,
    head_dim_encoder: int,
    head_dim_decoder: int,
    encoder_dim: int,
    batch_size: int = 2,
    sequence_length: int = 30,
    sequence_length2: int = 3,
    add_second_input: int = 1,
    **kwargs,  # unused
):
    """
    Generates input for task ``text2text-generation``.

    :param model: model to get the missing information
    :param config: configuration used to generate the model
    :param head_dim_encoder: last dimension of the cache for the encoder
    :param head_dim_decoder: last dimension of the cache for the decoder
    :param num_key_value_heads_encoder: number of heads for the encoder
    :param num_key_value_heads_decoder: number of heads for the decoder
    :param dummy_max_token_id: dummy max token id
    :param batch_size: batch size
    :param encoder_dim: last dimension of encoder_last_hidden_state
    :param sequence_length: sequence length
    :param sequence_length2: new sequence length
    :return: dictionary

    Stolen inputs for one model.

    ::

        cache_position:T7s1
        past_key_values:EncoderDecoderCache(
            self_attention_cache=DynamicCache(
                key_cache=#6[T1s1x8x1x64,...],
                value_cache=#6[T1s1x8x1x64,...]),
            cross_attention_cache=DynamicCache(
                key_cache=#6[T1s1x8x16x64,...],
                value_cache=#6[T1s1x8x16x64,...])),
        decoder_input_ids:T7s1x1,
        encoder_outputs:dict(last_hidden_state:T1s1x16x512)
    """
    assert (
        "cls_cache" not in kwargs
    ), f"Not yet implemented for cls_cache={kwargs['cls_cache']!r}."
    batch = "batch"
    seq_length = "seq_length"  # torch.export.Dim("seq_length", min=1, max=4096)
    cache_length = "cache_length_key"
    cache_length2 = "cache_length_val"

    shapes = {
        "input_ids": {0: batch, 1: seq_length},
        "decoder_input_ids": {0: batch, 1: "seq_ids"},
        "attention_mask": {0: batch, 1: "seq_mask"},
        # "cache_position": {0: batch, 1: torch.export.Dim.DYNAMIC},
        "past_key_values": [
            [{0: batch, 2: cache_length} for _ in range(num_hidden_layers * 2)],
            [{0: batch, 2: cache_length2} for _ in range(num_hidden_layers * 2)],
        ],
        # one these is selected based on the forward method signature
        # "encoder_last_hidden_state": {0: batch, 1: torch.export.Dim.DYNAMIC},
        # "encoder_outputs": {0: batch, 1: torch.export.Dim.DYNAMIC},
    }

    inputs = dict(
        input_ids=torch.randint(0, dummy_max_token_id, (batch_size, sequence_length)).to(
            torch.int64
        ),
        decoder_input_ids=torch.randint(
            0, dummy_max_token_id, (batch_size, sequence_length2)
        ).to(torch.int64),
        attention_mask=torch.ones((batch_size, sequence_length)).to(torch.int64),
        # cache_position=torch.arange(sequence_length, sequence_length + sequence_length2)
        # .to(torch.int64)
        # .expand((batch_size, -1)),
        past_key_values=make_encoder_decoder_cache(
            make_dynamic_cache(
                [
                    (
                        torch.randn(
                            batch_size,
                            num_key_value_heads_encoder,
                            sequence_length,
                            head_dim_encoder,
                        ),
                        torch.randn(
                            batch_size,
                            num_key_value_heads_encoder,
                            sequence_length,
                            head_dim_encoder,
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
                            num_key_value_heads_decoder,
                            sequence_length2,
                            head_dim_decoder,
                        ),
                        torch.randn(
                            batch_size,
                            num_key_value_heads_decoder,
                            sequence_length2,
                            head_dim_decoder,
                        ),
                    )
                    for i in range(num_hidden_layers)
                ]
            ),
        ),
        # one these is selected based on the forward method signature
        # encoder_last_hidden_state=torch.randn(batch_size, sequence_length2, encoder_dim),
        # encoder_outputs=torch.randn(batch_size, sequence_length2, encoder_dim),
    )
    res = dict(inputs=inputs, dynamic_shapes=shapes)
    if add_second_input:
        assert (
            add_second_input > 0
        ), f"Not implemented for add_second_input={add_second_input}."
        res["inputs_prompt"] = dict(input_ids=torch.randint(1000, 30000, (1, 11)))
        res["inputs2"] = get_inputs(
            model=model,
            config=config,
            dummy_max_token_id=dummy_max_token_id,
            num_key_value_heads_encoder=num_key_value_heads_encoder,
            num_key_value_heads_decoder=num_key_value_heads_decoder,
            num_hidden_layers=num_hidden_layers,
            head_dim_encoder=head_dim_encoder,
            head_dim_decoder=head_dim_decoder,
            encoder_dim=encoder_dim,
            batch_size=batch_size + 1,
            sequence_length=sequence_length + add_second_input,
            sequence_length2=sequence_length2 + 1,
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
        check_hasattr(
            config,
            "vocab_size",
            "hidden_size",
            "num_attention_heads",
            ("num_hidden_layers", "num_layers"),
            ("n_positions", "d_model"),
            (
                "num_key_value_heads",
                "num_heads",
                ("decoder_attention_heads", "encoder_attention_heads"),
            ),
        )
    # exceptions = {
    #     "PLBartForConditionalGeneration": (
    #         lambda c: c.encoder_attention_heads + c.decoder_attention_heads
    #    )
    # }
    kwargs = dict(
        batch_size=2,
        sequence_length=30,
        sequence_length2=3,
        head_dim_encoder=16 if config is None else _pick(config, "d_kv", "encoder_ffn_dim"),
        head_dim_decoder=16 if config is None else _pick(config, "d_kv", "decoder_ffn_dim"),
        dummy_max_token_id=31999 if config is None else config.vocab_size - 1,
        num_hidden_layers=(
            8 if config is None else _pick(config, "num_hidden_layers", "num_layers")
        ),
        num_key_value_heads_encoder=(
            16
            if config is None
            else _pick(
                config,
                "encoder_attention_heads",
                "num_key_value_heads",
                "num_heads",
            )
        ),
        num_key_value_heads_decoder=(
            16
            if config is None
            else _pick(
                config,
                "decoder_attention_heads",
                "num_key_value_heads",
                "num_heads",
            )
        ),
        encoder_dim=512 if config is None else _pick(config, "n_positions", "d_model"),
    )
    return kwargs, get_inputs
