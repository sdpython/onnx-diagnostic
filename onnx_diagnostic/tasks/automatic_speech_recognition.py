from typing import Any, Callable, Dict, Optional, Tuple
import torch
import transformers
from ..helpers.cache_helper import make_dynamic_cache, make_encoder_decoder_cache
from ..helpers.config_helper import update_config, check_hasattr

__TASK__ = "automatic-speech-recognition"


def reduce_model_config(config: Any, task: str) -> Dict[str, Any]:
    """Reduces a model size."""
    kwargs: Dict[str, Any] = {}
    if hasattr(config, "num_decoder_layers"):
        config.num_decoder_layers = min(config.num_decoder_layers, 2)
    if hasattr(config, "decoder_layers"):
        config.decoder_layers = min(config.decoder_layers, 2)
    if hasattr(config, "num_hidden_layers"):
        config.num_hidden_layers = min(config.num_hidden_layers, 2)
    update_config(config, kwargs)
    return kwargs


def get_inputs(
    model: torch.nn.Module,
    config: Optional[Any],
    dummy_max_token_id: int,
    max_source_positions: int,
    d_model: int,
    num_hidden_layers: int,
    encoder_attention_heads: int,
    encoder_layers: int,
    decoder_layers: int,
    head_dim: int,
    batch_size: int = 2,
    sequence_length: int = 30,
    **kwargs,  # unused
):
    """
    Generates inputs for task ``text2text-generation``.
    Example:

    ::

        dict(
            cache_position:T7s4,
            past_key_values:EncoderDecoderCache(
                self_attention_cache=DynamicCache[serialized](#2[#0[],#0[]]),
                cross_attention_cache=DynamicCache[serialized](#2[#0[],#0[]])
            ),
            decoder_input_ids:T7s1x4,
            encoder_outputs:BaseModelOutput(last_hidden_state:T1s1x1500x384),
            use_cache:bool,return_dict:bool
        )
        dict(
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
            encoder_outputs:BaseModelOutput(last_hidden_state:T1s1x1500x384),
            use_cache:bool,return_dict:bool
        )
    """
    batch = torch.export.Dim("batch", min=1, max=1024)
    seq_length = torch.export.Dim("seq_length", min=1, max=4096)

    shapes = {
        "decoder_input_ids": {0: batch, 1: seq_length},
        "cache_position": {0: seq_length},
        "encoder_outputs": {"last_hidden_state": {0: batch}},
        "past_key_values": [
            [
                [{0: batch} for _ in range(num_hidden_layers)],
                [{0: batch} for _ in range(num_hidden_layers)],
            ],
            [
                [{0: batch} for _ in range(num_hidden_layers)],
                [{0: batch} for _ in range(num_hidden_layers)],
            ],
        ],
    }
    inputs = dict(
        decoder_input_ids=torch.randint(
            0, dummy_max_token_id, (batch_size, sequence_length)
        ).to(torch.int64),
        cache_position=(torch.arange(sequence_length) + 5).to(torch.int64),
        encoder_outputs=transformers.modeling_outputs.BaseModelOutput(
            last_hidden_state=torch.randn(batch_size, max_source_positions, d_model)
        ),
        past_key_values=make_encoder_decoder_cache(
            make_dynamic_cache(
                [
                    (
                        torch.randn(
                            batch_size, encoder_attention_heads, encoder_layers, head_dim
                        ),
                        torch.randn(
                            batch_size, encoder_attention_heads, encoder_layers, head_dim
                        ),
                    )
                    for i in range(num_hidden_layers)
                ]
            ),
            make_dynamic_cache(
                [
                    (
                        torch.randn(
                            batch_size, encoder_attention_heads, max_source_positions, head_dim
                        ),
                        torch.randn(
                            batch_size, encoder_attention_heads, max_source_positions, head_dim
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
    return dict(inputs=inputs, dynamic_shapes=shapes)


def random_input_kwargs(config: Any, task: str) -> Tuple[Dict[str, Any], Callable]:
    """
    Inputs kwargs.

    If the configuration is None, the function selects typical dimensions.
    """
    if config is not None:
        check_hasattr(
            config,
            "d_model",
            "decoder_attention_heads",
            "decoder_layers",
            "encoder_attention_heads",
            "encoder_layers",
            "max_source_positions",
            "num_hidden_layers",
            "vocab_size",
        )
    kwargs = dict(
        batch_size=2,
        sequence_length=30,
        dummy_max_token_id=31000 if config is None else config.vocab_size,
        max_source_positions=1500 if config is None else config.max_source_positions,
        d_model=384 if config is None else config.d_model,
        num_hidden_layers=4 if config is None else config.num_hidden_layers,
        encoder_attention_heads=6 if config is None else config.encoder_attention_heads,
        encoder_layers=4 if config is None else config.encoder_layers,
        decoder_attention_heads=6 if config is None else config.decoder_attention_heads,
        decoder_layers=4 if config is None else config.decoder_layers,
        head_dim=(
            64 if config is None else (config.d_model // config.encoder_attention_heads)
        ),
    )
    return kwargs, get_inputs
