from typing import Any, Callable, Dict, Optional, Tuple
import torch
from ..helpers.cache_helper import make_dynamic_cache
from ..helpers.config_helper import update_config, check_hasattr, _pick

__TASK__ = "image-text-to-text"


def reduce_model_config(config: Any) -> Dict[str, Any]:
    """Reduces a model size."""
    kwargs: Dict[str, Any] = {}
    if hasattr(config, "num_hidden_layers"):
        config.num_hidden_layers = min(config.num_hidden_layers, 2)
    if hasattr(config, "vision_config") and hasattr(config.vision_config, "num_hidden_layers"):
        config.vision_config.num_hidden_layers = min(config.vision_config.num_hidden_layers, 2)
    update_config(config, kwargs)
    return kwargs


def get_inputs(
    model: torch.nn.Module,
    config: Optional[Any],
    dummy_max_token_id: int,
    num_key_value_heads: int,
    num_hidden_layers: int,
    head_dim: int,
    width: int,
    height: int,
    num_channels: int,
    batch_size: int = 2,
    sequence_length: int = 30,
    sequence_length2: int = 3,
    n_images: int = 2,
    dynamic_rope: bool = False,
    add_second_input: bool = False,
    **kwargs,  # unused
):
    """
    Generates input for task ``image-text-to-text``.

    :param model: model to get the missing information
    :param config: configuration used to generate the model
    :param head_dim: last dimension of the cache
    :param dummy_max_token_id: dummy max token id
    :param batch_size: batch size
    :param sequence_length: sequence length
    :param sequence_length2: new sequence length
    :param n_images: number of images
    :param width: width of the image
    :param height: height of the image
    :param num_channels: number of channels
    :param dynamic_rope: use dynamic rope (see :class:`transformers.LlamaConfig`)
    :return: dictionary
    """
    batch = torch.export.Dim("batch", min=1, max=1024)
    seq_length = "seq_length"  # torch.export.Dim("seq_length", min=1, max=4096)
    cache_length = "cache_length"  # torch.export.Dim("cache_length", min=1, max=4096)
    images = "images"  # torch.export.Dim("images", min=1, max=4096)

    shapes = {
        "input_ids": {0: batch, 1: seq_length},
        "attention_mask": {
            0: batch,
            1: "cache+seq",  # cache_length + seq_length
        },
        "position_ids": {
            0: batch,
            1: "cache+seq",  # cache_length + seq_length
        },
        "past_key_values": [
            [{0: batch, 2: cache_length} for _ in range(num_hidden_layers)],
            [{0: batch, 2: cache_length} for _ in range(num_hidden_layers)],
        ],
        "pixel_values": {0: batch, 1: images},
        "image_attention_mask": {0: batch, 1: seq_length, 2: images},
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
        image_attention_mask=torch.ones((batch_size, sequence_length2, n_images)).to(
            torch.int64
        ),
        pixel_values=torch.ones((batch_size, n_images, num_channels, width, height)).to(
            torch.int64
        ),
    )
    res = dict(inputs=inputs, dynamic_shapes=shapes)
    if add_second_input:
        res["inputs2"] = get_inputs(
            model=model,
            config=config,
            dummy_max_token_id=dummy_max_token_id,
            num_key_value_heads=num_key_value_heads,
            num_hidden_layers=num_hidden_layers,
            head_dim=head_dim,
            width=width,
            height=height,
            num_channels=num_channels,
            batch_size=batch_size + 1,
            sequence_length=sequence_length + 1,
            sequence_length2=sequence_length2 + 1,
            n_images=n_images + 1,
            dynamic_rope=dynamic_rope,
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
            ("num_key_value_heads", "num_attention_heads"),
            "intermediate_size",
            "hidden_size",
            "vision_config",
        )
        check_hasattr(config.vision_config, "image_size", "num_channels")
    kwargs = dict(
        batch_size=2,
        sequence_length=30,
        sequence_length2=3,
        head_dim=(
            16
            if config is None
            else getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        ),
        dummy_max_token_id=31999 if config is None else config.vocab_size - 1,
        num_hidden_layers=4 if config is None else config.num_hidden_layers,
        num_key_value_heads=(
            8
            if config is None
            else _pick(config, "num_key_value_heads", "num_attention_heads")
        ),
        intermediate_size=1024 if config is None else config.intermediate_size,
        hidden_size=512 if config is None else config.hidden_size,
        width=224 if config is None else config.vision_config.image_size,
        height=224 if config is None else config.vision_config.image_size,
        num_channels=3 if config is None else config.vision_config.num_channels,
    )
    return kwargs, get_inputs
