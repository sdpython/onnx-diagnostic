from typing import Any, Callable, Dict, Optional, Tuple
import torch
from ..helpers.cache_helper import make_dynamic_cache, make_hybrid_cache
from ..helpers.config_helper import (
    update_config,
    check_hasattr,
    _pick,
    default_num_hidden_layers as nhl,
)

__TASK__ = "image-text-to-text"


def reduce_model_config(config: Any) -> Dict[str, Any]:
    """Reduces a model size."""
    kwargs: Dict[str, Any] = {}
    if hasattr(config, "num_hidden_layers"):
        config.num_hidden_layers = min(config.num_hidden_layers, nhl())
    if hasattr(config, "mm_tokens_per_image"):
        config.mm_tokens_per_image = min(config.mm_tokens_per_image, 2)
    if hasattr(config, "vision_config"):
        if hasattr(config.vision_config, "num_hidden_layers"):
            config.vision_config.num_hidden_layers = min(
                config.vision_config.num_hidden_layers, 2
            )
        if hasattr(config.vision_config, "num_heads"):
            config.vision_config.num_heads = min(config.vision_config.num_heads, 4)
        if hasattr(config.vision_config, "image_size"):
            config.vision_config.image_size = min(config.vision_config.image_size, 168 // 2)
        if hasattr(config.vision_config, "intermediate_size"):
            config.vision_config.intermediate_size = min(
                config.vision_config.intermediate_size, 1076
            )
        if hasattr(config.vision_config, "patch_size"):
            config.vision_config.patch_size = min(config.vision_config.patch_size, 1)
        if hasattr(config.vision_config, "temporal_patch_size"):
            config.vision_config.temporal_patch_size = min(
                config.vision_config.temporal_patch_size, 8
            )
        if hasattr(config.vision_config, "hidden_size"):
            config.vision_config.hidden_size = min(config.vision_config.hidden_size, 16)
    if hasattr(config, "text_config"):
        if hasattr(config.text_config, "intermediate_size"):
            config.text_config.intermediate_size = min(
                config.text_config.intermediate_size, 320
            )
        if hasattr(config.text_config, "hidden_size"):
            config.text_config.hidden_size = min(config.text_config.hidden_size, 16)
        if hasattr(config.text_config, "num_hidden_layers"):
            config.text_config.num_hidden_layers = min(config.text_config.num_hidden_layers, 2)
        if hasattr(config.text_config, "layer_types"):
            config.text_config.layer_types = config.text_config.layer_types[
                : config.text_config.num_hidden_layers
            ]
        if hasattr(config.text_config, "num_attention_heads"):
            config.text_config.num_attention_heads = min(
                config.text_config.num_attention_heads, 2
            )
    update_config(config, kwargs)
    return kwargs


def _get_inputs_gemma3(
    model: torch.nn.Module,
    config: Optional[Any],
    dummy_max_token_id: int,
    num_key_value_heads: int,
    num_hidden_layers: int,
    pad_token_id: int,
    image_token_index: int,
    head_dim: int,
    width: int,
    height: int,
    num_channels: int,
    batch_size: int = 2,
    sequence_length: int = 43,
    sequence_length2: int = 43,
    n_images: int = 2,
    dynamic_rope: bool = False,
    max_sequence_length: int = 380,
    **kwargs,  # unused
):
    """
    ::

        dict(input_ids:T7s1x281,
            pixel_values:T16s1x3x896x896,
            attention_mask:dict(full_attention:T9s1x1x281x380,sliding_attention:T9s1x1x281x380),
            position_ids:T7s1x281,
            past_key_values:HybridCache(
                key_cache=#34[T1s1x4x380x256,...],
                value_cache=#34[T1s1x4x380x256,...]),
            token_type_ids:T7s1x281,
            cache_position:T7s281,
            logits_to_keep:1)
        dict(input_ids:T7s1x1,
            pixel_values:None,
            attention_mask:dict(full_attention:T9s1x1x1x380,sliding_attention:T9s1x1x1x380),
            position_ids:T7s1x1,
            past_key_values:HybridCache(
                key_cache=#34[T1s1x4x380x256,...],
                value_cache=#34[T1s1x4x380x256,...]),
            token_type_ids:T7s1x1,
            cache_position:T7s1,
            logits_to_keep:1)
    """
    assert (
        "cls_cache" not in kwargs
    ), f"Not yet implemented for cls_cache={kwargs['cls_cache']!r}."
    batch = torch.export.Dim("batch", min=1, max=1024)
    seq_length = "seq_length"  # torch.export.Dim("seq_length", min=1, max=4096)
    # cache_length = "cache_length"  # torch.export.Dim("cache_length", min=1, max=4096)

    shapes = {
        "input_ids": {0: batch, 1: seq_length},
        "token_type_ids": {0: batch, 1: seq_length},
        "attention_mask": {
            "full_attention": {0: batch, 2: seq_length},
            "sliding_attention": {0: batch, 2: seq_length},
        },
        "position_ids": {0: batch, 1: seq_length},
        "cache_position": {1: seq_length},
        "past_key_values": [
            [{0: batch} for _ in range(num_hidden_layers)],
            [{0: batch} for _ in range(num_hidden_layers)],
        ],
        "pixel_values": {0: batch},
        "use_cache": None,
    }

    input_ids = torch.randint(0, dummy_max_token_id, (batch_size, sequence_length2)).to(
        torch.int64
    )
    input_ids[:, 1] = image_token_index
    # input_ids[input_ids == image_token_index] = pad_token_id
    token_type_ids = torch.zeros_like(input_ids)
    token_type_ids[input_ids == image_token_index] = 1

    inputs = dict(
        input_ids=input_ids,
        token_type_ids=token_type_ids,
        attention_mask=dict(
            full_attention=torch.randn(batch_size, 1, sequence_length, max_sequence_length),
            sliding_attention=torch.randn(batch_size, 1, sequence_length, max_sequence_length),
        ),
        cache_position=torch.arange(0, sequence_length).to(torch.int64),
        position_ids=torch.arange(0, sequence_length).to(torch.int64).expand((batch_size, -1)),
        past_key_values=make_hybrid_cache(
            [
                (
                    torch.randn(
                        batch_size, num_key_value_heads, max_sequence_length, head_dim
                    ),
                    torch.randn(
                        batch_size, num_key_value_heads, max_sequence_length, head_dim
                    ),
                )
                for i in range(num_hidden_layers)
            ]
        ),
        pixel_values=torch.randn(n_images, num_channels, width, height).clamp(-1, 1),
        image_attention_mask=torch.ones((batch_size, sequence_length2, n_images)).to(
            torch.int64
        ),
        use_cache=True,  # Gemma3 does not set this value to true when a cache is provided
    )
    return dict(inputs=inputs, dynamic_shapes=shapes)


def get_inputs(
    model: torch.nn.Module,
    config: Optional[Any],
    dummy_max_token_id: int,
    num_key_value_heads: int,
    num_hidden_layers: int,
    pad_token_id: int,
    image_token_index: int,
    head_dim: int,
    width: int,
    height: int,
    num_channels: int,
    batch_size: int = 2,
    sequence_length: int = 43,
    sequence_length2: int = 43,
    n_images: int = 2,
    dynamic_rope: bool = False,
    add_second_input: int = 1,
    **kwargs,  # unused
):
    """
    Generates input for task ``image-text-to-text``.

    :param model: model to get the missing information
    :param config: configuration used to generate the model
    :param head_dim: last dimension of the cache
    :param dummy_max_token_id: dummy max token id
    :param pad_token_id: pad_token_id
    :param image_token_index: image_token_index
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
    if model.__class__.__name__.startswith("Gemma3"):
        res = _get_inputs_gemma3(
            model,
            config,
            dummy_max_token_id=dummy_max_token_id,
            num_key_value_heads=num_key_value_heads,
            num_hidden_layers=num_hidden_layers,
            pad_token_id=pad_token_id,
            image_token_index=image_token_index,
            head_dim=head_dim,
            width=width,
            height=height,
            num_channels=num_channels,
            batch_size=batch_size,
            sequence_length=sequence_length,
            sequence_length2=sequence_length2,
            n_images=n_images,
            dynamic_rope=dynamic_rope,
            **kwargs,
        )
    else:
        assert (
            "cls_cache" not in kwargs
        ), f"Not yet implemented for cls_cache={kwargs['cls_cache']!r}."
        batch = torch.export.Dim("batch", min=1, max=1024)
        batch_img = torch.export.Dim("batch_img", min=1, max=1024)
        seq_length = "seq_length"  # torch.export.Dim("seq_length", min=1, max=4096)
        cache_length = "cache_length"  # torch.export.Dim("cache_length", min=1, max=4096)
        images = "images"  # torch.export.Dim("images", min=1, max=4096)

        shapes = {
            "input_ids": {0: batch, 1: seq_length},
            "token_type_ids": {0: batch, 1: seq_length},
            "attention_mask": {0: batch, 1: "cache+seq"},
            "position_ids": {0: batch, 1: "cache+seq"},
            "past_key_values": [
                [{0: batch} for _ in range(num_hidden_layers)],
                [{0: batch, 2: cache_length} for _ in range(num_hidden_layers)],
            ],
            "pixel_values": (
                {0: batch, 1: images}
                if model.__class__.__name__ == "IdeficsForVisionText2Text"
                else {0: batch_img}
            ),
            "image_attention_mask": {0: batch, 1: seq_length, 2: images},
            "image_grid_thw": {0: batch},
            "use_cache": None,
        }

        input_ids = torch.randint(0, dummy_max_token_id, (batch_size, sequence_length2)).to(
            torch.int64
        )
        input_ids[0, 0] = image_token_index
        input_ids[1, 1] = image_token_index
        # input_ids[input_ids == image_token_index] = pad_token_id
        token_type_ids = torch.zeros_like(input_ids)
        token_type_ids[input_ids == image_token_index] = 1
        image_grid_thw = torch.zeros((n_images, 3), dtype=torch.int64)
        image_grid_thw[:, 1] = height
        image_grid_thw[:, 2] = width
        image_grid_thw[0, :] //= 2
        image_grid_thw[:, 0] = torch.arange(n_images, dtype=image_grid_thw.dtype)

        inputs = dict(
            input_ids=input_ids,
            attention_mask=torch.cat(
                [
                    torch.ones((batch_size, sequence_length), dtype=torch.int64),
                    input_ids.ne(pad_token_id).to(torch.int64),
                ],
                axis=-1,
            ),
            position_ids=torch.arange(0, sequence_length2)
            .to(torch.int64)
            .expand((batch_size, -1)),
            past_key_values=make_dynamic_cache(
                [
                    (
                        torch.randn(
                            batch_size, num_key_value_heads, sequence_length, head_dim
                        ),
                        torch.randn(
                            batch_size, num_key_value_heads, sequence_length, head_dim
                        ),
                    )
                    for i in range(num_hidden_layers)
                ]
            ),
            pixel_values=(
                torch.randn((batch_size, n_images, num_channels, width, height)).clamp(-1, 1)
                if model.__class__.__name__ == "IdeficsForVisionText2Text"
                else torch.randn(n_images, num_channels, width, height).clamp(-1, 1)
            ),
            image_attention_mask=torch.ones((batch_size, sequence_length2, n_images)).to(
                torch.int64
            ),
            token_type_ids=token_type_ids,
            image_grid_thw=image_grid_thw,
            use_cache=True,  # Gemma3 does not set this value to true when a cache is provided
        )
        res = dict(inputs=inputs, dynamic_shapes=shapes)
    if add_second_input:
        assert (
            add_second_input > 0
        ), f"Not implemented for add_second_input={add_second_input}."
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
            sequence_length=sequence_length + add_second_input,
            sequence_length2=sequence_length2 + 1,
            n_images=n_images + 1,
            dynamic_rope=dynamic_rope,
            pad_token_id=pad_token_id,
            image_token_index=image_token_index,
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
        if hasattr(config, "text_config"):
            check_hasattr(
                config.text_config,
                "vocab_size",
                "hidden_size",
                "num_attention_heads",
                ("num_key_value_heads", "num_attention_heads"),
                "intermediate_size",
                "hidden_size",
                "pad_token_id",
            )
            check_hasattr(config, "vision_config", ("image_token_index", "image_token_id"))
            text_config = True
        else:
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
            text_config = False
        check_hasattr(config.vision_config, ("num_channels", "in_chans", "in_channels"))
    kwargs = dict(
        batch_size=2,
        sequence_length=43,
        sequence_length2=43,
        head_dim=(
            16
            if config is None
            else getattr(
                config,
                "head_dim",
                (
                    config.text_config.head_dim
                    if text_config and hasattr(config.text_config, "head_dim")
                    else (
                        (config.text_config.hidden_size if text_config else config.hidden_size)
                        // (
                            config.text_config.num_attention_heads
                            if text_config
                            else config.num_attention_heads
                        )
                    )
                ),
            )
        ),
        dummy_max_token_id=(
            31999
            if config is None
            else (config.text_config.vocab_size if text_config else config.vocab_size) - 1
        ),
        num_hidden_layers=(
            4
            if config is None
            else (
                config.text_config.num_hidden_layers
                if text_config
                else config.num_hidden_layers
            )
        ),
        num_key_value_heads=(
            8
            if config is None
            else (
                _pick(config.text_config, "num_key_value_heads", "num_attention_heads")
                if text_config
                else _pick(config, "num_key_value_heads", "num_attention_heads")
            )
        ),
        intermediate_size=(
            1024
            if config is None
            else (
                config.text_config.intermediate_size
                if text_config
                else config.intermediate_size
            )
        ),
        hidden_size=(
            512
            if config is None
            else (config.text_config.hidden_size if text_config else config.hidden_size)
        ),
        width=(
            224
            if config is None or not hasattr(config.vision_config, "image_size")
            else config.vision_config.image_size
        ),
        height=(
            224
            if config is None or not hasattr(config.vision_config, "image_size")
            else config.vision_config.image_size
        ),
        num_channels=(
            3
            if config is None
            else _pick(config.vision_config, "num_channels", "in_chans", "in_channels")
        ),
        pad_token_id=(
            0
            if config is None
            or not hasattr(config, "text_config")
            or not hasattr(config.text_config, "pad_token_id")
            else config.text_config.pad_token_id
        ),
        image_token_index=(
            4
            if config is None
            or (
                not hasattr(config, "image_token_index")
                and not hasattr(config, "image_token_id")
            )
            else _pick(config, "image_token_index", "image_token_id")
        ),
    )
    return kwargs, get_inputs
