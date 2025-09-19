from typing import Any, Callable, Dict, Optional, Tuple
import torch
from ..helpers.config_helper import (
    update_config,
    check_hasattr,
    default_num_hidden_layers as nhl,
)

__TASK__ = "image-to-video"


def reduce_model_config(config: Any) -> Dict[str, Any]:
    """Reduces a model size."""
    if not hasattr(config, "num_hidden_layers") and not hasattr(config, "num_layers"):
        # We cannot reduce.
        return {}
    check_hasattr(config, ("num_hidden_layers", "num_layers"))
    kwargs = {}
    if hasattr(config, "num_layers"):
        kwargs["num_layers"] = min(config.num_layers, nhl())
    if hasattr(config, "num_hidden_layers"):
        kwargs["num_hidden_layers"] = min(config.num_hidden_layers, nhl())

    update_config(config, kwargs)
    return kwargs


def get_inputs(
    model: torch.nn.Module,
    config: Optional[Any],
    text_embed_dim: int,
    latent_channels: int,
    batch_size: int = 2,
    image_height: int = 704,
    image_width: int = 1280,
    latent_frames: int = 1,
    text_maxlen: int = 512,
    add_second_input: int = 1,
    **kwargs,  # unused
):
    """
    Generates inputs for task ``image-to-video``.
    """
    assert (
        "cls_cache" not in kwargs
    ), f"Not yet implemented for cls_cache={kwargs['cls_cache']!r}."
    latent_height = image_height // 8
    latent_width = image_width // 8
    dtype = torch.float32

    inputs = dict(
        hidden_states=torch.randn(
            batch_size,
            latent_channels,
            latent_frames,
            latent_height,
            latent_width,
            dtype=dtype,
        ),
        timestep=torch.tensor([1.0] * batch_size, dtype=dtype),
        encoder_hidden_states=torch.randn(
            batch_size, text_maxlen, text_embed_dim, dtype=dtype
        ),
        padding_mask=torch.ones(1, 1, image_height, image_width, dtype=dtype),
        fps=torch.tensor([16] * batch_size, dtype=dtype),
        condition_mask=torch.randn(
            batch_size, 1, latent_frames, latent_height, latent_width, dtype=dtype
        ),
    )
    shapes = dict(
        hidden_states={
            0: "batch_size",
            2: "latent_frames",
            3: "latent_height",
            4: "latent_width",
        },
        timestep={0: "batch_size"},
        encoder_hidden_states={0: "batch_size"},
        padding_mask={0: "batch_size", 2: "height", 3: "width"},
        fps={0: "batch_size"},
        condition_mask={
            0: "batch_size",
            2: "latent_frames",
            3: "latent_height",
            4: "latent_width",
        },
    )
    res = dict(inputs=inputs, dynamic_shapes=shapes)

    if add_second_input:
        assert (
            add_second_input > 0
        ), f"Not implemented for add_second_input={add_second_input}."
        res["inputs2"] = get_inputs(
            model=model,
            config=config,
            text_embed_dim=text_embed_dim,
            latent_channels=latent_channels,
            batch_size=batch_size,
            image_height=image_height,
            image_width=image_width,
            latent_frames=latent_frames,
            text_maxlen=text_maxlen,
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
        check_hasattr(config, "in_channels", "text_embed_dim"),
    kwargs = dict(
        text_embed_dim=1024 if config is None else config.text_embed_dim,
        latent_channels=16 if config is None else config.in_channels - 1,
        batch_size=1,
        image_height=8 * 50,
        image_width=8 * 80,
        latent_frames=1,
        text_maxlen=512,
    )
    return kwargs, get_inputs
