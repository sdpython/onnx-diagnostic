from typing import Any, Callable, Dict, Optional, Tuple
import torch

# from ..helpers.cache_helper import make_dynamic_cache
from ..helpers.config_helper import update_config  # , check_hasattr, _pick

__TASK__ = "MoE"


def reduce_model_config(config: Any) -> Dict[str, Any]:
    """Reduces a model size."""
    kwargs: Dict[str, Any] = {}
    if hasattr(config, "num_hidden_layers"):
        config.num_hidden_layers = min(config.num_hidden_layers, 2)
    if hasattr(config, "vision_config") and hasattr(config.vision_config, "num_hidden_layers"):
        config.vision_config.num_hidden_layers = min(config.vision_config.num_hidden_layers, 2)
    if hasattr(config, "audio_processor") and hasattr(
        config.audio_processor, "num_hidden_layers"
    ):
        config.audio_processor.num_hidden_layers = min(
            config.audio_processor.num_hidden_layers, 2
        )
    if hasattr(config, "audio_processor") and hasattr(config.audio_processor, "attention_dim"):
        config.audio_processor.attention_dim = min(config.audio_processor.attention_dim, 2)
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
    Generates input for task ``MoE``.

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
    assert not add_second_input, "add_second_input=True not yet implemented"
    raise NotImplementedError(f"get_inputs not yet implemented for task {__TASK__!r}.")


def random_input_kwargs(config: Any) -> Tuple[Dict[str, Any], Callable]:
    """
    Inputs kwargs.

    If the configuration is None, the function selects typical dimensions.
    """
    raise NotImplementedError(
        f"random_input_kwargs not yet implemented for task {__TASK__!r}."
    )
