from typing import Any, Callable, Dict, Optional, Tuple
import torch
from ..helpers.config_helper import update_config, check_hasattr

__TASK__ = "image-classification"


def reduce_model_config(config: Any, task: str) -> Dict[str, Any]:
    """Reduces a model size."""
    check_hasattr(config, ("num_hidden_layers", "hidden_sizes"))
    kwargs = dict(
        num_hidden_layers=(
            min(config.num_hidden_layers, 2)
            if hasattr(config, "num_hidden_layers")
            else len(config.hidden_sizes)
        )
    )
    update_config(config, kwargs)
    return kwargs


def get_inputs(
    model: torch.nn.Module,
    config: Optional[Any],
    input_width: int,
    input_height: int,
    input_channels: int,
    batch_size: int = 2,
    dynamic_rope: bool = False,
    **kwargs,  # unused
):
    """
    Generates inputs for task ``image-classification``.

    :param model: model to get the missing information
    :param config: configuration used to generate the model
    :param batch_size: batch size
    :param input_channels: input channel
    :param input_width: input width
    :param input_height: input height
    :return: dictionary
    """
    assert isinstance(
        input_width, int
    ), f"Unexpected type for input_width {type(input_width)}{config}"
    assert isinstance(
        input_width, int
    ), f"Unexpected type for input_height {type(input_height)}{config}"

    shapes = {
        "pixel_values": {
            0: torch.export.Dim("batch", min=1, max=1024),
            2: torch.export.Dim("width", min=1, max=4096),
            3: torch.export.Dim("height", min=1, max=4096),
        },
    }
    inputs = dict(
        pixel_values=torch.randn(batch_size, input_channels, input_width, input_height).clamp(
            -1, 1
        ),
    )
    return dict(inputs=inputs, dynamic_shapes=shapes)


def random_input_kwargs(config: Any, task: str) -> Tuple[Dict[str, Any], Callable]:
    """
    Inputs kwargs.

    If the configuration is None, the function selects typical dimensions.
    """
    if config is not None:
        check_hasattr(config, ("image_size", "architectures"), "num_channels")
    if config is not None:
        if hasattr(config, "image_size"):
            image_size = config.image_size
        else:
            assert config.architectures, f"empty architecture in {config}"
            from ..torch_models.hghub.hub_api import get_architecture_default_values

            default_values = get_architecture_default_values(config.architectures[0])
            image_size = default_values["image_size"]
    if config is None or isinstance(image_size, int):
        kwargs = dict(
            batch_size=2,
            input_width=224 if config is None else image_size,
            input_height=224 if config is None else image_size,
            input_channels=3 if config is None else config.num_channels,
        )
    else:
        kwargs = dict(
            batch_size=2,
            input_width=config.image_size[0],
            input_height=config.image_size[1],
            input_channels=config.num_channels,
        )
    return kwargs, get_inputs
