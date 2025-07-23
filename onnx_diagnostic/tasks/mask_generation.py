from typing import Any, Callable, Dict, Optional, Tuple
import torch
from ..helpers.config_helper import update_config, check_hasattr

__TASK__ = "mask-generation"


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
    batch_size: int,
    width: int,
    height: int,
    num_channels: int,
    output_channels: int,
    window_size: int,
    add_second_input: bool = True,
    **kwargs,  # unused
):
    """
    Generates input for task ``mask-generation``.

    :param model: model to get the missing information
    :param config: configuration used to generate the model
    :param batch_size: batch size
    :param width: width of the image
    :param height: height of the image
    :param num_channels: number of channels in the image
    :param output_channels: number of output channels
    :param window_size: size of the window for the vision model
    :return: dictionary with inputs and dynamic shapes

    """
    assert (
        "cls_cache" not in kwargs
    ), f"Not yet implemented for cls_cache={kwargs['cls_cache']!r}."

    # TODO(anyone): input_masks is weridly failing all the time with mismatch channels
    # with Conv or embedding_size. I guess maybe the model is too implicit on the
    # input_masks shape.

    shapes = {
        "pixel_values": {0: "batch", 2: "height", 3: "width"},  # 1: num_channels is static
        "input_points": {0: "batch", 1: "point_batch_size", 2: "nb_points_per_image"},
        "input_boxes": {0: "batch", 1: "point_batch_size"},
        # "input_masks": {0: "batch", 2: "height", 3: "width"},
    }
    inputs = dict(
        pixel_values=torch.randn(
            (batch_size, num_channels, height, width), dtype=torch.float32
        ),
        input_points=torch.randn(
            (batch_size, 1, 10, 2), dtype=torch.float32
        ),  # 10 points per image
        input_boxes=torch.randn((batch_size, 1, 4), dtype=torch.float32),  # 1 box per image
        # input_masks=torch.randn(
        #     (batch_size, 1, height, width), dtype=torch.float32
        # ),  # mask for the image
    )

    res = dict(inputs=inputs, dynamic_shapes=shapes)
    if add_second_input:
        assert (
            add_second_input > 0
        ), f"Not implemented for add_second_input={add_second_input}."
        res["inputs2"] = get_inputs(
            model=model,
            config=config,
            batch_size=batch_size + 1,
            width=width // 2,
            height=height // 2,
            num_channels=num_channels,
            output_channels=output_channels,
            window_size=window_size,
            add_second_input=False,
            **kwargs,
        )["inputs"]
    return res


def random_input_kwargs(config: Any) -> Tuple[Dict[str, Any], Callable]:
    """
    Inputs kwargs.

    If the configuration is None, the function selects typical dimensions.
    """
    if config is not None:
        # generates mask as outputs
        if hasattr(config, "mask_decoder_config"):
            check_hasattr(
                config.mask_decoder_config,
                "hidden_size",
                "iou_head_hidden_dim",
                "iou_head_depth",
                "num_hidden_layers",
                "num_multimask_outputs",
            )
        if hasattr(config, "prompt_encoder_config"):
            check_hasattr(
                config.prompt_encoder_config,
                "hidden_size",
                "image_embedding_size",
                "image_size",
                "mask_input_channels",
            )
        if hasattr(config, "vision_config"):
            check_hasattr(
                config.vision_config,
                "image_size",
                "hidden_size",
                "intermediate_size",
                "num_hidden_layers",
                "output_channels",
                "num_channels",
                "window_size",
            )
    kwargs = dict(
        batch_size=2,
        width=1024 if config is None else config.vision_config.image_size,
        height=1024 if config is None else config.vision_config.image_size,
        num_channels=3 if config is None else config.vision_config.num_channels,
        output_channels=256 if config is None else config.vision_config.output_channels,
        window_size=14 if config is None else config.vision_config.window_size,
    )
    return kwargs, get_inputs
