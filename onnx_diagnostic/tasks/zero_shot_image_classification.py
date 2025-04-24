from typing import Any, Callable, Dict, Optional, Tuple
import torch
from ..helpers.config_helper import update_config, check_hasattr

__TASK__ = "zero-shot-image-classification"


def reduce_model_config(config: Any) -> Dict[str, Any]:
    """Reduces a model size."""
    check_hasattr(config, "vision_config", "text_config")
    check_hasattr(config.vision_config, "num_hidden_layers", "num_attention_heads")
    check_hasattr(config.text_config, "num_hidden_layers", "num_attention_heads")
    kwargs = dict(
        vision_config=dict(
            num_hidden_layers=min(2, config.vision_config.num_hidden_layers),
            num_attention_heads=min(2, config.vision_config.num_attention_heads),
        ),
        text_config=dict(
            num_hidden_layers=min(2, config.text_config.num_hidden_layers),
            num_attention_heads=min(2, config.text_config.num_attention_heads),
        ),
    )
    update_config(config, kwargs)
    return kwargs


def get_inputs(
    model: torch.nn.Module,
    config: Optional[Any],
    dummy_max_token_id: int,
    batch_size: int = 2,
    sequence_length: int = 30,
    input_width: int = 224,
    input_height: int = 224,
    input_channels: int = 3,
    batch_size_image=3,
    add_second_input: bool = False,
    **kwargs,  # unused
):
    """
    Generates inputs for task ``zero-short-image-classification``.

    :param model: model to get the missing information
    :param config: configuration used to generate the model
    :param dummy_max_token_id: vocabulary size
    :param batch_size: batch size
    :param sequence_length: sequence length
    :param batch_size_image: number of images
    :param input_channels: input channel
    :param input_width: input width
    :param input_height: input height
    :return: dictionary

    # input_ids:T7s2x7
    # attention_mask:T7s2x7
    # pixel_values:T1s2x3x224x224
    """
    assert isinstance(
        input_width, int
    ), f"Unexpected type for input_width {type(input_width)}{config}"
    assert isinstance(
        input_width, int
    ), f"Unexpected type for input_height {type(input_height)}{config}"

    batch = torch.export.Dim("batch", min=1, max=1024)
    seq_length = "seq_length"  # torch.export.Dim("seq_length", min=1, max=4096)
    shapes = {
        "input_ids": {0: batch, 1: seq_length},
        "attention_mask": {0: batch, 1: seq_length},
        "pixel_values": {
            0: torch.export.Dim("batch_img", min=1, max=1024),
            # 2: torch.export.Dim("width", min=1, max=4096),
            # 3: torch.export.Dim("height", min=1, max=4096),
        },
    }
    inputs = dict(
        input_ids=torch.randint(0, dummy_max_token_id, (batch_size, sequence_length)).to(
            torch.int64
        ),
        attention_mask=torch.ones((batch_size, sequence_length)).to(torch.int64),
        pixel_values=torch.randn(
            batch_size_image, input_channels, input_width, input_height
        ).clamp(-1, 1),
    )
    res = dict(inputs=inputs, dynamic_shapes=shapes)
    if add_second_input:
        res["inputs2"] = get_inputs(
            model=model,
            config=config,
            dummy_max_token_id=dummy_max_token_id,
            batch_size=batch_size + 1,
            sequence_length=sequence_length + 1,
            input_width=input_width,
            input_height=input_height,
            input_channels=input_channels,
            batch_size_image=batch_size_image + 1,
            **kwargs,
        )["inputs"]
    return res


def random_input_kwargs(config: Any) -> Tuple[Dict[str, Any], Callable]:
    """
    Inputs kwargs.

    If the configuration is None, the function selects typical dimensions.
    """
    if config is not None:
        check_hasattr(config, "vision_config", "text_config")
        check_hasattr(config.vision_config, "image_size", "num_channels")
        check_hasattr(config.text_config, "vocab_size")
    kwargs = dict(
        batch_size=2,
        batch_size_image=3,
        sequence_length=30,
        dummy_max_token_id=(49408 if config is None else (config.text_config.vocab_size - 1)),
        input_width=224 if config is None else config.vision_config.image_size,
        input_height=224 if config is None else config.vision_config.image_size,
        input_channels=3 if config is None else config.vision_config.num_channels,
    )
    return kwargs, get_inputs
