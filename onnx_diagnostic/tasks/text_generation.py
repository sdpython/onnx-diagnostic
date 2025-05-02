from typing import Any, Callable, Dict, Optional, Tuple, Union
import torch
import transformers
from ..helpers.cache_helper import (
    make_dynamic_cache,
    make_mamba_cache,
    make_sliding_window_cache,
)
from ..helpers.config_helper import update_config, check_hasattr, _pick

__TASK__ = "text-generation"


def reduce_model_config(config: Any) -> Dict[str, Any]:
    """Reduces a model size."""
    # FalconMambaConfig: use_mambapy
    check_hasattr(
        config,
        ("head_dim", ("hidden_size", "num_attention_heads"), "use_mambapy"),
        "num_hidden_layers",
        ("num_key_value_heads", "num_attention_heads", "use_mambapy"),
        "hidden_size",
        "vocab_size",
    )
    if config.__class__.__name__ == "FalconMambaConfig":
        check_hasattr(config, "conv_kernel", "state_size", "intermediate_size")  # 4 and 8
        kwargs = dict(
            num_hidden_layers=min(config.num_hidden_layers, 2),
            intermediate_size=256 if config is None else min(512, config.intermediate_size),
            hidden_size=256 if config is None else min(256, config.hidden_size),
            cls_cache="MambaCache",
            state_size=8 if config is None else getattr(config, "state_size", None),
            conv_kernel=4 if config is None else getattr(config, "conv_kernel", None),
        )
    else:
        kwargs = dict(
            head_dim=getattr(
                config, "head_dim", config.hidden_size // config.num_attention_heads
            ),
            num_hidden_layers=min(config.num_hidden_layers, 2),
            num_key_value_heads=(
                config.num_key_value_heads
                if hasattr(config, "num_key_value_heads")
                else config.num_attention_heads
            ),
            hidden_size=(
                min(config.hidden_size, 3072 // 4)
                if config.hidden_size % 4 == 0
                else config.hidden_size
            ),
        )
        if config is None or hasattr(config, "intermediate_size"):
            kwargs["intermediate_size"] = (
                min(config.intermediate_size, 24576 // 4)
                if config.intermediate_size % 4 == 0
                else config.intermediate_size
            )
    update_config(config, kwargs)
    return kwargs


def get_inputs(
    model: torch.nn.Module,
    config: Optional[Any],
    dummy_max_token_id: int,
    num_hidden_layers: int,
    batch_size: int = 2,
    sequence_length: int = 30,
    sequence_length2: int = 3,
    dynamic_rope: bool = False,
    num_key_value_heads: Optional[int] = None,
    head_dim: Optional[int] = None,
    cls_cache: Optional[Union[type, str]] = None,
    add_second_input: bool = False,
    **kwargs,  # unused
):
    """
    Generates input for task ``text-generation``.

    :param model: model to get the missing information
    :param config: configuration used to generate the model
    :param head_dim: last dimension of the cache
    :param dummy_max_token_id: dummy max token id
    :param batch_size: batch size
    :param sequence_length: sequence length
    :param sequence_length2: new sequence length
    :param dynamic_rope: use dynamic rope (see :class:`transformers.LlamaConfig`)
    :param cls_cache: cache class, by default it is
        :class:`transformers.cache_utils.DynamicCache`
    :return: dictionary
    """
    batch = torch.export.Dim("batch", min=1, max=1024)
    seq_length = "seq_length"  # torch.export.Dim("seq_length", min=1, max=4096)
    cache_length = "cache_length"  # torch.export.Dim("cache_length", min=1, max=4096)

    if config is not None and config.__class__.__name__ == "FalconMambaConfig":
        assert cls_cache in (
            "MambaCache",
            transformers.cache_utils.MambaCache,
        ), f"Unexpected value for cls_cache={cls_cache} and config={config}"
        seq_length_multiple = 8
        sequence_length = (
            (sequence_length + seq_length_multiple)
            // seq_length_multiple
            * seq_length_multiple
        )
        # sequence_inc = seq_length_multiple
        sequence_length2 = seq_length_multiple

        shapes = {
            "input_ids": {0: batch, 1: torch.export.Dim.DYNAMIC},
            "attention_mask": {
                0: batch,
                1: "cache+seq",  # cache_length + seq_length
            },
            "cache_position": {
                0: batch,
                1: "cache+seq",  # cache_length + seq_length
            },
            "cache_params": [
                [{0: batch} for _ in range(num_hidden_layers)],
                [{0: batch} for _ in range(num_hidden_layers)],
            ],
        }
        inputs = dict(
            input_ids=torch.randint(
                0, dummy_max_token_id, (batch_size, sequence_length + sequence_length2)
            ).to(torch.int64),
            attention_mask=torch.ones((batch_size, sequence_length + sequence_length2)).to(
                torch.int64
            ),
            cache_position=torch.arange(0, kwargs["conv_kernel"]).to(torch.int64),
            # .expand((batch_size, -1))
            cache_params=make_mamba_cache(
                [
                    (
                        torch.randn(
                            batch_size, kwargs["intermediate_size"], kwargs["conv_kernel"]
                        ),
                        torch.randn(
                            batch_size, kwargs["intermediate_size"], kwargs["state_size"]
                        ),
                    )
                    for i in range(num_hidden_layers)
                ]
            ),
        )
        res = dict(inputs=inputs, dynamic_shapes=shapes)
    else:
        if head_dim is None:
            assert config, "head_dim is None, the value cannot be set without a configuration"
            head_dim = config.hidden_size // config.num_attention_heads

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
        }

        make_cache = (
            make_sliding_window_cache
            if cls_cache in ("SlidingWindowCache", transformers.cache_utils.SlidingWindowCache)
            else make_dynamic_cache
        )

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
            past_key_values=make_cache(
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
        )
        res = dict(inputs=inputs, dynamic_shapes=shapes)
    if add_second_input:
        res["inputs2"] = get_inputs(
            model=model,
            config=config,
            dummy_max_token_id=dummy_max_token_id,
            num_hidden_layers=num_hidden_layers,
            batch_size=batch_size + 1,
            sequence_length=sequence_length + 1,
            sequence_length2=sequence_length2 + 1,
            dynamic_rope=dynamic_rope,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            cls_cache=cls_cache,
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
            ("num_attention_heads", "use_mambapy"),
            ("num_key_value_heads", "num_attention_heads", "use_mambapy"),
            "hidden_size",
        )
    if config.__class__.__name__ == "FalconMambaConfig":
        check_hasattr(config, "conv_kernel", "state_size", "intermediate_size")  # 4 and 8
        kwargs = dict(
            batch_size=2,
            sequence_length=30,
            sequence_length2=3,
            dummy_max_token_id=31999 if config is None else (config.vocab_size - 1),
            num_hidden_layers=4 if config is None else config.num_hidden_layers,
            intermediate_size=256 if config is None else config.intermediate_size,
            cls_cache="MambaCache",
            state_size=8 if config is None else getattr(config, "state_size", None),
            conv_kernel=8 if config is None else getattr(config, "conv_kernel", None),
        )
    else:
        kwargs = dict(
            batch_size=2,
            sequence_length=30,
            sequence_length2=3,
            head_dim=(
                16
                if config is None
                else getattr(
                    config, "head_dim", config.hidden_size // config.num_attention_heads
                )
            ),
            dummy_max_token_id=31999 if config is None else (config.vocab_size - 1),
            num_hidden_layers=4 if config is None else config.num_hidden_layers,
            num_key_value_heads=(
                24
                if config is None
                else _pick(config, "num_key_value_heads", "num_attention_heads")
            ),
            hidden_size=512 if config is None else config.hidden_size,
        )
        if config is None or hasattr(config, "intermediate_size"):
            kwargs["intermediate_size"] = (
                1024 if config is None else config.intermediate_size,
            )

    return kwargs, get_inputs
