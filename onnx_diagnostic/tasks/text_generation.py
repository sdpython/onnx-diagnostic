from typing import Any, Dict
from ..helpers.config_helper import update_config, check_hasattr

__TASK__ = "text-generation"


def reduce_model_config(config: Any, task: str) -> Dict[str, Any]:
    """Reduces a model size."""
    check_hasattr(
        config,
        ("head_dim", ("hidden_size", "num_attention_heads")),
        "num_hidden_layers",
        ("num_key_value_heads", "num_attention_heads"),
        "intermediate_size",
        "hidden_size",
    )
    kwargs = dict(
        head_dim=getattr(config, "head_dim", config.hidden_size // config.num_attention_heads),
        num_hidden_layers=min(config.num_hidden_layers, 2),
        num_key_value_heads=(
            config.num_key_value_heads
            if hasattr(config, "num_key_value_heads")
            else config.num_attention_heads
        ),
        intermediate_size=(
            min(config.intermediate_size, 24576 // 4)
            if config.intermediate_size % 4 == 0
            else config.intermediate_size
        ),
        hidden_size=(
            min(config.hidden_size, 3072 // 4)
            if config.hidden_size % 4 == 0
            else config.hidden_size
        ),
    )
    update_config(config, kwargs)
    return kwargs
