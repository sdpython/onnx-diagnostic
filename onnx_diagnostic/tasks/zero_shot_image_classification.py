from typing import Any, Dict
from ..helpers.config_helper import update_config, check_hasattr

__TASK__ = "zero-shot-image-generation"


def reduce_model_config(config: Any, task: str) -> Dict[str, Any]:
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
