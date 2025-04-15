from typing import Any, Dict
from ..helpers.config_helper import update_config

__TASK__ = "image-text-to-text"


def reduce_model_config(config: Any, task: str) -> Dict[str, Any]:
    """Reduces a model size."""
    kwargs = {}
    if hasattr(config, "num_hidden_layers"):
        config.num_hidden_layers = min(config.num_hidden_layers, 2)
    if hasattr(config, "vision_config") and hasattr(config.vision_config, "num_hidden_layers"):
        config.vision_config.num_hidden_layers = min(config.vision_config.num_hidden_layers, 2)
    update_config(config, kwargs)
    return kwargs
