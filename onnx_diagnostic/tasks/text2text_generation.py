from typing import Any, Dict
from ..helpers.config_helper import update_config, check_hasattr

__TASK__ = "text-to-text-generation"


def reduce_model_config(config: Any, task: str) -> Dict[str, Any]:
    """Reduces a model size."""
    kwargs = {}
    if hasattr(config, "num_decoder_layers"):
        check_hasattr(config.num_decoder_layers, "num_decoder_layers")
        config.num_decoder_layers = min(config.num_decoder_layers, 2)
    if hasattr(config, "num_hidden_layers"):
        check_hasattr(config.num_decoder_layers, "num_hidden_layers")
        config.num_hidden_layers = min(config.num_hidden_layers, 2)
    update_config(config, kwargs)
    return kwargs
