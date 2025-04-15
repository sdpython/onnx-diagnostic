from typing import Any, Dict
from ..helpers.config_helper import update_config

__TASK__ = "automatic_speech_recognition"


def reduce_model_config(config: Any, task: str) -> Dict[str, Any]:
    """Reduces a model size."""
    kwargs = {}
    if hasattr(config, "num_decoder_layers"):
        config.num_decoder_layers = min(config.num_decoder_layers, 2)
    if hasattr(config, "decoder_layers"):
        config.decoder_layers = min(config.decoder_layers, 2)
    if hasattr(config, "num_hidden_layers"):
        config.num_hidden_layers = min(config.num_hidden_layers, 2)
    update_config(config, kwargs)
    return kwargs
