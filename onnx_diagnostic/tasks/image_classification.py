from typing import Any, Dict
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
