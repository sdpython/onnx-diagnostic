from typing import Any, Dict, Optional, Union
from .hghub.model_inputs import random_input_kwargs


def get_inputs_for_task(task: str, config: Optional[Any] = None) -> Dict[str, Any]:
    """
    Returns dummy inputs for a specific task.

    :param task: requested task
    :param config: returns dummy inputs for a specific config if available
    :return: dummy inputs and dynamic shapes
    """
    kwargs, f = random_input_kwargs(config, task)
    return f(model=None, config=config, **kwargs)


def validate_model(
    model_id: str,
    task: Optional[str] = None,
    do_run: bool = False,
    do_export: bool = False,
    do_same: bool = False,
    verbose: int = 0,
) -> Dict[str, Union[int, float, str]]:
    """
    Validates a model.


    """
