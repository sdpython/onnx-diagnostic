from typing import Any, Callable, Dict, List, Tuple
from . import (
    automatic_speech_recognition,
    feature_extraction,
    fill_mask,
    image_classification,
    image_text_to_text,
    mixture_of_expert,
    object_detection,
    sentence_similarity,
    text_classification,
    text_generation,
    text2text_generation,
    zero_shot_image_classification,
)

__TASKS__ = [
    automatic_speech_recognition,
    feature_extraction,
    fill_mask,
    image_classification,
    image_text_to_text,
    mixture_of_expert,
    object_detection,
    sentence_similarity,
    text_classification,
    text_generation,
    text2text_generation,
    zero_shot_image_classification,
]


def supported_tasks() -> List[str]:
    "Returns the list of supported tasks."
    return sorted(mod.__TASK__ for mod in __TASKS__)


def reduce_model_config(config: Any, task: str) -> Dict[str, Any]:
    """Reduces a model size."""
    tasks = {mod.__TASK__: mod.reduce_model_config for mod in __TASKS__}
    assert task in tasks, f"Task {task!r} not found in {sorted(tasks)}"
    return tasks[task](config)


def random_input_kwargs(config: Any, task: str) -> Tuple[Dict[str, Any], Callable]:
    """
    Inputs kwargs.
    If the configuration is None, the function selects typical dimensions.
    It returns parameters and a function. The function creates dummy inputs
    if it receives the parameters returned as a first result.
    """
    tasks = {mod.__TASK__: mod.random_input_kwargs for mod in __TASKS__}
    assert task in tasks, f"Task {task!r} not found in {sorted(tasks)}"
    return tasks[task](config)
