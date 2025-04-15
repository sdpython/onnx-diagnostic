from typing import Any, Dict
from . import (
    automatic_speech_recognition,
    image_classification,
    image_text_to_text,
    text_generation,
    text2text_generation,
    zero_shot_image_classification,
)

__TASKS__ = [
    automatic_speech_recognition,
    image_classification,
    image_text_to_text,
    text_generation,
    text2text_generation,
    zero_shot_image_classification,
]


def reduce_model_config(config: Any, task: str) -> Dict[str, Any]:
    """Reduces a model size."""
    tasks = {mod.__TASK__: mod.reduce_model_config for mod in __TASKS__}
    assert task in tasks, f"Task {task!r} not found in {sorted(tasks)}"
    return tasks[task](config, task)
