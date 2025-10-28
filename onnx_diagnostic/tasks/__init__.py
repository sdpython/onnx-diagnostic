from typing import Any, Callable, Dict, List, Tuple
from . import (
    automatic_speech_recognition,
    feature_extraction,
    fill_mask,
    image_classification,
    image_text_to_text,
    image_to_video,
    mask_generation,
    mixture_of_expert,
    object_detection,
    sentence_similarity,
    summarization,
    text_classification,
    text_generation,
    text_to_image,
    text2text_generation,
    zero_shot_image_classification,
)

__TASKS__ = [
    automatic_speech_recognition,
    feature_extraction,
    fill_mask,
    image_classification,
    image_text_to_text,
    image_to_video,
    mask_generation,
    mixture_of_expert,
    object_detection,
    sentence_similarity,
    summarization,
    text_classification,
    text_generation,
    text_to_image,
    text2text_generation,
    zero_shot_image_classification,
]


def supported_tasks() -> List[str]:
    "Returns the list of supported tasks."
    return sorted(mod.__TASK__ for mod in __TASKS__)


def reduce_model_config(config: Any, task: str) -> Dict[str, Any]:
    """Reduces a model size."""
    head_size0 = (
        config.head_dim
        if hasattr(config, "head_dim") and config.head_dim
        else (
            config.hidden_size // config.num_attention_heads
            if hasattr(config, "hidden_size") and hasattr(config, "num_attention_heads")
            else None
        )
    )
    tasks = {mod.__TASK__: mod.reduce_model_config for mod in __TASKS__}
    assert task in tasks, f"Task {task!r} not found in {sorted(tasks)}"
    res = tasks[task](config)
    if head_size0 and "head_dim" in res:
        head_size = (
            config.head_dim
            if hasattr(config, "head_dim") and config.head_dim
            else config.hidden_size // config.num_attention_heads
        )
        assert head_size0 == head_size or head_size % 16 == 0, (
            f"head_size should be a multiple of 16 "
            f"(head_size0={head_size0}), res={res}, "
            f"config=\n{config}"
        )
    return res


def random_input_kwargs(config: Any, task: str) -> Tuple[Dict[str, Any], Callable]:
    """
    Inputs kwargs.
    If the configuration is None, the function selects typical dimensions.
    It returns parameters and a function. The function creates dummy inputs
    if it receives the parameters returned as a first result.

    .. code-block:: python

        config = get_pretrained_config(model_id)
        task = task = task_from_id(name)
        kwargs, fct = random_input_kwargs(config, task)
        res = fct(model, config, add_second_input=False, **kwargs)
    """
    tasks = {mod.__TASK__: mod.random_input_kwargs for mod in __TASKS__}
    assert task in tasks, f"Task {task!r} not found in {sorted(tasks)}"
    return tasks[task](config)
