import copy
import functools
import json
import os
from typing import Any, Dict, List, Optional, Union
import transformers
from huggingface_hub import HfApi, model_info, hf_hub_download
from ...helpers.config_helper import update_config
from . import hub_data_cached_configs
from .hub_data import __date__, __data_tasks__, load_architecture_task, __data_arch_values__


@functools.cache
def get_architecture_default_values(architecture: str):
    """
    The configuration may miss information to build the dummy inputs.
    This information returns the missing pieces.
    """
    assert architecture in __data_arch_values__, (
        f"No known default values for {architecture!r}, "
        f"expecting one architecture in {', '.join(sorted(__data_arch_values__))}"
    )
    return __data_arch_values__[architecture]


@functools.cache
def _retrieve_cached_configurations() -> Dict[str, transformers.PretrainedConfig]:
    res = {}
    for k, v in hub_data_cached_configs.__dict__.items():
        if k.startswith("_ccached_"):
            doc = v.__doc__
            res[doc] = v
    return res


def get_cached_configuration(name: str, **kwargs) -> Optional[transformers.PretrainedConfig]:
    """
    Returns cached configuration to avoid having to many accesses to internet.
    It returns None if not Cache. The list of cached models follows.

    .. runpython::

        import pprint
        from onnx_diagnostic.torch_models.hghub.hub_api import _retrieve_cached_configurations

        configs = _retrieve_cached_configurations()
        pprint.pprint(sorted(configs))
    """
    cached = _retrieve_cached_configurations()
    assert cached, "no cached configuration, which is weird"
    if name in cached:
        conf = cached[name]()
        if kwargs:
            conf = copy.deepcopy(conf)
            update_config(conf, kwargs)
        return conf
    if os.environ.get("NOHTTP", ""):
        raise AssertionError(f"Unable to find {name!r} in {sorted(cached)}")
    return None


def get_pretrained_config(
    model_id: str,
    trust_remote_code: bool = True,
    use_preinstalled: bool = True,
    subfolder: Optional[str] = None,
    **kwargs,
) -> Any:
    """
    Returns the config for a model_id.

    :param model_id: model id
    :param trust_remote_code: trust_remote_code,
        see :meth:`transformers.AutoConfig.from_pretrained`
    :param use_preinstalled: if use_preinstalled, uses this version to avoid
        accessing the network, if available, it is returned by
        :func:`get_cached_configuration`, the cached list is mostly for
        unit tests
    :param subfolder: subfolder for the given model id
    :param kwargs: additional kwargs
    :return: a configuration
    """
    if use_preinstalled:
        conf = get_cached_configuration(model_id, subfolder=subfolder, **kwargs)
        if conf is not None:
            return conf
    if subfolder:
        try:
            return transformers.AutoConfig.from_pretrained(
                model_id, trust_remote_code=trust_remote_code, subfolder=subfolder, **kwargs
            )
        except ValueError:
            # Then we try to download it.
            config = hf_hub_download(
                model_id, filename="config.json", subfolder=subfolder, **kwargs
            )
            try:
                return transformers.AutoConfig.from_pretrained(
                    config, trust_remote_code=trust_remote_code, **kwargs
                )
            except ValueError:
                # Diffusers uses a dictionayr.
                with open(config, "r") as f:
                    return json.load(f)
    return transformers.AutoConfig.from_pretrained(
        model_id, trust_remote_code=trust_remote_code, **kwargs
    )


def get_model_info(model_id) -> Any:
    """Returns the model info for a model_id."""
    return model_info(model_id)


def _guess_task_from_config(config: Any) -> Optional[str]:
    """Tries to infer a task from the configuration."""
    if hasattr(config, "bbox_loss_coefficient") and hasattr(config, "giou_loss_coefficient"):
        return "object-detection"
    if hasattr(config, "architecture") and config.architecture:
        return task_from_arch(config.architecture)
    return None


@functools.cache
def task_from_arch(arch: str, default_value: Optional[str] = None) -> str:
    """
    This function relies on stored information. That information needs to be refresh.

    :param arch: architecture name
    :param default_value: default value in case the task cannot be determined
    :return: task

    .. runpython::

        from onnx_diagnostic.torch_models.hghub.hub_data import __date__
        print("last refresh", __date__)

    List of supported architectures, see
    :func:`load_architecture_task
    <onnx_diagnostic.torch_models.hghub.hub_data.load_architecture_task>`.
    """
    data = load_architecture_task()
    if default_value is not None:
        return data.get(arch, default_value)
    assert arch in data, f"Architecture {arch!r} is unknown, last refresh in {__date__}"
    return data[arch]


def task_from_id(
    model_id: str,
    default_value: Optional[str] = None,
    pretrained: bool = False,
    fall_back_to_pretrained: bool = True,
) -> str:
    """
    Returns the task attached to a model id.

    :param model_id: model id
    :param default_value: if specified, the function returns this value
        if the task cannot be determined
    :param pretrained: uses the config
    :param fall_back_to_pretrained: falls back to pretrained config
    :return: task
    """
    if not pretrained:
        try:
            transformers.pipelines.get_task(model_id)
        except RuntimeError:
            if not fall_back_to_pretrained:
                raise
    config = get_pretrained_config(model_id)
    try:
        return config.pipeline_tag
    except AttributeError:
        guess = _guess_task_from_config(config)
        if guess is not None:
            return guess
        assert config.architectures is not None and len(config.architectures) == 1, (
            f"Cannot return the task of {model_id!r}, pipeline_tag is not setup, "
            f"architectures={config.architectures} in config={config}"
        )
        return task_from_arch(config.architectures[0], default_value=default_value)


def task_from_tags(tags: Union[str, List[str]]) -> str:
    """
    Guesses the task from the list of tags.
    If given by a string, ``|`` should be the separator.
    """
    if isinstance(tags, str):
        tags = tags.split("|")
    stags = set(tags)
    for task in __data_tasks__:
        if task in stags:
            return task
    raise ValueError(f"Unable to guess the task from tags={tags!r}")


def enumerate_model_list(
    n: int = 50,
    task: Optional[str] = None,
    library: Optional[str] = None,
    tags: Optional[Union[str, List[str]]] = None,
    search: Optional[str] = None,
    dump: Optional[str] = None,
    filter: Optional[str] = None,
    verbose: int = 0,
):
    """
    Enumerates models coming from :epkg:`huggingface_hub`.

    :param n: number of models to retrieve (-1 for all)
    :param task: see :meth:`huggingface_hub.HfApi.list_models`
    :param tags: see :meth:`huggingface_hub.HfApi.list_models`
    :param library: see :meth:`huggingface_hub.HfApi.list_models`
    :param search: see :meth:`huggingface_hub.HfApi.list_models`
    :param filter: see :meth:`huggingface_hub.HfApi.list_models`
    :param dump: dumps the result in this csv file
    :param verbose: show progress
    """
    api = HfApi()
    models = api.list_models(
        task=task,
        library=library,
        tags=tags,
        search=search,
        full=True,
        filter=filter,
        limit=n if n > 0 else None,
    )
    seen = 0
    found = 0

    if dump:
        with open(dump, "w") as f:
            f.write(
                ",".join(
                    [
                        "id",
                        "model_name",
                        "author",
                        "created_at",
                        "last_modified",
                        "downloads",
                        "downloads_all_time",
                        "likes",
                        "trending_score",
                        "private",
                        "gated",
                        "tags",
                    ]
                )
            )
            f.write("\n")

    for m in models:
        seen += 1  # noqa: SIM113
        if verbose and seen % 1000 == 0:
            print(f"[enumerate_model_list] {seen} models, found {found}")
        if verbose > 1:
            print(
                f"[enumerate_model_list]     id={m.id!r}, "
                f"library={m.library_name!r}, task={m.task!r}"
            )
        with open(dump, "a") as f:  # type: ignore
            f.write(
                ",".join(
                    map(
                        str,
                        [
                            m.id,
                            getattr(m, "model_name", "") or "",
                            m.author or "",
                            str(m.created_at or "").split(" ")[0],
                            str(m.last_modified or "").split(" ")[0],
                            m.downloads or "",
                            m.downloads_all_time or "",
                            m.likes or "",
                            m.trending_score or "",
                            m.private or "",
                            m.gated or "",
                            (
                                ("|".join(m.tags)).replace(",", "_").replace(" ", "_")
                                if m.tags
                                else ""
                            ),
                        ],
                    )
                )
            )
            f.write("\n")
        yield m
        found += 1  # noqa: SIM113
        if n >= 0:
            n -= 1
            if n == 0:
                break
