from typing import List, Optional, Union
import transformers
from huggingface_hub import HfApi
from .hub_data import __date__, load_architecture_task


def get_pretrained_config(model_id) -> str:
    """Returns the config for a model_id."""
    return transformers.AutoConfig.from_pretrained(model_id)


def task_from_arch(arch: str) -> str:
    """
    This function relies on stored information. That information needs to be refresh.

    :param arch: architecture name
    :return: task

    .. runpython::

        from onnx_diagnostic.torch_models.hub_data import __date__
        print("last refresh", __date__)

    List of supported architectures, see
    :func:`load_architecture_task
    <onnx_diagnostic.torch_models.hghub.hub_data.load_architecture_task>`.
    """
    data = load_architecture_task()
    assert arch in data, f"Architecture {arch!r} is unknown, last refresh in {__date__}"
    return data[arch]


def task_from_id(model_id: str, pretrained: bool = False) -> str:
    """
    Returns the task attached to a model id.

    :param model_id: model id
    :param pretrained: uses the config
    :return: task
    """
    if pretrained:
        config = get_pretrained_config(model_id)
        try:
            return config.pipeline_tag
        except AttributeError:
            assert config.architectures is not None and len(config.architectures) == 1, (
                f"Cannot return the task of {model_id!r}, pipeline_tag is not setup, "
                f"architectures={config.architectures} in config={config}"
            )
            return task_from_arch(config.architectures[0])
    return transformers.pipelines.get_task(model_id)


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
                            m.author or "",
                            str(m.created_at or "").split(" ")[0],
                            str(m.last_modified or "").split(" ")[0],
                            m.downloads or "",
                            m.downloads_all_time or "",
                            m.likes or "",
                            m.trending_score or "",
                            m.private or "",
                            m.gated or "",
                            ("|".join(m.tags)).replace(",", "_").replace(" ", "_"),
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
