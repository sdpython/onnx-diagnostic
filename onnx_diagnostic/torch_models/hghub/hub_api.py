from typing import List, Optional, Union
from huggingface_hub import HfApi


def enumerate_model_list(
    n: int = 50,
    task: Optional[str] = None,
    library: Optional[str] = None,
    tags: Optional[Union[str, List[str]]] = None,
    dump: Optional[str] = None,
    verbose: int = 0,
):
    """
    Enumerates models coming from :epkg:`huggingface_hub`.

    :param n: number of models to retrieve (-1 for all)
    :param task: see :meth:`huggingface_hub.HfApi.list_models`
    :param tags: see :meth:`huggingface_hub.HfApi.list_models`
    :param library: see :meth:`huggingface_hub.HfApi.list_models`
    :param dump: dumps the result in this csv file
    :param verbose: show progress
    """
    api = HfApi()
    models = api.list_models(task=task, library=library, tags=tags)
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
        with open(dump, "a") as f:
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
