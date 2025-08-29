import functools
import importlib
import inspect
import os
import re
from typing import Any, Callable, Dict, Optional, Tuple, Union
import transformers


def check_hasattr(config: Any, *args: Union[str, Tuple[Any, ...]]):
    """
    Checks the confiugation has all the attributes in ``args``.
    Raises an exception otherwise.
    """
    for a in args:
        assert isinstance(a, (str, tuple)), f"unexpected type {type(a)} in {args!r}"
        if isinstance(a, str):
            assert (isinstance(config, dict) and a in config) or hasattr(
                config, a
            ), f"Missing attribute {a!r} in\n{config}"
        elif isinstance(a, tuple):
            assert any(
                (isinstance(name, str) and hasattr(config, name))
                or all(hasattr(config, _) for _ in name)
                for name in a
            ), f"All attributes in {a!r} are missing from\n{config}"


def update_config(config: Any, mkwargs: Dict[str, Any]):
    """Updates a configuration with different values."""
    for k, v in mkwargs.items():
        if k == "attn_implementation":
            config._attn_implementation = v
            if getattr(config, "_attn_implementation_autoset", False):
                config._attn_implementation_autoset = False
            continue
        if isinstance(v, dict):
            if not hasattr(config, k) or getattr(config, k) is None:
                setattr(config, k, v)
                continue
            existing = getattr(config, k)
            if type(existing) is dict:
                existing.update(v)
            else:
                update_config(getattr(config, k), v)
            continue
        if type(config) is dict:
            config[k] = v
        else:
            setattr(config, k, v)


def _pick(config, *atts, exceptions: Optional[Dict[str, Callable]] = None):
    """Returns the first value found in the configuration."""
    if (
        exceptions
        and hasattr(config, "architectures")
        and len(config.architectures) == 1
        and config.architectures[0] in exceptions
    ):
        excs = exceptions[config.architectures[0]]
        return excs(config)
    for a in atts:
        if isinstance(a, str):
            if hasattr(config, a):
                return getattr(config, a)
        elif isinstance(a, tuple):
            if all(hasattr(config, _) for _ in a[1:]):
                return a[0]([getattr(config, _) for _ in a[1:]])
    raise AssertionError(f"Unable to find any of these {atts!r} in {config}")


def pick(config, name: str, default_value: Any) -> Any:
    """
    Returns the value of a attribute if config has it
    otherwise the default value.
    """
    if not config:
        return default_value
    if type(config) is dict:
        return config.get(name, default_value)
    return getattr(config, name, default_value)


@functools.cache
def config_class_from_architecture(arch: str, exc: bool = False) -> Optional[type]:
    """
    Retrieves the configuration class for a given architecture.

    :param arch: architecture (clas name)
    :param exc: raise an exception if not found
    :return: type
    """
    cls = getattr(transformers, arch)
    mod_name = cls.__module__
    mod = importlib.import_module(mod_name)
    source = inspect.getsource(mod)
    reg = re.compile("config: ([A-Za-z0-9]+)")
    fall = reg.findall(source)
    if len(fall) == 0:
        assert not exc, (
            f"Unable to guess Configuration class name for arch={arch!r}, "
            f"module={mod_name!r}, no candidate, source is\n{source}"
        )
        return None
    unique = set(fall)
    assert len(unique) == 1, (
        f"Unable to guess Configuration class name for arch={arch!r}, "
        f"module={mod_name!r}, found={unique} (#{len(unique)}), "
        f"source is\n{source}"
    )
    cls_name = unique.pop()
    return getattr(transformers, cls_name)


def default_num_hidden_layers():
    """
    Returns the default number of layers.
    It is lower when the unit tests are running
    when ``UNITTEST_GOING=1``.
    """
    import torch

    if torch.cuda.is_available():
        capa = torch.cuda.get_device_capability(0)
        if capa[0] < 9:
            return 2
    return 2 if os.environ.get("UNITTEST_GOING", "0") == "1" else 4


def build_diff_config(config0, config1):
    """
    Returns all the modified values between two configuration
    """
    import torch

    diff = {}
    for k in config0:
        assert isinstance(k, str), f"k={k!r}, wrong type in {config0}"
        if k not in config1:
            v0 = getattr(config0, k) if hasattr(config0, k) else config0[k]
            diff[k] = f"-{v0}"
    for k in config1:
        assert isinstance(k, str), f"k={k!r}, wrong type in {config1}"
        if k not in config0:
            v1 = getattr(config1, k) if hasattr(config1, k) else config1[k]
            diff[k] = f"+{v1}"
    for k in config0:
        if k not in config1:
            continue
        v0 = getattr(config0, k) if hasattr(config0, k) else config0[k]
        v1 = getattr(config1, k) if hasattr(config1, k) else config1[k]
        if (
            v0 is None
            or v1 is None
            or isinstance(v1, (float, int, bool, str, list, tuple, torch.dtype))
            or (
                isinstance(v0, dict)
                and isinstance(v1, dict)
                and all(isinstance(k, int) for k in v1)
            )
        ):
            if v1 != v0:
                diff[k] = f"{v0} -> {v1}"
        else:
            d = build_diff_config(v0, v1)
            if d:
                diff[k] = d
    return diff
