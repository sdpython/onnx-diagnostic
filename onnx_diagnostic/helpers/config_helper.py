from typing import Any, Dict, Tuple, Union


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
        if isinstance(v, dict):
            assert hasattr(
                config, k
            ), f"missing attribute {k!r} in config={config}, cannot update it with {v}"
            update_config(getattr(config, k), v)
        else:
            setattr(config, k, v)


def _pick(config, *atts):
    """Returns the first value found in the configuration."""
    for a in atts:
        if isinstance(a, str):
            if hasattr(config, a):
                return getattr(config, a)
        elif isinstance(a, tuple):
            if all(hasattr(config, _) for _ in a[1:]):
                return a[0]([getattr(config, _) for _ in a[1:]])
    raise AssertionError(f"Unable to find any of these {atts!r} in {config}")
