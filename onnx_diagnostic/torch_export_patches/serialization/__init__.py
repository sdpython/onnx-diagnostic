import re
from typing import Any, Callable, List, Set, Tuple
import torch


def _lower_name_with_(name):
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def make_serialization_function_for_dataclass(
    cls: type, supported_classes: Set[type]
) -> Tuple[Callable, Callable, Callable]:
    """
    Automatically creates serialization function for a class decorated with
    ``dataclasses.dataclass``.
    """

    def flatten_cls(obj: cls) -> Tuple[List[Any], torch.utils._pytree.Context]:
        """Serializes a ``%s`` with python objects."""
        return list(obj.values()), list(obj.keys())

    def flatten_with_keys_cls(
        obj: cls,
    ) -> Tuple[List[Tuple[torch.utils._pytree.KeyEntry, Any]], torch.utils._pytree.Context]:
        """Serializes a ``%s`` with python objects with keys."""
        values, context = list(obj.values()), list(obj.keys())
        return [
            (torch.utils._pytree.MappingKey(k), v) for k, v in zip(context, values)
        ], context

    def unflatten_cls(
        values: List[Any], context: torch.utils._pytree.Context, output_type=None
    ) -> cls:
        """Restores an instance of ``%s`` from python objects."""
        return cls(**dict(zip(context, values)))

    name = _lower_name_with_(cls.__name__)
    flatten_cls.__name__ = f"flatten_{name}"
    flatten_with_keys_cls.__name__ = f"flatten_with_keys_{name}"
    unflatten_cls.__name__ = f"unflatten_{name}"
    flatten_cls.__doc__ = flatten_cls.__doc__ % cls.__name__
    flatten_with_keys_cls.__doc__ = flatten_with_keys_cls.__doc__ % cls.__name__
    unflatten_cls.__doc__ = unflatten_cls.__doc__ % cls.__name__
    supported_classes.add(cls)
    return flatten_cls, flatten_with_keys_cls, unflatten_cls
