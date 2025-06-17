from typing import Any, Set
from ..helpers.cache_helper import flatten_unflatten_for_dynamic_shapes


def all_dynamic_shape_from_inputs(inputs: Any, dim_prefix: Any = "d") -> Any:
    """
    Returns the dyanmic shapes for the given inputs.
    All dimensions are considered as dynamic.
    ``dim_prefix`` can be a string (the function uses it as a prefix),
    or ``torch.export.Dim.AUTO`` or ``torch.export.Dim.DYNAMIC``.
    """
    if isinstance(dim_prefix, str):
        prefixes: Set[str] = set()

        def tensor_to_shape(tensor):
            n = len(prefixes)
            p = f"{dim_prefix}_{n}"
            prefixes.add(p)
            return {i: f"{p}_{i}" for i in range(tensor.ndim)}

    else:

        def tensor_to_shape(tensor):
            return {i: dim_prefix for i in range(tensor.ndim)}  # noqa: C420

    return flatten_unflatten_for_dynamic_shapes(
        inputs, change_function=tensor_to_shape, use_dict=True
    )
