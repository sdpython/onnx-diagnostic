from typing import Any, Set
from ..helpers.cache_helper import flatten_unflatten_for_dynamic_shapes


def all_dynamic_shape_from_inputs(inputs: Any, dim_prefix: Any = "d") -> Any:
    """
    Returns the dynamic shapes for the given inputs.
    All dimensions are considered as dynamic.
    ``dim_prefix`` can be a string (the function uses it as a prefix),
    or ``torch.export.Dim.AUTO`` or ``torch.export.Dim.DYNAMIC``.

    .. runpython::
        :showcode:

        import pprint
        import torch
        from onnx_diagnostic.helpers.cache_helper import make_dynamic_cache
        from onnx_diagnostic.export.shape_helper import all_dynamic_shape_from_inputs

        bsize, nheads, slen, dim = 2, 1, 30, 96
        inputs = dict(
            input_ids=torch.randint(15, size=(2, 3), dtype=torch.int64),
            attention_mask=torch.randint(1, size=(2, 33), dtype=torch.int64),
            position_ids=torch.arange(3, dtype=torch.int64),
            past_key_values=make_dynamic_cache(
                [(torch.randn(bsize, nheads, slen, dim),
                  torch.randn(bsize, nheads, slen, dim))]
            ),
        )
        ds = all_dynamic_shape_from_inputs(inputs)
        pprint.pprint(ds)
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
