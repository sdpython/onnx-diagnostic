from typing import Any, Dict, List, Set, Tuple, Union
from ..helpers.cache_helper import flatten_unflatten_for_dynamic_shapes
from .dynamic_shapes import ModelInputs


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


def guess_dynamic_shapes_from_inputs(
    inputs: List[Any], auto: Union[bool, str] = False
) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    """
    Guesses which dimension is dimension from a set of inputs.
    Every dimension having different values over multiple sets
    of inputs. Every dimension not changing remains static.

    :param inputs: a list of input sets
    :param auto: True for ``torch.export.Dim.AUTO``,
        False for ``torch.export.Dim.DYNAMIC``,
        a string to get a unique string for every dynamic dimension
    :return: args and kwargs

    .. runpython::
        :showcode:

        import pprint
        import torch
        from onnx_diagnostic.helpers.cache_helper import make_dynamic_cache
        from onnx_diagnostic.export.shape_helper import guess_dynamic_shapes_from_inputs

        bsize, nheads, slen, dim = 2, 1, 30, 96
        inputs1 = dict(
            input_ids=torch.randint(15, size=(2, 3), dtype=torch.int64),
            attention_mask=torch.randint(1, size=(2, 33), dtype=torch.int64),
            position_ids=torch.arange(3, dtype=torch.int64),
            past_key_values=make_dynamic_cache(
                [
                    (
                        torch.randn(bsize, nheads, slen, dim),
                        torch.randn(bsize, nheads, slen, dim),
                    ),
                ]
            ),
        )
        bsize, nheads, slen, dim = 3, 1, 33, 96
        inputs2 = dict(
            input_ids=torch.randint(15, size=(3, 4), dtype=torch.int64),
            attention_mask=torch.randint(1, size=(3, 34), dtype=torch.int64),
            position_ids=torch.arange(4, dtype=torch.int64),
            past_key_values=make_dynamic_cache(
                [
                    (
                        torch.randn(bsize, nheads, slen, dim),
                        torch.randn(bsize, nheads, slen, dim),
                    ),
                ]
            ),
        )
        ds = guess_dynamic_shapes_from_inputs([inputs1, inputs2], auto="d")
        pprint.pprint(ds)

    This function returns sometihng equivalent to function
    :class:`torch.export.dynamic_shapes.AdditionalInputs` but this
    one needs a model.

    .. runpython::
        :showcode:

        import pprint
        import torch
        from onnx_diagnostic.helpers.cache_helper import make_dynamic_cache
        from onnx_diagnostic.export.shape_helper import guess_dynamic_shapes_from_inputs
        from onnx_diagnostic.torch_models.hghub import get_untrained_model_with_inputs

        data = get_untrained_model_with_inputs("arnir0/Tiny-LLM", add_second_input=True)
        ds = torch.export.dynamic_shapes.AdditionalInputs()
        ds.add((), data["inputs"])
        ds.add((), data["inputs2"])
        pprint.pprint(ds.dynamic_shapes(data["model"], (), inputs1))
    """
    mi = ModelInputs(None, inputs)
    return mi.guess_dynamic_shapes(auto=auto)
