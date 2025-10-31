from typing import Any, Dict, List, Set, Optional, Tuple, Union
from ..helpers.cache_helper import flatten_unflatten_for_dynamic_shapes
from .dynamic_shapes import ModelInputs


def all_dynamic_shapes_from_inputs(inputs: Any, dim_prefix: Any = "d") -> Any:
    """
    Returns the dynamic shapes for the given inputs.
    All dimensions are considered as dynamic.
    ``dim_prefix`` can be a string (the function uses it as a prefix),
    or ``torch.export.Dim.AUTO`` or ``torch.export.Dim.DYNAMIC``.
    Depending on the version of transformers, serializations function
    of DynamicCache class is automatically serialized or not (>= 4.51, < 4.55).

    .. runpython::
        :showcode:

        import pprint
        import torch
        from onnx_diagnostic.helpers.cache_helper import make_dynamic_cache
        from onnx_diagnostic.export.shape_helper import all_dynamic_shapes_from_inputs
        from onnx_diagnostic.torch_export_patches import torch_export_patches

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
        with torch_export_patches(patch_transformers=True):
            ds = all_dynamic_shapes_from_inputs(inputs)
        pprint.pprint(ds)

    For this function to work, patches must be enabled if :epkg:`transformers`
    does not implement the serialization functions.

    .. runpython::
        :showcode:

        import pprint
        import torch
        from onnx_diagnostic.helpers.cache_helper import (
            make_dynamic_cache,
            make_encoder_decoder_cache,
            make_mamba_cache,
            make_sliding_window_cache,
            make_static_cache,
        )
        from onnx_diagnostic.export.shape_helper import all_dynamic_shapes_from_inputs
        from onnx_diagnostic.torch_export_patches import torch_export_patches

        caches = [
            make_dynamic_cache(
                [
                    (torch.rand((4, 4, 4)), torch.rand((4, 4, 4))),
                    (torch.rand((4, 4, 4)), torch.rand((4, 4, 4))),
                    (torch.rand((4, 4, 4)), torch.rand((4, 4, 4))),
                ]
            ),
            make_encoder_decoder_cache(
                make_dynamic_cache(
                    [
                        (torch.rand((4, 4, 4)), torch.rand((4, 4, 4))),
                        (torch.rand((4, 4, 4)), torch.rand((4, 4, 4))),
                        (torch.rand((4, 4, 4)), torch.rand((4, 4, 4))),
                    ]
                ),
                make_dynamic_cache(
                    [
                        (torch.rand((5, 5, 5)), torch.rand((5, 5, 5))),
                        (torch.rand((5, 5, 5)), torch.rand((5, 5, 5))),
                        (torch.rand((5, 5, 5)), torch.rand((5, 5, 5))),
                    ]
                ),
            ),
            make_sliding_window_cache(
                [
                    (torch.rand((4, 5, 6, 7)), torch.rand((4, 5, 6, 7))),
                    (torch.rand((4, 5, 6, 7)), torch.rand((4, 5, 6, 7))),
                    (torch.rand((4, 5, 6, 7)), torch.rand((4, 5, 6, 7))),
                ]
            ),
            make_static_cache(
                [
                    (torch.rand((4, 5, 6, 7)), torch.rand((4, 5, 6, 7))),
                    (torch.rand((4, 5, 6, 7)), torch.rand((4, 5, 6, 7))),
                    (torch.rand((4, 5, 6, 7)), torch.rand((4, 5, 6, 7))),
                ],
                max_cache_len=15,
            ),
            make_mamba_cache(
                [
                    (torch.rand((4, 4, 4)), torch.rand((4, 4, 4))),
                    (torch.rand((4, 4, 4)), torch.rand((4, 4, 4))),
                    (torch.rand((4, 4, 4)), torch.rand((4, 4, 4))),
                ]
            ),
        ]

        with torch_export_patches(patch_transformers=True):
            for cache in caches:
                print(f"-- {cache.__class__.__name__}")
                pprint.pprint(all_dynamic_shapes_from_inputs(cache))
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

    This function returns something equivalent to function
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
        pprint.pprint(ds.dynamic_shapes(data["model"], (), data["inputs"]))
    """
    mi = ModelInputs(None, inputs)
    return mi.guess_dynamic_shapes(auto=auto)


def make_fake_with_dynamic_dimensions(
    x: Any, dynamic_shapes: Any, context: Optional["FakeTensorContext"] = None  # noqa: F821
) -> Tuple[Any, "FakeTensorContext"]:  # noqa: F821
    """
    Replaces all tensors by fake tensor respecting the same
    constraints as the following dynamic shapes.
    This uses function :func:`onnx_diagnostic.helpers.fake_tensor_helper.make_fake`.
    Parameter ``existing`` is used to reused the same object when the dynamic
    dimension is given the same name as another one.

    A simple tensor:

    .. runpython::
        :showcode:

        import torch
        from onnx_diagnostic.helpers.cache_helper import make_dynamic_cache
        from onnx_diagnostic.export.shape_helper import make_fake_with_dynamic_dimensions

        inputs, _ = make_fake_with_dynamic_dimensions(
            torch.rand((2, 3, 4, 5), dtype=torch.float32),
            {0: "batch", 2: "cache_length"},
        )
        print(inputs)

    Two tensors:

    .. runpython::
        :showcode:

        import torch
        from onnx_diagnostic.helpers.cache_helper import make_dynamic_cache
        from onnx_diagnostic.export.shape_helper import make_fake_with_dynamic_dimensions

        inputs, _ = make_fake_with_dynamic_dimensions(
            (
                torch.rand((2, 3, 4, 5), dtype=torch.float32),
                torch.rand((2, 3, 4, 5), dtype=torch.float32),
            ),
            ({0: "batch", 2: "cache_length"}, {0: "batch", 2: "cache_length"}),
        )
        print(inputs)

    With a cache:

    .. runpython::
        :showcode:

        import pprint
        import torch
        from onnx_diagnostic.helpers.cache_helper import make_dynamic_cache
        from onnx_diagnostic.export.shape_helper import make_fake_with_dynamic_dimensions

        inputs, _ = make_fake_with_dynamic_dimensions(
            dict(
                input_ids=torch.randint(30360, size=(2, 3), dtype=torch.int64),
                attention_mask=torch.randint(1, size=(2, 33), dtype=torch.int64),
                position_ids=torch.randint(32, size=(2, 3), dtype=torch.int64),
                past_key_values=make_dynamic_cache(
                    [
                        (
                            torch.rand((2, 32, 30, 96), dtype=torch.float16),
                            torch.rand((2, 32, 30, 96), dtype=torch.float16),
                        ),
                        (
                            torch.rand((2, 32, 30, 96), dtype=torch.float16),
                            torch.rand((2, 32, 30, 96), dtype=torch.float16),
                        ),
                    ]
                ),
            ),
            dynamic_shapes={
                "input_ids": {0: "batch", 1: "seq_length"},
                "attention_mask": {0: "batch", 1: "cache+seq"},
                "position_ids": {0: "batch", 1: "seq_length"},
                "past_key_values": [
                    {0: "batch", 2: "cache_length"},
                    {0: "batch", 2: "cache_length"},
                    {0: "batch", 2: "cache_length"},
                    {0: "batch", 2: "cache_length"},
                ],
            },
        )
        pprint.pprint(inputs)
    """
    if x is None:
        return None, None
    if context is None:
        from ..helpers.fake_tensor_helper import FakeTensorContext

        context = FakeTensorContext()

    return context.make_fake_with_dynamic_dimensions(x, dynamic_shapes), context
