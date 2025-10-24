from typing import Any, Dict, Optional, Tuple


_UNIQUE = set()


def _unique():
    i = 129 + 1
    while i in _UNIQUE:
        i += 1
    _UNIQUE.add(i)
    return i


def fake_reshape(
    true_tensor: "Tensor",  # noqa: F821
    sh: Dict[int, Any],  # noqa: F821
    fake_tensor: Optional["FakeTensor"] = None,  # noqa: F821
    fake_mode: Optional["FakeTensorMode"] = None,  # noqa: F821
) -> "FakeTensor":  # noqa: F821
    """
    Changes the shape of a true tensor to make it dynamic.

    :param true_tensor: true tensor
    :param sh: dynamic shape
    :param fake_tensor: fake tensor, if None, make a fake one
    :param fake_mode: fake tensor mode
    :return: fake tensor
    """
    import torch

    # deal with 0/1
    for i in sh:
        if true_tensor.shape[i] <= 1:
            expanded_shape = list(true_tensor.shape)
            expanded_shape[i] = _unique()
            true_tensor = torch.empty(
                tuple(expanded_shape), dtype=true_tensor.dtype, device=true_tensor.device
            )

    # deal with equivalent dimension
    new_shape = list(true_tensor.shape)
    mapping = {}
    for i, s in sh.items():
        d = true_tensor.shape[i]
        if d not in mapping:
            mapping[d] = s
        elif mapping[d] != s:
            d = _unique()
            mapping[d] = s
            new_shape[i] = d
    true_tensor = torch.empty(
        tuple(new_shape), dtype=true_tensor.dtype, device=true_tensor.device
    )

    # now switch to FakeTensor
    if fake_mode is None:
        from torch.fx.experimental.symbolic_shapes import ShapeEnv
        from torch._subclasses.fake_tensor import FakeTensorMode

        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(shape_env=shape_env)
    if fake_tensor is None:
        fake_tensor = fake_mode.from_tensor(true_tensor, static_shapes=False)
    assert fake_mode is not None, "fake_mode must be provided"

    new_shape = list(true_tensor.shape)
    for i in sh:
        new_shape[i] = fake_tensor.shape[i]

    reduced_tensor = fake_mode.from_tensor(true_tensor, static_shapes=True).sum(
        axis=tuple(sorted(sh)), keepdim=True
    )
    return reduced_tensor.expand(*new_shape)


def make_fake(
    x: Any, fake_mode: Optional["FakeTensorMode"] = None  # noqa: F821
) -> Tuple[Optional["FakeTensor"], Optional["FakeTensorMode"]]:  # noqa: F821
    """
    Replaces all tensors by fake tensors.
    This modification happens inplace for caches.
    This function is only implemented for cache with
    ``transformers>=4.55``.

    .. runpython::
        :showcode:

        from onnx_diagnostic.helpers.fake_tensor_helper import make_fake

        inputs, _ = make_fake(
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
            )
        )
        print(inputs)
    """
    if x is None:
        return None, None
    if fake_mode is None:
        from torch.fx.experimental.symbolic_shapes import ShapeEnv
        from torch._subclasses.fake_tensor import FakeTensorMode

        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(shape_env=shape_env)

    if isinstance(x, (list, tuple)):
        return x.__class__([make_fake(i, fake_mode=fake_mode)[0] for i in x]), fake_mode
    if isinstance(x, dict):
        return {k: make_fake(v, fake_mode=fake_mode)[0] for k, v in x.items()}, fake_mode

    if x.__class__.__name__ in {"DynamicCache", "StaticCache", "HybridCache"}:
        assert hasattr(x, "layers"), (
            f"Une more recent version of transformers (>=4.55), "
            f"'layers' not found in class {type(x)}"
        )
        for layer in x.layers:
            assert hasattr(layer, "keys") and hasattr(layer, "values"), (
                f"Une more recent version of transformers (>=4.55), 'layers' "
                f"not found in class {type(layer)} ({dir(layer)})"
            )
            layer.keys = make_fake(layer.keys, fake_mode=fake_mode)[0]
            layer.values = make_fake(layer.values, fake_mode=fake_mode)[0]
        return x, fake_mode
    if x.__class__.__name__ == "EncoderDecoderCache":
        make_fake(x.self_attention_cache, fake_mode=fake_mode)
        make_fake(x.cross_attention_cache, fake_mode=fake_mode)
        return x, fake_mode
    if hasattr(x, "shape"):
        t = fake_mode.from_tensor(x, static_shapes=False)
        return t, fake_mode
    from . import string_type

    raise TypeError(
        f"Unexpected type {type(x)} for x, content is {string_type(x, with_shape=True)}"
    )
