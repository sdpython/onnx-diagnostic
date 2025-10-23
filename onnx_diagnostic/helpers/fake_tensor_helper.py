from typing import Any, Optional, Tuple


def make_fake(
    x: Any, fake_mode: Optional["FakeTensorMode"] = None  # noqa: F821
) -> Tuple["FakeTensor", "FaleTensorMode"]:  # noqa: F821
    """
    Replaces all tensors by fake tensors.
    This modification happens inplace for caches.

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
        return None
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
        assert hasattr(
            x, "layers"
        ), f"Une more recent version of transformers, 'layers' not found in class {type(x)}"
        for layer in x.layers:
            assert hasattr(layer, "keys") and hasattr(layer, "values"), (
                f"Une more recent version of transformers, 'layers' "
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
