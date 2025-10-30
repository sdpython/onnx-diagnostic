from typing import Any, Dict, Optional, Set, Tuple


class FakeTensorContext:
    """Stores information used to reused same dimension for the same dimension names."""

    def __init__(self, fake_mode: Optional["FakeTensorMode"] = None):  # noqa: F821
        if fake_mode is None:
            from torch.fx.experimental.symbolic_shapes import ShapeEnv
            from torch._subclasses.fake_tensor import FakeTensorMode

            shape_env = ShapeEnv()
            self.fake_mode = FakeTensorMode(shape_env=shape_env)
        else:
            self.fake_mode = fake_mode
        self._candidates = self._first_primes()
        self._unique_: Set[str] = set()
        self._mapping_int: Dict[int, str] = {}
        self._mapping_str: Dict[str, int] = {}

    @classmethod
    def _first_primes(cls, n=1000):
        sieve = [True] * (n + 1)
        sieve[0:2] = [False, False]

        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                # Élimine les multiples de i
                sieve[i * i : n + 1 : i] = [False] * len(range(i * i, n + 1, i))

        return [i for i, prime in enumerate(sieve) if prime and i >= 13]

    def _unique(self) -> int:
        i = 0
        c = self._candidates[i]
        while c in self._unique_ or c in self._mapping_int:
            i += 1
            assert i < len(
                self._candidates
            ), f"Two many unique dimensions to generate, requested: {len(self._unique_)}"
            c = self._candidates[i]
        self._unique_.add(c)
        return c

    def from_tensor(self, x, static_shapes=False) -> "FakeTensor":  # noqa: F821
        """
        Returns a fake tensor.
        ``pytorch`` returns the same name for the same dimension.
        """
        fake = self.fake_mode.from_tensor(x, static_shapes=static_shapes)
        for i, s in zip(x.shape, fake.shape):
            assert i not in self._mapping_int or self._mapping_int[i] == s, (
                f"Inconsistency between {x.shape} and {fake.shape}, "
                f"mapping has {self._mapping_int[i]} and s={s}"
            )
            self._mapping_int[i] = s
        return fake

    def fake_reshape(
        self,
        true_tensor: "torch.Tensor",  # noqa: F821
        sh: Dict[int, Any],  # noqa: F821
        fake_tensor: Optional["FakeTensor"] = None,  # noqa: F821
    ) -> "FakeTensor":  # noqa: F821
        """
        Changes the shape of a true tensor to make it dynamic.

        :param true_tensor: true tensor
        :param sh: dynamic shape
        :param fake_tensor: fake tensor, if None, make a fake one
        :return: fake tensor
        """
        import torch

        # deal with 0/1
        for i in sh:
            if true_tensor.shape[i] <= 1:
                expanded_shape = list(true_tensor.shape)
                expanded_shape[i] = self._unique()
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
                d = self._unique()
                mapping[d] = s
                new_shape[i] = d
        true_tensor = torch.empty(
            tuple(new_shape), dtype=true_tensor.dtype, device=true_tensor.device
        )

        # now switch to FakeTensor
        fake_tensor = self.from_tensor(true_tensor, static_shapes=False)
        new_shape = list(true_tensor.shape)
        for i in sh:
            new_shape[i] = fake_tensor.shape[i]

        reduced_tensor = self.from_tensor(true_tensor, static_shapes=True).sum(
            axis=tuple(sorted(sh)), keepdim=True
        )
        return reduced_tensor.expand(*new_shape)

    def make_fake(self, x: Any) -> Optional["FakeTensor"]:  # noqa: F821
        """See :func:`onnx_diagnostic.helpers.fake_tensor_helper.make_fake`."""
        if x is None:
            return None
        if isinstance(x, (list, tuple)):
            return x.__class__([self.make_fake(i) for i in x])
        if isinstance(x, dict):
            return {k: self.make_fake(v) for k, v in x.items()}
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
                layer.keys = self.make_fake(layer.keys)
                layer.values = self.make_fake(layer.values)
            return x
        if x.__class__.__name__ == "EncoderDecoderCache":
            self.make_fake(x.self_attention_cache)
            self.make_fake(x.cross_attention_cache)
            return x
        if hasattr(x, "shape"):
            return self.from_tensor(x, static_shapes=False)
        from . import string_type

        raise TypeError(
            f"Unexpected type {type(x)} for x, content is {string_type(x, with_shape=True)}"
        )

    def make_fake_with_dynamic_dimensions(self, x: Any, dynamic_shapes: Any) -> Any:
        """
        See
        :func:`onnx_diagnostic.helpers.shape_helper.make_fake_with_dynamic_dimensions`.
        """
        if x is None:
            return None, None
        if isinstance(x, (list, tuple)):
            return x.__class__(
                [
                    self.make_fake_with_dynamic_dimensions(i, dynamic_shapes=ds)
                    for i, ds in zip(x, dynamic_shapes)
                ]
            )
        if isinstance(x, dict):
            return {
                k: self.make_fake_with_dynamic_dimensions(v, dynamic_shapes=dynamic_shapes[k])
                for k, v in x.items()
            }
        if x.__class__.__name__ in {"DynamicCache", "StaticCache", "HybridCache"}:
            assert hasattr(x, "layers"), (
                f"Une more recent version of transformers (>=4.55), "
                f"'layers' not found in class {type(x)}"
            )
            assert isinstance(dynamic_shapes, list) and (
                not dynamic_shapes or not isinstance(dynamic_shapes[0], list)
            ), f"Unexpected dynamic_shapes={dynamic_shapes} for a DynamicCache"
            for il, layer in enumerate(x.layers):
                assert hasattr(layer, "keys") and hasattr(layer, "values"), (
                    f"Une more recent version of transformers (>=4.55), 'layers' "
                    f"not found in class {type(layer)} ({dir(layer)})"
                )
                layer.keys = self.make_fake_with_dynamic_dimensions(
                    layer.keys, dynamic_shapes=dynamic_shapes[il * 2]
                )
                layer.values = self.make_fake_with_dynamic_dimensions(
                    layer.values, dynamic_shapes=dynamic_shapes[il * 2 + 1]
                )
            return x
        if x.__class__.__name__ == "EncoderDecoderCache":
            self.make_fake_with_dynamic_dimensions(
                x.self_attention_cache, dynamic_shapes=dynamic_shapes[0]
            )
            self.make_fake_with_dynamic_dimensions(
                x.cross_attention_cache, dynamic_shapes=dynamic_shapes[1]
            )
            return x
        if hasattr(x, "shape"):
            assert dynamic_shapes is None or isinstance(dynamic_shapes, dict), (
                f"dynamic_shapes must be a dictionary at this stage but "
                f"dynamic_shapes={dynamic_shapes}"
            )
            # We need to overwrite the values.
            new_shape = []
            for idim, dim in enumerate(x.shape):
                if dynamic_shapes is not None and idim in dynamic_shapes:
                    s = dynamic_shapes[idim]
                    assert isinstance(s, str), (
                        f"Unexpected type {type(s)} in dynamic_shapes={dynamic_shapes} "
                        f"at index {idim}"
                    )
                    if s in self._mapping_str:
                        dim = self._mapping_str[s]
                    else:
                        i = self._unique()
                        self._mapping_str[s] = i
                        dim = i
                assert isinstance(dim, int), (
                    f"Unexpected type {type(dim)}, dynamic_shapes={dynamic_shapes} "
                    f"at index {idim}, dim={dim}"
                )
                new_shape.append(dim)
            if tuple(new_shape) != x.shape:
                import torch

                x = torch.empty(tuple(new_shape), dtype=x.dtype, device=x.device)

            t = self.fake_reshape(x, dynamic_shapes)  # type: ignore[arg-type]
            assert t.device == x.device, f"device mismatch {x.device} -> {t.device}"
            assert t.dtype == x.dtype, f"dtype mismatch {x.dtype} -> {t.dtype}"
            return t
        from ..helpers import string_type

        raise TypeError(
            f"Unexpected type {type(x)} for x, content is {string_type(x, with_shape=True)}"
        )


def make_fake(
    x: Any, context: Optional[FakeTensorContext] = None
) -> Tuple[Optional["FakeTensor"], Optional[FakeTensorContext]]:  # noqa: F821
    """
    Replaces all tensors by fake tensors.
    This modification happens inplace for caches.
    This function is only implemented for cache with
    ``transformers>=4.55``.

    .. runpython::
        :showcode:

        import pprint
        import torch
        from onnx_diagnostic.helpers.cache_helper import make_dynamic_cache
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
        pprint.pprint(inputs)
    """
    if x is None:
        return None, None
    if context is None:
        context = FakeTensorContext()
    return context.make_fake(x), context
