from typing import Any, Callable, List, Optional, Tuple
import packaging.version as pv
import torch
import transformers
import transformers.cache_utils

try:
    from transformers.models.mamba.modeling_mamba import MambaCache
except ImportError:
    from transformers.cache_utils import MambaCache


class CacheKeyValue:
    def __init__(self, cache: "Cache"):  # noqa: F821
        if hasattr(cache, "layers"):
            self.key_cache = [layer.keys for layer in cache.layers if layer.keys is not None]
            self.value_cache = [
                layer.values for layer in cache.layers if layer.values is not None
            ]
        else:
            self.key_cache = cache.key_cache
            self.value_cache = cache.value_cache


def flatten_unflatten_for_dynamic_shapes(
    obj: Any,
    use_dict: bool = False,
    change_function: Optional[Callable[[torch.Tensor], Any]] = None,
) -> Any:
    """
    Returns the object in a different structure similar to what
    the definition of the dynamic shapes should use.

    :param obj: object from a custom class
    :param use_dict: closer to the original result but
        :func:`torch.export.export` only considers the values,
        the context gives the dictionary keys but it is not expressed
        in the dynamic shapes, these specifications seems to be different
        for the strict and non strict mode. It also preserves tuple.
    :param change_function: to modifies the tensor in the structure itself,
        like replace them by a shape
    :return: the serialized object
    """
    if isinstance(obj, torch.Tensor):
        return change_function(obj) if change_function else obj
    flat, spec = torch.utils._pytree.tree_flatten(obj)
    start = 0
    end = 0
    subtrees = []
    for subspec in spec.children_specs:
        end += subspec.num_leaves
        value = subspec.unflatten(flat[start:end])
        value = flatten_unflatten_for_dynamic_shapes(
            value, use_dict=use_dict, change_function=change_function
        )
        subtrees.append(value)
        start = end
    if use_dict:
        if spec.type is dict:
            # This a dictionary.
            return dict(zip(spec.context, subtrees))
        if spec.type is tuple:
            return tuple(subtrees)
        if spec.type is list:
            return list(subtrees)
        if spec.context:
            # This is a custom class with attributes.
            # It is returned as a list.
            return list(subtrees)
        raise ValueError(
            f"Unable to interpret spec type {spec.type} "
            f"(type is {type(spec.type)}, context is {spec.context})."
        )
    # This is a list.
    return subtrees


def is_cache_dynamic_registered(fast: bool = False) -> bool:
    """
    Tells class :class:`transformers.cache_utils.DynamicCache` can be
    serialized and deserialized. Only then, :func:`torch.export.export`
    can export a model.

    :param fast: if True, do not check the serialization is ok as well
    :return: result
    """
    if fast:
        return transformers.cache_utils.DynamicCache in torch.utils._pytree.SUPPORTED_NODES
    bsize, nheads, slen, dim = 2, 4, 3, 7
    cache = make_dynamic_cache(
        [
            (
                torch.randn(bsize, nheads, slen, dim),
                torch.randn(bsize, nheads, slen, dim),
            )
            for i in range(2)
        ]
    )
    values, spec = torch.utils._pytree.tree_flatten(cache)
    cache2 = torch.utils._pytree.tree_unflatten(values, spec)
    return len(cache2.key_cache) == len(cache.value_cache)


if pv.Version(transformers.__version__) > pv.Version("4.49.99999"):

    def make_dynamic_cache(
        key_value_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> transformers.cache_utils.DynamicCache:
        """
        Creates an instance of :class:`transformers.cache_utils.DynamicCache`.
        This version is valid for ``transformers >= 4.50``.

        :param key_value_pairs: list of pairs of (key, values)
        :return: :class:`transformers.cache_utils.DynamicCache`

        Example:

        .. runpython::
            :showcode:

            import torch
            from onnx_diagnostic.helpers import string_type
            from onnx_diagnostic.helpers.cache_helper import make_dynamic_cache

            n_layers = 2
            bsize, nheads, slen, dim = 2, 4, 3, 7

            past_key_values = make_dynamic_cache(
                [
                    (
                        torch.randn(bsize, nheads, slen, dim),
                        torch.randn(bsize, nheads, slen, dim),
                    )
                    for i in range(n_layers)
                ]
            )
            print(string_type(past_key_values, with_shape=True))
        """
        return transformers.cache_utils.DynamicCache(key_value_pairs)

else:

    def make_dynamic_cache(
        key_value_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> transformers.cache_utils.DynamicCache:
        """
        Creates an instance of :class:`transformers.cache_utils.DynamicCache`.
        This version is valid for ``transformers < 4.50``.

        :param key_value_pairs: list of pairs of (key, values)
        :return: :class:`transformers.cache_utils.DynamicCache`

        Example:

        .. runpython::
            :showcode:

            import torch
            from onnx_diagnostic.helpers import string_type
            from onnx_diagnostic.helpers.cache_helper import make_dynamic_cache

            n_layers = 2
            bsize, nheads, slen, dim = 2, 4, 3, 7

            past_key_values = make_dynamic_cache(
                [
                    (
                        torch.randn(bsize, nheads, slen, dim),
                        torch.randn(bsize, nheads, slen, dim),
                    )
                    for i in range(n_layers)
                ]
            )
            print(string_type(past_key_values, with_shape=True))
        """
        cache = transformers.cache_utils.DynamicCache(len(key_value_pairs))  # type: ignore
        for i, (key, value) in enumerate(key_value_pairs):
            cache.update(key, value, i)
        return cache


def make_static_cache(
    key_value_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    max_cache_len: Optional[int] = None,
) -> transformers.cache_utils.DynamicCache:
    """
    Creates an instance of :class:`transformers.cache_utils.StaticCache`.
    :param key_value_pairs: list of pairs of (key, values)
    :param max_cache_len: max_cache_length or something inferred from the vector
    :return: :class:`transformers.cache_utils.StaticCache`

    Example:

    .. runpython::
        :showcode:

        import torch
        from onnx_diagnostic.helpers import string_type
        from onnx_diagnostic.helpers.cache_helper import make_static_cache

        n_layers = 2
        bsize, nheads, slen, dim = 2, 4, 3, 7

        past_key_values = make_static_cache(
            [
                (
                    torch.randn(bsize, nheads, slen, dim),
                    torch.randn(bsize, nheads, slen, dim),
                )
                for i in range(n_layers)
            ],
            max_cache_len=10,
        )
        print(string_type(past_key_values, with_shape=True))
    """

    class _config:
        def __init__(self):
            self.head_dim = key_value_pairs[0][0].shape[-1]
            self.num_attention_heads = key_value_pairs[0][0].shape[1]
            self.num_hidden_layers = len(key_value_pairs)

    assert max_cache_len is not None, (
        f"max_cache_len={max_cache_len} cannot be setup "
        f"automatically yet from shape {key_value_pairs[0][0].shape}"
    )
    torch._check(
        max_cache_len >= key_value_pairs[0][0].shape[2],
        (
            f"max_cache_len={max_cache_len} cannot be smaller "
            f"shape[2]={key_value_pairs[0][0].shape[2]} in shape "
            f"{key_value_pairs[0][0].shape}"
        ),
    )
    cache = transformers.cache_utils.StaticCache(
        config=_config(),
        max_batch_size=key_value_pairs[0][0].shape[0],
        device=key_value_pairs[0][0].device,
        dtype=key_value_pairs[0][0].dtype,
        max_cache_len=max_cache_len,
    )
    ca = CacheKeyValue(cache)
    for i in range(len(key_value_pairs)):
        assert (
            key_value_pairs[i][0].shape == key_value_pairs[i][1].shape
        ), f"Shape mismatch {key_value_pairs[i][0].shape} != {key_value_pairs[i][1].shape}"
        d = key_value_pairs[i][1].shape[2]
        ca.key_cache[i][:, :, :d, :] = key_value_pairs[i][0]
        ca.value_cache[i][:, :, :d, :] = key_value_pairs[i][1]
    return cache


def make_encoder_decoder_cache(
    self_attention_cache: transformers.cache_utils.DynamicCache,
    cross_attention_cache: transformers.cache_utils.DynamicCache,
) -> transformers.cache_utils.EncoderDecoderCache:
    """Creates an EncoderDecoderCache."""
    return transformers.cache_utils.EncoderDecoderCache(
        self_attention_cache=self_attention_cache, cross_attention_cache=cross_attention_cache
    )


def make_mamba_cache(key_value_pairs: List[Tuple[torch.Tensor, torch.Tensor]]) -> MambaCache:
    "Creates a ``MambaCache``."
    dtype = key_value_pairs[0][0].dtype

    class _config:
        def __init__(self):
            self.intermediate_size = key_value_pairs[0][0].shape[1]
            self.conv_kernel = key_value_pairs[0][0].shape[-1]
            self.state_size = key_value_pairs[0][1].shape[-1]
            self.num_hidden_layers = len(key_value_pairs)
            self.dtype = dtype

    cache = MambaCache(
        _config(),
        max_batch_size=key_value_pairs[0][0].shape[0],
        device=key_value_pairs[0][0].device,
        dtype=dtype,
    )
    for i in range(len(key_value_pairs)):
        assert cache.conv_states[i].dtype == dtype, (
            f"Type mismatch for cache.conv_states[{i}].dtype="
            f"{cache.conv_states[i].dtype} != {dtype}"
        )
        assert cache.ssm_states[i].dtype == dtype, (
            f"Type mismatch for cache.ssm_states[{i}].dtype="
            f"{cache.ssm_states[i].dtype} != {dtype}"
        )
        assert cache.conv_states[i].shape == key_value_pairs[i][0].shape, (
            f"Shape mismatch, expected {cache.conv_states[i].shape}, "
            f"got {key_value_pairs[i][0].shape}"
        )
        cache.conv_states[i][:, :, :] = key_value_pairs[i][0]
        assert cache.ssm_states[i].shape == key_value_pairs[i][1].shape, (
            f"Shape mismatch, expected {cache.ssm_states[i].shape}, "
            f"got {key_value_pairs[i][1].shape}"
        )
        cache.ssm_states[i][:, :, :] = key_value_pairs[i][1]
    return cache


def make_sliding_window_cache(
    key_value_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
) -> transformers.cache_utils.SlidingWindowCache:
    "Creates a :class:`transformers.cache_utils.SlidingWindowCache`."

    class _config:
        def __init__(self):
            self.head_dim = key_value_pairs[0][0].shape[-1]
            self.num_attention_heads = key_value_pairs[0][0].shape[1]
            self.num_hidden_layers = len(key_value_pairs)
            self.sliding_window = key_value_pairs[0][0].shape[2]

    cache = transformers.cache_utils.SlidingWindowCache(
        config=_config(),
        max_batch_size=key_value_pairs[0][0].shape[0],
        max_cache_len=key_value_pairs[0][0].shape[2],  # same as sliding_window
        device=key_value_pairs[0][0].device,
        dtype=key_value_pairs[0][0].dtype,
    )
    ca = CacheKeyValue(cache)
    for i in range(len(key_value_pairs)):
        assert ca.key_cache[i].shape == key_value_pairs[i][0].shape, (
            f"Shape mismatch, expected {cache.key_cache[i].shape}, "
            f"got {key_value_pairs[i][0].shape}"
        )
        ca.key_cache[i][:, :, :, :] = key_value_pairs[i][0]
        assert ca.value_cache[i].shape == key_value_pairs[i][1].shape, (
            f"Shape mismatch, expected {cache.value_cache[i].shape}, "
            f"got {key_value_pairs[i][1].shape}"
        )
        ca.value_cache[i][:, :, :, :] = key_value_pairs[i][1]
    return cache


def make_hybrid_cache(
    key_value_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    max_cache_len: Optional[int] = None,
    max_batch_size: Optional[int] = None,
) -> transformers.cache_utils.HybridCache:
    """
    Creates an instance of :class:`transformers.cache_utils.HybridCache`.
    This version is valid for ``transformers < 4.50``.

    :param key_value_pairs: list of pairs of (key, values)
    :return: :class:`transformers.cache_utils.HybridCache`

    Example:

    .. runpython::
        :showcode:

        import torch
        from onnx_diagnostic.helpers import string_type
        from onnx_diagnostic.helpers.cache_helper import make_hybrid_cache

        n_layers = 2
        bsize, nheads, slen, dim = 2, 4, 3, 7

        past_key_values = make_hybrid_cache(
            [
                (
                    torch.randn(bsize, nheads, slen, dim),
                    torch.randn(bsize, nheads, slen, dim),
                )
                for i in range(n_layers)
            ]
        )
        print(string_type(past_key_values, with_shape=True))
    """
    if key_value_pairs:
        assert (
            not max_batch_size and not max_cache_len
        ), "key_value_pairs is not empty, do not specify max_cache_len and max_batch_size"
        max_batch_size = key_value_pairs[0][0].shape[0]
        max_cache_len = key_value_pairs[0][0].shape[2]
    else:
        assert (
            max_batch_size and max_cache_len
        ), "key_value_pairs is empty, max_batch_size and max_cache_len are required"
    _ = max_cache_len

    class _config:
        max_cache_len = _
        batch_size = max_batch_size
        num_heads = key_value_pairs[0][0].shape[1] if key_value_pairs else None
        head_dim = key_value_pairs[0][0].shape[-1] if key_value_pairs else None
        num_attention_heads = key_value_pairs[0][1].shape[1] if key_value_pairs else None
        num_hidden_layers = len(key_value_pairs)

    cache = transformers.cache_utils.HybridCache(
        config=_config(), max_cache_len=max_cache_len, max_batch_size=max_batch_size
    )
    for i, (key, value) in enumerate(key_value_pairs):
        cache.update(key, value, i)
    return cache
