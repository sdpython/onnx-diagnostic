from typing import Any, List, Tuple
import packaging.version as pv
import torch
import transformers
import transformers.cache_utils


def flatten_unflatten_for_dynamic_shapes(obj: Any, use_dict: bool = False) -> Any:
    """
    Returns the object in a different structure similar to what
    the definition of the dynamic shapes should use.

    :param obj: object from a custom class
    :param use_dict: closer to the original result but
        :func:`torch.export.export` only considers the values,
        the context gives the dictionary keys but it is not expressed
        in the dynamic shapes, these specifications seems to be different
        for the strict and non strict mode.
    :return: the serialized object
    """
    if isinstance(obj, torch.Tensor):
        return obj
    flat, spec = torch.utils._pytree.tree_flatten(obj)
    start = 0
    end = 0
    subtrees = []
    for subspec in spec.children_specs:
        end += subspec.num_leaves
        value = subspec.unflatten(flat[start:end])
        value = flatten_unflatten_for_dynamic_shapes(value, use_dict=use_dict)
        subtrees.append(value)
        start = end
    if use_dict and (spec.type is dict or spec.context):
        # This a dictionary.
        return dict(zip(spec.context, subtrees))
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


def make_encoder_decoder_cache(
    self_attention_cache: transformers.cache_utils.DynamicCache,
    cross_attention_cache: transformers.cache_utils.DynamicCache,
) -> transformers.cache_utils.EncoderDecoderCache:
    """Creates an EncoderDecoderCache."""
    return transformers.cache_utils.EncoderDecoderCache(
        self_attention_cache=self_attention_cache, cross_attention_cache=cross_attention_cache
    )


def make_mamba_cache(
    key_value_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
) -> transformers.cache_utils.MambaCache:
    "Creates a :class:`transformers.cache_utils.MambaCache`."
    dtype = key_value_pairs[0][0].dtype

    class _config:
        def __init__(self):
            self.intermediate_size = key_value_pairs[0][0].shape[1]
            self.conv_kernel = key_value_pairs[0][0].shape[-1]
            self.state_size = key_value_pairs[0][1].shape[-1]
            self.num_hidden_layers = len(key_value_pairs)
            self.dtype = dtype

    cache = transformers.cache_utils.MambaCache(
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
) -> transformers.cache_utils.MambaCache:
    "Creates a :class:`transformers.cache_utils.SlidingWindowCache`."

    class _config:
        def __init__(self):
            self.head_dim = key_value_pairs[0][0].shape[-1]
            self.num_attention_heads = key_value_pairs[0][0].shape[1]
            self.num_hidden_layers = len(key_value_pairs)
            self.sliding_window = key_value_pairs[0][0].shape[2]

    cache = transformers.cache_utils.SlidingWindowCache(
        _config(),
        max_batch_size=key_value_pairs[0][0].shape[0],
        max_cache_len=key_value_pairs[0][0].shape[2],  # same as sliding_window
        device=key_value_pairs[0][0].device,
        dtype=key_value_pairs[0][0].dtype,
    )
    for i in range(len(key_value_pairs)):
        assert cache.key_cache[i].shape == key_value_pairs[i][0].shape, (
            f"Shape mismatch, expected {cache.key_cache[i].shape}, "
            f"got {key_value_pairs[i][0].shape}"
        )
        cache.key_cache[i][:, :, :, :] = key_value_pairs[i][0]
        assert cache.value_cache[i].shape == key_value_pairs[i][1].shape, (
            f"Shape mismatch, expected {cache.value_cache[i].shape}, "
            f"got {key_value_pairs[i][1].shape}"
        )
        cache.value_cache[i][:, :, :, :] = key_value_pairs[i][1]
    return cache
