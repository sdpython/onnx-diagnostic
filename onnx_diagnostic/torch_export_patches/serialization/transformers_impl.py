import itertools
from typing import Any, Callable, List, Set, Tuple
import torch
from transformers.cache_utils import Cache, DynamicCache, EncoderDecoderCache, StaticCache

try:
    from transformers.cache_utils import SlidingWindowCache
except ImportError:
    SlidingWindowCache = None


try:
    from transformers.cache_utils import HybridCache
except ImportError:
    HybridCache = None

try:
    from transformers.models.mamba.modeling_mamba import MambaCache
except ImportError:
    from transformers.cache_utils import MambaCache
from transformers.modeling_outputs import BaseModelOutput
from ...helpers.cache_helper import make_dynamic_cache, make_static_cache, CacheKeyValue
from . import make_serialization_function_for_dataclass

SUPPORTED_DATACLASSES: Set[type] = set()
WRONG_REGISTRATIONS = {
    DynamicCache: "4.50",
    BaseModelOutput: None,
}


def _flatten_key_value_cache(cache: Cache) -> Tuple[List[Any], torch.utils._pytree.Context]:
    ca = CacheKeyValue(cache)
    flat = list(itertools.chain.from_iterable(zip(ca.key_cache, ca.value_cache)))
    keys = list(
        itertools.chain.from_iterable(
            (f"key_{i}", f"value_{i}") for i in range(len(ca.key_cache))
        )
    )
    return flat, keys


def _flatten_with_keys_cache(
    cache: Cache,
) -> Tuple[List[Tuple[torch.utils._pytree.KeyEntry, Any]], torch.utils._pytree.Context]:
    values, context = _flatten_key_value_cache(cache)
    return [(torch.utils._pytree.MappingKey(k), v) for k, v in zip(context, values)], context


def _unflatten_cache(
    make_cache: Callable,
    values: List[Any],
    context: torch.utils._pytree.Context,
    output_type=None,
) -> DynamicCache:
    """Restores a :class:`transformers.cache_utils.DynamicCache` from python objects."""
    res = make_cache(list(zip(values[::2], values[1::2])))
    assert output_type is None or isinstance(
        res, output_type
    ), f"Type mismatch between {output_type} (expected) and {type(res)}"
    return res


##############
# DynamicCache
##############


def flatten_dynamic_cache(
    dynamic_cache: DynamicCache,
) -> Tuple[List[Any], torch.utils._pytree.Context]:
    """Serializes a :class:`transformers.cache_utils.DynamicCache` with python objects."""
    assert (
        not hasattr(dynamic_cache, "layers")
        or not dynamic_cache.layers
        or all(lay.__class__.__name__ == "DynamicLayer" for lay in dynamic_cache.layers)
    ), (
        f"The serialization does not work yet on other layers "
        f"than DynamicLayer, but layers={[lay.__class__ for lay in dynamic_cache.layers]}"
    )
    return _flatten_key_value_cache(dynamic_cache)


def flatten_with_keys_dynamic_cache(
    dynamic_cache: DynamicCache,
) -> Tuple[List[Tuple[torch.utils._pytree.KeyEntry, Any]], torch.utils._pytree.Context]:
    """Serializes a :class:`transformers.cache_utils.DynamicCache` with python objects."""
    assert (
        not hasattr(dynamic_cache, "layers")
        or not dynamic_cache.layers
        or all(lay.__class__.__name__ == "DynamicLayer" for lay in dynamic_cache.layers)
    ), (
        f"The serialization does not work yet on other layers "
        f"than DynamicLayer, but layers={[lay.__class__ for lay in dynamic_cache.layers]}"
    )
    return _flatten_with_keys_cache(dynamic_cache)


def unflatten_dynamic_cache(
    values: List[Any], context: torch.utils._pytree.Context, output_type=None
) -> DynamicCache:
    """Restores a :class:`transformers.cache_utils.DynamicCache` from python objects."""
    return _unflatten_cache(make_dynamic_cache, values, context, output_type=output_type)


#############
# HybridCache
#############

if HybridCache:

    def flatten_hybrid_cache(
        cache: HybridCache,
    ) -> Tuple[List[Any], torch.utils._pytree.Context]:
        """Serializes a :class:`transformers.cache_utils.HybridCache` with python objects."""
        return _flatten_key_value_cache(cache)

    def flatten_with_keys_hybrid_cache(
        cache: HybridCache,
    ) -> Tuple[List[Tuple[torch.utils._pytree.KeyEntry, Any]], torch.utils._pytree.Context]:
        """Serializes a :class:`transformers.cache_utils.HybridCache` with python objects."""
        return _flatten_with_keys_cache(cache)

    def unflatten_hybrid_cache(
        values: List[Any], context: torch.utils._pytree.Context, output_type=None
    ) -> HybridCache:
        """Restores a :class:`transformers.cache_utils.HybridCache` from python objects."""
        from ...helpers.cache_helper import make_hybrid_cache

        return _unflatten_cache(make_hybrid_cache, values, context, output_type=output_type)


#############
# StaticCache
#############


def flatten_static_cache(
    cache: StaticCache,
) -> Tuple[List[Any], torch.utils._pytree.Context]:
    """Serializes a :class:`transformers.cache_utils.StaticCache` with python objects."""
    ca = CacheKeyValue(cache)
    assert not ca.key_cache or cache.max_cache_len == ca.key_cache[0].shape[2], (
        f"Serialization doet not work when "
        f"cache.max_cache_len={cache.max_cache_len} != "
        f"cache.key_cache[0].shape[2]={ca.key_cache[0].shape[2]}"
    )
    return _flatten_key_value_cache(cache)


def flatten_with_keys_static_cache(
    cache: StaticCache,
) -> Tuple[List[Tuple[torch.utils._pytree.KeyEntry, Any]], torch.utils._pytree.Context]:
    """Serializes a :class:`transformers.cache_utils.StaticCache` with python objects."""
    return _flatten_with_keys_cache(cache)


def unflatten_static_cache(
    values: List[Any], context: torch.utils._pytree.Context, output_type=None
) -> StaticCache:
    """Restores a :class:`transformers.cache_utils.StaticCache` from python objects."""
    return _unflatten_cache(
        lambda *args: make_static_cache(*args, max_cache_len=values[0].shape[2]),
        values,
        context,
        output_type=output_type,
    )


####################
# SlidingWindowCache
####################


if SlidingWindowCache:

    def flatten_sliding_window_cache(
        cache: SlidingWindowCache,
    ) -> Tuple[List[Any], torch.utils._pytree.Context]:
        """
        Serializes a :class:`transformers.cache_utils.SlidingWindowCache`
        with python objects.
        """
        return _flatten_key_value_cache(cache)

    def flatten_with_keys_sliding_window_cache(
        cache: SlidingWindowCache,
    ) -> Tuple[List[Tuple[torch.utils._pytree.KeyEntry, Any]], torch.utils._pytree.Context]:
        """
        Serializes a :class:`transformers.cache_utils.SlidingWindowCache`
        with python objects.
        """
        return _flatten_with_keys_cache(cache)

    def unflatten_sliding_window_cache(
        values: List[Any], context: torch.utils._pytree.Context, output_type=None
    ) -> SlidingWindowCache:
        """
        Restores a :class:`transformers.cache_utils.SlidingWindowCache`
        from python objects.
        """
        from ...helpers.cache_helper import make_sliding_window_cache

        return _unflatten_cache(
            make_sliding_window_cache, values, context, output_type=output_type
        )


#####################
# EncoderDecoderCache
#####################


def flatten_encoder_decoder_cache(
    ec_cache: EncoderDecoderCache,
) -> Tuple[List[Any], torch.utils._pytree.Context]:
    """
    Serializes a :class:`transformers.cache_utils.EncoderDecoderCache`
    with python objects.
    """
    dictionary = {
        "self_attention_cache": ec_cache.self_attention_cache,
        "cross_attention_cache": ec_cache.cross_attention_cache,
    }
    return torch.utils._pytree._dict_flatten(dictionary)


def flatten_with_keys_encoder_decoder_cache(ec_cache: EncoderDecoderCache) -> Tuple[
    List[Tuple[torch.utils._pytree.KeyEntry, Any]],
    torch.utils._pytree.Context,
]:
    """
    Serializes a :class:`transformers.cache_utils.EncoderDecoderCache`
    with python objects.
    """
    dictionary = {
        "self_attention_cache": ec_cache.self_attention_cache,
        "cross_attention_cache": ec_cache.cross_attention_cache,
    }
    return torch.utils._pytree._dict_flatten_with_keys(dictionary)


def unflatten_encoder_decoder_cache(
    values: List[Any], context: torch.utils._pytree.Context, output_type=None
) -> EncoderDecoderCache:
    """Restores a :class:`transformers.cache_utils.EncoderDecoderCache` from python objects."""
    dictionary = torch.utils._pytree._dict_unflatten(values, context)
    return EncoderDecoderCache(
        dictionary["self_attention_cache"], dictionary["cross_attention_cache"]
    )


############
# MambaCache
############


def flatten_mamba_cache(
    mamba_cache: MambaCache,
) -> Tuple[List[Any], torch.utils._pytree.Context]:
    """Serializes a :class:`transformers.cache_utils.MambaCache` with python objects."""
    assert isinstance(mamba_cache.conv_states, list) and isinstance(
        mamba_cache.ssm_states, list
    ), (
        f"Unexpected types for conv_states and ssm_states {type(mamba_cache.conv_states)}, "
        f"{type(mamba_cache.ssm_states)}"
    )
    flat = [
        ("conv_states", mamba_cache.conv_states),
        ("ssm_states", mamba_cache.ssm_states),
    ]
    return [f[1] for f in flat], [f[0] for f in flat]


def unflatten_mamba_cache(
    values: List[Any], context: torch.utils._pytree.Context, output_type=None
) -> MambaCache:
    """Restores a :class:`transformers.cache_utils.MambaCache` from python objects."""
    conv_states, ssm_states = values

    class _config:
        def __init__(self):
            if isinstance(conv_states, list):
                self.intermediate_size = conv_states[0].shape[1]
                self.state_size = ssm_states[0].shape[2]
                self.conv_kernel = conv_states[0].shape[2]
                self.num_hidden_layers = len(conv_states)
            else:
                self.intermediate_size = conv_states.shape[2]
                self.state_size = ssm_states.shape[3]
                self.conv_kernel = conv_states.shape[3]
                self.num_hidden_layers = conv_states.shape[0]

    cache = MambaCache(
        _config(),
        max_batch_size=1,
        dtype=values[-1][0].dtype,
        device="cpu" if values[-1][0].get_device() < 0 else "cuda",
    )
    values = dict(zip(context, values))
    for k, v in values.items():
        setattr(cache, k, v)
    return cache


def flatten_with_keys_mamba_cache(cache: MambaCache) -> Tuple[
    List[Tuple[torch.utils._pytree.KeyEntry, Any]],
    torch.utils._pytree.Context,
]:
    """Serializes a :class:`transformers.cache_utils.MambaCache` with python objects."""
    values, context = flatten_mamba_cache(cache)
    return [(torch.utils._pytree.MappingKey(k), v) for k, v in zip(context, values)], context


#############
# dataclasses
#############


(
    flatten_base_model_output,
    flatten_with_keys_base_model_output,
    unflatten_base_model_output,
) = make_serialization_function_for_dataclass(BaseModelOutput, SUPPORTED_DATACLASSES)
