import re
from typing import Any, Callable, List, Tuple
import torch
import transformers
from transformers.cache_utils import (
    DynamicCache,
    MambaCache,
    EncoderDecoderCache,
    SlidingWindowCache,
    StaticCache,
)
from transformers.modeling_outputs import BaseModelOutput

try:
    from diffusers.models.autoencoders.vae import DecoderOutput, EncoderOutput
    from diffusers.models.unets.unet_1d import UNet1DOutput
    from diffusers.models.unets.unet_2d import UNet2DOutput
    from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput
    from diffusers.models.unets.unet_3d_condition import UNet3DConditionOutput
except ImportError as e:
    try:
        import diffusers
    except ImportError:
        diffusers = None
        DecoderOutput, EncoderOutput = None, None
        UNet1DOutput, UNet2DOutput = None, None
        UNet2DConditionOutput, UNet3DConditionOutput = None, None
    if diffusers:
        raise e

from ..helpers.cache_helper import make_static_cache


SUPPORTED_DATACLASSES = set()

############
# MambaCache
############


def flatten_mamba_cache(
    mamba_cache: MambaCache,
) -> Tuple[List[Any], torch.utils._pytree.Context]:
    """Serializes a :class:`transformers.cache_utils.MambaCache` with python objects."""
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


##############
# DynamicCache
##############


def flatten_dynamic_cache(
    dynamic_cache: DynamicCache,
) -> Tuple[List[Any], torch.utils._pytree.Context]:
    """Serializes a :class:`transformers.cache_utils.DynamicCache` with python objects."""
    if hasattr(transformers.cache_utils, "_flatten_dynamic_cache"):
        return transformers.cache_utils._flatten_dynamic_cache(dynamic_cache)
    flat = [("key_cache", dynamic_cache.key_cache), ("value_cache", dynamic_cache.value_cache)]
    return [f[1] for f in flat], [f[0] for f in flat]


def flatten_with_keys_dynamic_cache(
    dynamic_cache: DynamicCache,
) -> Tuple[List[Tuple[torch.utils._pytree.KeyEntry, Any]], torch.utils._pytree.Context]:
    """Serializes a :class:`transformers.cache_utils.DynamicCache` with python objects."""
    if hasattr(transformers.cache_utils, "_flatten_with_keys_dynamic_cache"):
        return transformers.cache_utils._flatten_with_keys_dynamic_cache(dynamic_cache)
    values, context = flatten_dynamic_cache(dynamic_cache)
    return [(torch.utils._pytree.MappingKey(k), v) for k, v in zip(context, values)], context


def unflatten_dynamic_cache(
    values: List[Any], context: torch.utils._pytree.Context, output_type=None
) -> DynamicCache:
    """Restores a :class:`transformers.cache_utils.DynamicCache` from python objects."""
    if hasattr(transformers.cache_utils, "_unflatten_dynamic_cache"):
        assert output_type is None, f"output_type={output_type} not supported"
        return transformers.cache_utils._unflatten_dynamic_cache(values, context)

    cache = transformers.cache_utils.DynamicCache()
    values = dict(zip(context, values))
    for k, v in values.items():
        setattr(cache, k, v)
    return cache


#############
# StaticCache
#############


def flatten_static_cache(
    cache: StaticCache,
) -> Tuple[List[Any], torch.utils._pytree.Context]:
    """Serializes a :class:`transformers.cache_utils.StaticCache` with python objects."""
    flat = [("key_cache", cache.key_cache), ("value_cache", cache.value_cache)]
    return [f[1] for f in flat], [f[0] for f in flat]


def flatten_with_keys_static_cache(
    cache: StaticCache,
) -> Tuple[List[Tuple[torch.utils._pytree.KeyEntry, Any]], torch.utils._pytree.Context]:
    """Serializes a :class:`transformers.cache_utils.StaticCache` with python objects."""
    values, context = flatten_static_cache(cache)
    return [(torch.utils._pytree.MappingKey(k), v) for k, v in zip(context, values)], context


def unflatten_static_cache(
    values: List[Any], context: torch.utils._pytree.Context, output_type=None
) -> StaticCache:
    """Restores a :class:`transformers.cache_utils.StaticCache` from python objects."""
    return make_static_cache(list(zip(values[0], values[1])))


####################
# SlidingWindowCache
####################


def flatten_sliding_window_cache(
    cache: SlidingWindowCache,
) -> Tuple[List[Any], torch.utils._pytree.Context]:
    """
    Serializes a :class:`transformers.cache_utils.SlidingWindowCache`
    with python objects.
    """
    flat = [("key_cache", cache.key_cache), ("value_cache", cache.value_cache)]
    return [f[1] for f in flat], [f[0] for f in flat]


def flatten_with_keys_sliding_window_cache(
    cache: SlidingWindowCache,
) -> Tuple[List[Tuple[torch.utils._pytree.KeyEntry, Any]], torch.utils._pytree.Context]:
    """
    Serializes a :class:`transformers.cache_utils.SlidingWindowCache`
    with python objects.
    """
    values, context = flatten_sliding_window_cache(cache)
    return [(torch.utils._pytree.MappingKey(k), v) for k, v in zip(context, values)], context


def unflatten_sliding_window_cache(
    values: List[Any], context: torch.utils._pytree.Context, output_type=None
) -> SlidingWindowCache:
    """Restores a :class:`transformers.cache_utils.SlidingWindowCache` from python objects."""
    key_cache, value_cache = values

    class _config:
        def __init__(self):
            self.head_dim = key_cache[0].shape[-1]
            self.num_attention_heads = key_cache[0].shape[1]
            self.num_hidden_layers = len(key_cache)
            self.sliding_window = key_cache[0].shape[2]

    cache = SlidingWindowCache(
        _config(),
        max_batch_size=key_cache[0].shape[0],
        max_cache_len=key_cache[0].shape[2],  # sligding window
        device=key_cache[0].device,
        dtype=key_cache[0].dtype,
    )

    values = dict(zip(context, values))
    for k, v in values.items():
        setattr(cache, k, v)
    return cache


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
    return EncoderDecoderCache(**dictionary)


#############
# dataclasses
#############


def _lower_name_with_(name):
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def make_serialization_function_for_dataclass(cls) -> Tuple[Callable, Callable, Callable]:
    """
    Automatically creates serialization function for a class decorated with
    ``dataclasses.dataclass``.
    """

    def flatten_cls(obj: cls) -> Tuple[List[Any], torch.utils._pytree.Context]:
        """Serializes a ``%s`` with python objects."""
        return list(obj.values()), list(obj.keys())

    def flatten_with_keys_cls(
        obj: cls,
    ) -> Tuple[List[Tuple[torch.utils._pytree.KeyEntry, Any]], torch.utils._pytree.Context]:
        """Serializes a ``%s`` with python objects with keys."""
        values, context = list(obj.values()), list(obj.keys())
        return [
            (torch.utils._pytree.MappingKey(k), v) for k, v in zip(context, values)
        ], context

    def unflatten_cls(
        values: List[Any], context: torch.utils._pytree.Context, output_type=None
    ) -> cls:
        """Restores an instance of ``%s`` from python objects."""
        return cls(**dict(zip(context, values)))

    name = _lower_name_with_(cls.__name__)
    flatten_cls.__name__ = f"flatten_{name}"
    flatten_with_keys_cls.__name__ = f"flatten_with_keys_{name}"
    unflatten_cls.__name__ = f"unflatten_{name}"
    flatten_cls.__doc__ = flatten_cls.__doc__ % cls.__name__
    flatten_with_keys_cls.__doc__ = flatten_with_keys_cls.__doc__ % cls.__name__
    unflatten_cls.__doc__ = unflatten_cls.__doc__ % cls.__name__
    SUPPORTED_DATACLASSES.add(cls)
    return flatten_cls, flatten_with_keys_cls, unflatten_cls


(
    flatten_base_model_output,
    flatten_with_keys_base_model_output,
    unflatten_base_model_output,
) = make_serialization_function_for_dataclass(BaseModelOutput)


if UNet2DConditionOutput is not None:
    (
        flatten_u_net2_d_condition_output,
        flatten_with_keys_u_net2_d_condition_output,
        unflatten_u_net2_d_condition_output,
    ) = make_serialization_function_for_dataclass(UNet2DConditionOutput)
