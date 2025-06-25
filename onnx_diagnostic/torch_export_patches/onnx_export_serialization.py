import pprint
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import packaging.version as pv
import optree
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

from ..helpers import string_type
from ..helpers.cache_helper import make_static_cache


PATCH_OF_PATCHES: Set[Any] = set()
WRONG_REGISTRATIONS: Dict[str, str] = {
    DynamicCache: "4.50",
    BaseModelOutput: None,
    UNet2DConditionOutput: None,
}


def register_class_serialization(
    cls,
    f_flatten: Callable,
    f_unflatten: Callable,
    f_flatten_with_keys: Callable,
    f_check: Optional[Callable] = None,
    verbose: int = 0,
) -> bool:
    """
    Registers a class.
    It can be undone with :func:`unregister`.

    :param cls: class to register
    :param f_flatten: see ``torch.utils._pytree.register_pytree_node``
    :param f_unflatten: see ``torch.utils._pytree.register_pytree_node``
    :param f_flatten_with_keys: see ``torch.utils._pytree.register_pytree_node``
    :param f_check: called to check the registration was successful
    :param verbose: verbosity
    :return: registered or not
    """
    if cls is not None and cls in torch.utils._pytree.SUPPORTED_NODES:
        if verbose and cls is not None:
            print(f"[register_class_serialization] already registered {cls.__name__}")
        return False

    if verbose:
        print(f"[register_class_serialization] ---------- register {cls.__name__}")
    torch.utils._pytree.register_pytree_node(
        cls,
        f_flatten,
        f_unflatten,
        serialized_type_name=f"{cls.__module__}.{cls.__name__}",
        flatten_with_keys_fn=f_flatten_with_keys,
    )
    if pv.Version(torch.__version__) < pv.Version("2.7"):
        if verbose:
            print(
                f"[register_class_serialization] "
                f"---------- register {cls.__name__} for torch=={torch.__version__}"
            )
        torch.fx._pytree.register_pytree_flatten_spec(cls, lambda x, _: f_flatten(x)[0])

    # check
    if f_check:
        inst = f_check()
        values, spec = torch.utils._pytree.tree_flatten(inst)
        restored = torch.utils._pytree.tree_unflatten(values, spec)
        assert string_type(inst, with_shape=True) == string_type(restored, with_shape=True), (
            f"Issue with registration of class {cls} "
            f"inst={string_type(inst, with_shape=True)}, "
            f"restored={string_type(restored, with_shape=True)}"
        )
    return True


def register_cache_serialization(verbose: int = 0) -> Dict[str, bool]:
    """
    Registers many classes with :func:`register_class_serialization`.
    Returns information needed to undo the registration.
    """
    registration_functions = serialization_functions(verbose=verbose)

    # DynamicCache serialization is different in transformers and does not
    # play way with torch.export.export.
    # see test test_export_dynamic_cache_cat with NOBYPASS=1
    # :: NOBYBASS=1 python _unittests/ut_torch_export_patches/test_dynamic_class.py -k e_c
    # This is caused by this line:
    # torch.fx._pytree.register_pytree_flatten_spec(
    #           DynamicCache, _flatten_dynamic_cache_for_fx)
    # so we remove it anyway
    # BaseModelOutput serialization is incomplete.
    # It does not include dynamic shapes mapping.
    for cls, version in WRONG_REGISTRATIONS.items():
        if (
            cls in torch.utils._pytree.SUPPORTED_NODES
            and cls not in PATCH_OF_PATCHES
            # and pv.Version(torch.__version__) < pv.Version("2.7")
            and (
                version is None or pv.Version(transformers.__version__) >= pv.Version(version)
            )
        ):
            assert cls in registration_functions, (
                f"{cls} has no registration functions mapped to it, "
                f"available {sorted(registration_functions)}"
            )
            if verbose:
                print(
                    f"[_fix_registration] {cls.__name__} is unregistered and "
                    f"registered first"
                )
            unregister_class_serialization(cls, verbose=verbose)
            registration_functions[cls](verbose=verbose)
            if verbose:
                print(f"[_fix_registration] {cls.__name__} done.")
            # To avoid doing it multiple times.
            PATCH_OF_PATCHES.add(cls)

    # classes with no registration at all.
    done = {}
    for k, v in registration_functions.items():
        done[k] = v(verbose=verbose)
    return done


def serialization_functions(verbose: int = 0) -> Dict[type, Union[Callable[[], bool], int]]:
    """Returns the list of serialization functions."""
    transformers_classes = {
        DynamicCache: lambda verbose=verbose: register_class_serialization(
            DynamicCache,
            flatten_dynamic_cache,
            unflatten_dynamic_cache,
            flatten_with_keys_dynamic_cache,
            # f_check=make_dynamic_cache([(torch.rand((4, 4, 4)), torch.rand((4, 4, 4)))]),
            verbose=verbose,
        ),
        MambaCache: lambda verbose=verbose: register_class_serialization(
            MambaCache,
            flatten_mamba_cache,
            unflatten_mamba_cache,
            flatten_with_keys_mamba_cache,
            verbose=verbose,
        ),
        EncoderDecoderCache: lambda verbose=verbose: register_class_serialization(
            EncoderDecoderCache,
            flatten_encoder_decoder_cache,
            unflatten_encoder_decoder_cache,
            flatten_with_keys_encoder_decoder_cache,
            verbose=verbose,
        ),
        BaseModelOutput: lambda verbose=verbose: register_class_serialization(
            BaseModelOutput,
            flatten_base_model_output,
            unflatten_base_model_output,
            flatten_with_keys_base_model_output,
            verbose=verbose,
        ),
        SlidingWindowCache: lambda verbose=verbose: register_class_serialization(
            SlidingWindowCache,
            flatten_sliding_window_cache,
            unflatten_sliding_window_cache,
            flatten_with_keys_sliding_window_cache,
            verbose=verbose,
        ),
        StaticCache: lambda verbose=verbose: register_class_serialization(
            StaticCache,
            flatten_static_cache,
            unflatten_static_cache,
            flatten_with_keys_static_cache,
            verbose=verbose,
        ),
    }
    if UNet2DConditionOutput:
        diffusers_classes = {
            UNet2DConditionOutput: lambda verbose=verbose: register_class_serialization(
                UNet2DConditionOutput,
                flatten_unet_2d_condition_output,
                unflatten_unet_2d_condition_output,
                flatten_with_keys_unet_2d_condition_output,
                verbose=verbose,
            )
        }
        transformers_classes.update(diffusers_classes)
    return transformers_classes


def unregister_class_serialization(cls: type, verbose: int = 0):
    """Undo the registration."""
    # torch.utils._pytree._deregister_pytree_flatten_spec(cls)
    if cls in torch.fx._pytree.SUPPORTED_NODES:
        del torch.fx._pytree.SUPPORTED_NODES[cls]
    if cls in torch.fx._pytree.SUPPORTED_NODES_EXACT_MATCH:
        del torch.fx._pytree.SUPPORTED_NODES_EXACT_MATCH[cls]
    if hasattr(torch.utils._pytree, "_deregister_pytree_node"):
        # torch >= 2.7
        torch.utils._pytree._deregister_pytree_node(cls)
    else:
        if cls in torch.utils._pytree.SUPPORTED_NODES:
            del torch.utils._pytree.SUPPORTED_NODES[cls]
    optree.unregister_pytree_node(cls, namespace="torch")
    if cls in torch.utils._pytree.SUPPORTED_NODES:
        import packaging.version as pv

        if pv.Version(torch.__version__) < pv.Version("2.7.0"):
            del torch.utils._pytree.SUPPORTED_NODES[cls]
    assert cls not in torch.utils._pytree.SUPPORTED_NODES, (
        f"{cls} was not successful unregistered "
        f"from torch.utils._pytree.SUPPORTED_NODES="
        f"{pprint.pformat(list(torch.utils._pytree.SUPPORTED_NODES))}"
    )
    if verbose:
        print(f"[unregister_cache_serialization] unregistered {cls.__name__}")


def unregister_cache_serialization(undo: Dict[str, bool], verbose: int = 0):
    """Undo all registrations."""
    cls_ensemble = {MambaCache, DynamicCache, EncoderDecoderCache, BaseModelOutput} | set(undo)
    for cls in cls_ensemble:
        if undo.get(cls.__name__, False):
            unregister_class_serialization(cls, verbose)


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


#################
# BaseModelOutput
#################


def flatten_base_model_output(
    bo: BaseModelOutput,
) -> Tuple[List[Any], torch.utils._pytree.Context]:
    """
    Serializes a :class:`transformers.modeling_outputs.BaseModelOutput`
    with python objects.
    """
    return list(bo.values()), list(bo.keys())


def flatten_with_keys_base_model_output(
    bo: BaseModelOutput,
) -> Tuple[List[Tuple[torch.utils._pytree.KeyEntry, Any]], torch.utils._pytree.Context]:
    """
    Serializes a :class:`transformers.modeling_outputs.BaseModelOutput`
    with python objects.
    """
    values, context = flatten_base_model_output(bo)
    return [(torch.utils._pytree.MappingKey(k), v) for k, v in zip(context, values)], context


def unflatten_base_model_output(
    values: List[Any],
    context: torch.utils._pytree.Context,
    output_type=None,
) -> BaseModelOutput:
    """
    Restores a :class:`transformers.modeling_outputs.BaseModelOutput`
    from python objects.
    """
    return BaseModelOutput(**dict(zip(context, values)))


#######################
# UNet2DConditionOutput
#######################


def flatten_unet_2d_condition_output(
    obj: UNet2DConditionOutput,
) -> Tuple[List[Any], torch.utils._pytree.Context]:
    """
    Serializes a :class:`diffusers.models.unets.unet_2d_condition.UNet2DConditionOutput`
    with python objects.
    """
    return list(obj.values()), list(obj.keys())


def flatten_with_keys_unet_2d_condition_output(
    obj: UNet2DConditionOutput,
) -> Tuple[List[Tuple[torch.utils._pytree.KeyEntry, Any]], torch.utils._pytree.Context]:
    """
    Serializes a :class:`diffusers.models.unets.unet_2d_condition.UNet2DConditionOutput`
    with python objects.
    """
    values, context = flatten_unet_2d_condition_output(obj)
    return [(torch.utils._pytree.MappingKey(k), v) for k, v in zip(context, values)], context


def unflatten_unet_2d_condition_output(
    values: List[Any],
    context: torch.utils._pytree.Context,
    output_type=None,
) -> UNet2DConditionOutput:
    """
    Restores a :class:`diffusers.models.unets.unet_2d_condition.UNet2DConditionOutput`
    from python objects.
    """
    return UNet2DConditionOutput(**dict(zip(context, values)))
