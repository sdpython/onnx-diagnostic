import pprint
from typing import Any, Dict, List, Set, Tuple
import optree
import torch
import transformers
import packaging.version as pv


PATCH_OF_PATCHES: Set[Any] = set()


def _register_cache_serialization(verbose: int = 0) -> Dict[str, bool]:
    # Cache serialization: to be moved into appropriate packages

    try:
        from transformers.cache_utils import DynamicCache
    except ImportError:
        DynamicCache = None

    try:
        from transformers.cache_utils import MambaCache
    except ImportError:
        MambaCache = None

    try:
        from transformers.cache_utils import EncoderDecoderCache
    except ImportError:
        EncoderDecoderCache = None

    # MambaCache
    unregistered_mamba_cache = True
    if MambaCache is not None and MambaCache in torch.utils._pytree.SUPPORTED_NODES:
        if verbose > 1:
            print(f"[_register_cache_serialization] {MambaCache} already registered")
        # It is already registered because bypass_export_some_errors was called
        # within a section already calling bypass_export_some_errors or transformers
        # has updated its code to do it.
        # No need to register and unregister then.
        unregistered_mamba_cache = False
    else:
        if verbose:
            print("[_register_cache_serialization] register MambaCache")
        torch.utils._pytree.register_pytree_node(
            MambaCache,
            flatten_mamba_cache,
            unflatten_mamba_cache,
            serialized_type_name=f"{MambaCache.__module__}.{MambaCache.__name__}",
            flatten_with_keys_fn=flatten_with_keys_mamba_cache,
        )

    # DynamicCache serialization is different in transformers and does not
    # play way with torch.export.export.
    # see test test_export_dynamic_cache_cat with NOBYPASS=1
    # :: NOBYBASS=1 python _unittests/ut_torch_export_patches/test_dynamic_class.py -k e_c
    # This is caused by this line:
    # torch.fx._pytree.register_pytree_flatten_spec(
    #           DynamicCache, _flatten_dynamic_cache_for_fx)
    # so we remove it anyway
    if (
        DynamicCache in torch.fx._pytree.SUPPORTED_NODES
        and not PATCH_OF_PATCHES
        # and pv.Version(torch.__version__) < pv.Version("2.7")
        and pv.Version(transformers.__version__) >= pv.Version("4.50")
    ):
        if verbose:
            print(
                "[_register_cache_serialization] DynamicCache "
                "is unregistered and registered first."
            )
        _unregister(DynamicCache)
        torch.utils._pytree.register_pytree_node(
            DynamicCache,
            flatten_dynamic_cache,
            unflatten_dynamic_cache,
            serialized_type_name=f"{DynamicCache.__module__}.{DynamicCache.__name__}",
            flatten_with_keys_fn=flatten_with_keys_dynamic_cache,
        )
        if pv.Version(torch.__version__) < pv.Version("2.7"):
            torch.fx._pytree.register_pytree_flatten_spec(
                DynamicCache, lambda x, _: [x.key_cache, x.value_cache]
            )
        # To avoid doing it multiple times.
        PATCH_OF_PATCHES.add(DynamicCache)

    unregistered_dynamic_cache = True
    if DynamicCache is not None and DynamicCache in torch.utils._pytree.SUPPORTED_NODES:
        if verbose > 1:
            print(f"[_register_cache_serialization] {DynamicCache} already registered")
        unregistered_dynamic_cache = False
    else:
        if verbose:
            print("[_register_cache_serialization] register DynamicCache")
        torch.utils._pytree.register_pytree_node(
            DynamicCache,
            flatten_dynamic_cache,
            unflatten_dynamic_cache,
            serialized_type_name=f"{DynamicCache.__module__}.{DynamicCache.__name__}",
            flatten_with_keys_fn=flatten_with_keys_dynamic_cache,
        )
        if pv.Version(torch.__version__) < pv.Version("2.7"):
            torch.fx._pytree.register_pytree_flatten_spec(
                DynamicCache, lambda x, _: [x.key_cache, x.value_cache]
            )

        # check
        from ..helpers.cache_helper import make_dynamic_cache

        cache = make_dynamic_cache([(torch.rand((4, 4, 4)), torch.rand((4, 4, 4)))])
        values, spec = torch.utils._pytree.tree_flatten(cache)
        cache2 = torch.utils._pytree.tree_unflatten(values, spec)
        # torch.fx._pytree.tree_flatten(cache)
        assert len(cache2.key_cache) == 1

    # EncoderDecoderCache
    unregistered_encode_decode_cache = True
    if (
        EncoderDecoderCache is not None
        and EncoderDecoderCache in torch.utils._pytree.SUPPORTED_NODES
    ):
        if verbose > 1:
            print(f"[_register_cache_serialization] {EncoderDecoderCache} already registered")
        # It is already registered because bypass_export_some_errors was called
        # within a section already calling bypass_export_some_errors or transformers
        # has updated its code to do it.
        # No need to register and unregister then.
        unregistered_mamba_cache = False
    else:
        if verbose:
            print("[_register_cache_serialization] register EncoderDecoderCache")
        torch.utils._pytree.register_pytree_node(
            EncoderDecoderCache,
            flatten_encoder_decoder_cache,
            unflatten_encoder_decoder_cache,
            serialized_type_name=f"{EncoderDecoderCache.__module__}.{EncoderDecoderCache.__name__}",
            flatten_with_keys_fn=flatten_with_keys_encoder_decoder_cache,
        )

    return dict(
        DynamicCache=unregistered_dynamic_cache,
        MambaCache=unregistered_mamba_cache,
        EncoderDecoderCache=unregistered_encode_decode_cache,
    )


def _unregister(cls: type, verbose: int = 0):
    # torch.fx._pytree._deregister_pytree_flatten_spec(cls)
    if cls in torch.fx._pytree.SUPPORTED_NODES:
        del torch.fx._pytree.SUPPORTED_NODES[cls]
    if cls in torch.fx._pytree.SUPPORTED_NODES_EXACT_MATCH:
        del torch.fx._pytree.SUPPORTED_NODES_EXACT_MATCH[cls]
    if hasattr(torch.utils._pytree, "_deregister_pytree_node"):
        # torch >= 2.7
        torch.utils._pytree._deregister_pytree_node(cls)
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
        print(f"[_unregister_cache_serialization] unregistered {cls.__name__}")


def _unregister_cache_serialization(undo: Dict[str, bool], verbose: int = 0):
    if undo.get("MambaCache", False):
        _unregister(transformers.cache_utils.MambaCache, verbose)
    elif verbose > 1:
        print("[_unregister_cache_serialization] skip unregister MambaCache")

    if undo.get("DynamicCache", False):
        _unregister(transformers.cache_utils.DynamicCache, verbose)
    elif verbose > 1:
        print("[_unregister_cache_serialization] skip unregister DynamicCache")

    if undo.get("EncoderDecoderCache", False):
        _unregister(transformers.cache_utils.EncoderDecoderCache, verbose)
    elif verbose > 1:
        print("[_unregister_cache_serialization] skip unregister EncoderDecoderCache")


############
# MambaCache
############


# self.conv_states: torch.Tensor = torch.zeros(
#     config.num_hidden_layers,
#     self.max_batch_size,
#     self.intermediate_size,
#     self.conv_kernel_size,
#     device=device,
#     dtype=dtype,
# )
# self.ssm_states: torch.Tensor = torch.zeros(
#     config.num_hidden_layers,
#     self.max_batch_size,
#     self.intermediate_size,
#     self.ssm_state_size,
#     device=device,
#     dtype=dtype,
# )
def flatten_mamba_cache(
    mamba_cache: transformers.cache_utils.MambaCache,
) -> Tuple[List[Any], torch.utils._pytree.Context]:
    """Serializes a :class:`transformers.cache_utils.MambaCache` with python objects."""
    flat = [
        (k, getattr(mamba_cache, k))
        for k in [
            # "max_batch_size",  # new in transformers==4.47
            # "intermediate_size",
            # "ssm_state_size",
            # "conv_kernel_size",
            "conv_states",
            "ssm_states",
        ]
        if hasattr(mamba_cache, k)
    ]
    return [f[1] for f in flat], [f[0] for f in flat]


def unflatten_mamba_cache(
    values: List[Any],
    context: torch.utils._pytree.Context,
    output_type=None,
) -> transformers.cache_utils.MambaCache:
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

    from transformers.cache_utils import MambaCache

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


def flatten_with_keys_mamba_cache(d: Dict[Any, Any]) -> Tuple[
    List[Tuple[torch.utils._pytree.KeyEntry, Any]],
    torch.utils._pytree.Context,
]:
    """Serializes a :class:`transformers.cache_utils.MambaCache` with python objects."""
    values, context = flatten_mamba_cache(d)
    return [(torch.utils._pytree.MappingKey(k), v) for k, v in zip(context, values)], context


##############
# DynamicCache
##############


def flatten_dynamic_cache(
    dynamic_cache: transformers.cache_utils.DynamicCache,
) -> Tuple[List[Any], torch.utils._pytree.Context]:
    """Serializes a :class:`transformers.cache_utils.DynamicCache` with python objects."""
    if hasattr(transformers.cache_utils, "_flatten_dynamic_cache"):
        return transformers.cache_utils._flatten_dynamic_cache(dynamic_cache)
    flat = [
        (k, getattr(dynamic_cache, k))
        for k in ["key_cache", "value_cache"]
        if hasattr(dynamic_cache, k)
    ]
    return [f[1] for f in flat], [f[0] for f in flat]


def flatten_with_keys_dynamic_cache(
    dynamic_cache: transformers.cache_utils.DynamicCache,
) -> Tuple[
    List[Tuple[torch.utils._pytree.KeyEntry, Any]],
    torch.utils._pytree.Context,
]:
    """Serializes a :class:`transformers.cache_utils.DynamicCache` with python objects."""
    if hasattr(transformers.cache_utils, "_flatten_with_keys_dynamic_cache"):
        return transformers.cache_utils._flatten_with_keys_dynamic_cache(dynamic_cache)
    values, context = flatten_dynamic_cache(dynamic_cache)
    return [(torch.utils._pytree.MappingKey(k), v) for k, v in zip(context, values)], context


def unflatten_dynamic_cache(
    values: List[Any],
    context: torch.utils._pytree.Context,
    output_type=None,
) -> transformers.cache_utils.DynamicCache:
    """Restores a :class:`transformers.cache_utils.DynamicCache` from python objects."""
    if hasattr(transformers.cache_utils, "_unflatten_dynamic_cache"):
        assert output_type is None, f"output_type={output_type} not supported"
        return transformers.cache_utils._unflatten_dynamic_cache(values, context)

    cache = transformers.cache_utils.DynamicCache()
    values = dict(zip(context, values))
    for k, v in values.items():
        setattr(cache, k, v)
    return cache


#####################
# EncoderDecoderCache
#####################


def flatten_encoder_decoder_cache(
    ec_cache: transformers.cache_utils.DynamicCache,
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


def flatten_with_keys_encoder_decoder_cache(
    ec_cache: transformers.cache_utils.DynamicCache,
) -> Tuple[
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
    values: List[Any],
    context: torch.utils._pytree.Context,
    output_type=None,
) -> transformers.cache_utils.EncoderDecoderCache:
    """Restores a :class:`transformers.cache_utils.EncoderDecoderCache` from python objects."""
    dictionary = torch.utils._pytree._dict_unflatten(values, context)
    return transformers.cache_utils.EncoderDecoderCache(**dictionary)
