import pprint
from typing import Any, Callable, Dict, Optional, Set
import packaging.version as pv
import optree
import torch
import transformers
from transformers.cache_utils import DynamicCache, StaticCache

try:
    from transformers.cache_utils import EncoderDecoderCache
except ImportError:
    EncoderDecoderCache = None
from ..helpers import string_type
from .serialization import _lower_name_with_

PATCH_OF_PATCHES: Set[Any] = set()


def get_mamba_cache_cls() -> type:
    try:
        from transformers.models.mamba.modeling_mamba import MambaCache

        return MambaCache
    except ImportError:
        try:
            from transformers.cache_utils import MambaCache

            return MambaCache
        except ImportError:
            return None


def get_hybrid_cache_cls() -> type:
    try:
        from transformers.cache_utils import HybridCache

        return HybridCache
    except ImportError:
        return None


def get_sliding_window_cache_cls() -> type:
    try:
        from transformers.cache_utils import SlidingWindowCache

        return SlidingWindowCache
    except ImportError:
        return None


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
    It can be undone with
    :func:`onnx_diagnostic.torch_export_patches.onnx_export_serialization.unregister_class_serialization`.

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


def register_cache_serialization(
    patch_transformers: bool = False, patch_diffusers: bool = True, verbose: int = 0
) -> Dict[str, bool]:
    """
    Registers many classes with
    :func:`onnx_diagnostic.torch_export_patches.onnx_export_serialization.register_class_serialization`.
    Returns information needed to undo the registration.

    :param patch_transformers: add serialization function for
        :epkg:`transformers` package
    :param patch_diffusers: add serialization function for
        :epkg:`diffusers` package
    :param verbosity: verbosity level
    :return: information to unpatch
    """
    wrong: Dict[type, Optional[str]] = {}
    if patch_transformers:
        from .serialization.transformers_impl import WRONG_REGISTRATIONS

        wrong |= WRONG_REGISTRATIONS
    if patch_diffusers:
        from .serialization.diffusers_impl import WRONG_REGISTRATIONS

        wrong |= WRONG_REGISTRATIONS

    registration_functions = serialization_functions(
        patch_transformers=patch_transformers, patch_diffusers=patch_diffusers, verbose=verbose
    )

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
    for cls, version in wrong.items():
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
                f"available options are {list(registration_functions)}"
            )
            if verbose:
                print(
                    f"[_fix_registration] {cls.__name__} is unregistered and "
                    f"registered first"
                )
            unregister_class_serialization(cls, verbose=verbose)
            registration_functions[cls](verbose=verbose)  # type: ignore[arg-type, call-arg]
            if verbose:
                print(f"[_fix_registration] {cls.__name__} done.")
            # To avoid doing it multiple times.
            PATCH_OF_PATCHES.add(cls)

    # classes with no registration at all.
    done = {}
    for k, v in registration_functions.items():
        done[k] = v(verbose=verbose)  # type: ignore[arg-type, call-arg]
    return done


def serialization_functions(
    patch_transformers: bool = False, patch_diffusers: bool = False, verbose: int = 0
) -> Dict[type, Callable[[int], bool]]:
    """Returns the list of serialization functions."""

    supported_classes: Set[type] = set()
    classes: Dict[type, Callable[[int], bool]] = {}
    all_functions: Dict[type, Optional[str]] = {}

    if patch_transformers:
        from .serialization.transformers_impl import (
            __dict__ as dtr,
            SUPPORTED_DATACLASSES,
            flatten_dynamic_cache,
            unflatten_dynamic_cache,
            flatten_with_keys_dynamic_cache,
            flatten_encoder_decoder_cache,
            unflatten_encoder_decoder_cache,
            flatten_with_keys_encoder_decoder_cache,
            flatten_static_cache,
            unflatten_static_cache,
            flatten_with_keys_static_cache,
        )

        all_functions.update(dtr)
        supported_classes |= SUPPORTED_DATACLASSES

        transformers_classes = {
            DynamicCache: lambda verbose=verbose: register_class_serialization(
                DynamicCache,
                flatten_dynamic_cache,
                unflatten_dynamic_cache,
                flatten_with_keys_dynamic_cache,
                # f_check=make_dynamic_cache([(torch.rand((4, 4, 4)), torch.rand((4, 4, 4)))]),
                verbose=verbose,
            ),
            EncoderDecoderCache: lambda verbose=verbose: register_class_serialization(
                EncoderDecoderCache,
                flatten_encoder_decoder_cache,
                unflatten_encoder_decoder_cache,
                flatten_with_keys_encoder_decoder_cache,
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
        MambaCache = get_mamba_cache_cls()
        if MambaCache:
            from .serialization.transformers_impl import (
                flatten_mamba_cache,
                unflatten_mamba_cache,
                flatten_with_keys_mamba_cache,
            )

            transformers_classes[MambaCache] = (
                lambda verbose=verbose: register_class_serialization(
                    MambaCache,
                    flatten_mamba_cache,
                    unflatten_mamba_cache,
                    flatten_with_keys_mamba_cache,
                    verbose=verbose,
                )
            )
        HybridCache = get_hybrid_cache_cls()
        if HybridCache:
            from .serialization.transformers_impl import (
                flatten_hybrid_cache,
                unflatten_hybrid_cache,
                flatten_with_keys_hybrid_cache,
            )

            transformers_classes[HybridCache] = (
                lambda verbose=verbose: register_class_serialization(
                    HybridCache,
                    flatten_hybrid_cache,
                    unflatten_hybrid_cache,
                    flatten_with_keys_hybrid_cache,
                    verbose=verbose,
                )
            )

        SlidingWindowCache = get_sliding_window_cache_cls()
        if SlidingWindowCache:
            from .serialization.transformers_impl import (
                flatten_sliding_window_cache,
                unflatten_sliding_window_cache,
                flatten_with_keys_sliding_window_cache,
            )

            transformers_classes[SlidingWindowCache] = (
                lambda verbose=verbose: register_class_serialization(
                    SlidingWindowCache,
                    flatten_sliding_window_cache,
                    unflatten_sliding_window_cache,
                    flatten_with_keys_sliding_window_cache,
                    verbose=verbose,
                )
            )

        classes.update(transformers_classes)

    if patch_diffusers:
        from .serialization.diffusers_impl import SUPPORTED_DATACLASSES, __dict__ as dfu

        all_functions.update(dfu)
        supported_classes |= SUPPORTED_DATACLASSES

    for cls in supported_classes:
        lname = _lower_name_with_(cls.__name__)
        assert (
            f"flatten_{lname}" in all_functions
        ), f"Unable to find function 'flatten_{lname}' in {list(all_functions)}"
        classes[cls] = (
            lambda verbose=verbose, _ln=lname, cls=cls, _al=all_functions: register_class_serialization(  # noqa: E501
                cls,
                _al[f"flatten_{_ln}"],
                _al[f"unflatten_{_ln}"],
                _al[f"flatten_with_keys_{_ln}"],
                verbose=verbose,
            )
        )
    return classes


def unregister_class_serialization(cls: type, verbose: int = 0):
    """Undo the registration for a class."""
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
    """
    Undo the registration made by
    :func:`onnx_diagnostic.torch_export_patches.onnx_export_serialization.register_cache_serialization`.
    """
    cls_ensemble = {DynamicCache, EncoderDecoderCache} | set(undo)
    for cls in cls_ensemble:
        if undo.get(cls.__name__, False):
            unregister_class_serialization(cls, verbose)
