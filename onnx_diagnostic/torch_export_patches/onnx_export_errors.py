import contextlib
import pprint
from typing import Any, Callable, Dict, List, Optional, Set
from .onnx_export_serialization import (
    flatten_with_keys_dynamic_cache,
    flatten_dynamic_cache,
    unflatten_dynamic_cache,
    flatten_mamba_cache,
    flatten_with_keys_mamba_cache,
    unflatten_mamba_cache,
)
from .patches import patch_transformers as patch_transformers_list


def patch_module_or_classes(mod, verbose: int = 0) -> Dict[type, Dict[type, Callable]]:
    """
    Applies all patches defined in classes prefixed by ``patched_``
    ``cls._PATCHED_CLASS_`` defines the class to patch,
    ``cls._PATCHES_`` defines the method to patch.
    The returns information needs to be sent to :func:`unpatch_module_or_classes`
    to revert the changes.

    :param mod: module of list of clsses to patch
    :param verbose: verbosity
    :return: patch info
    """
    if isinstance(mod, list):
        to_patch = mod
        name = "list"
    else:
        to_patch = []
        for k in dir(mod):
            if k.startswith("patched_"):
                v = getattr(mod, k)
                if hasattr(v, "_PATCHED_CLASS_") and hasattr(v, "_PATCHES_"):
                    to_patch.append(v)
        name = mod.__name__

    res = {}
    for cls in to_patch:
        original = cls._PATCHED_CLASS_
        methods = cls._PATCHES_
        if verbose:
            print(f"[patch_module_or_classes] {name} - {cls.__name__}: {', '.join(methods)}")

        keep = {n: getattr(original, n, None) for n in methods}
        for n in methods:
            setattr(original, n, getattr(cls, n))
        res[cls] = keep

    return res


def unpatch_module_or_classes(mod, info: Dict[type, Dict[type, Callable]], verbose: int = 0):
    """
    Reverts modification made by :func:`patch_module_or_classes`.

    :param mod: module of list of clsses to patch
    :param verbose: verbosity
    """
    if isinstance(mod, list):
        to_patch = mod
        name = "list"
    else:
        to_patch = []
        for k in dir(mod):
            if k.startswith("patched_"):
                v = getattr(mod, k)
                if hasattr(v, "_PATCHED_CLASS_") and hasattr(v, "_PATCHES_"):
                    to_patch.append(v)
        name = mod.__name__
    set_patch = set(to_patch)

    for cls, methods in info.items():
        assert cls in set_patch, f"No patch registered for {cls} in {mod} (found {set_patch})"
        if verbose:
            print(f"[unpatch_module_or_classes] {name} - {cls.__name__}: {', '.join(methods)}")
        original = cls._PATCHED_CLASS_
        for n, v in methods.items():
            if v is None:
                # The method did not exist. We remove it.
                delattr(original, n)
            else:
                setattr(original, n, v)


PATCH_OF_PATCHES: Set[Any] = set()


def _register_cache_serialization(verbose: int = 0) -> Dict[str, bool]:
    # Cache serialization: to be moved into appropriate packages
    import torch
    import transformers
    import packaging.version as pv

    try:
        from transformers.cache_utils import DynamicCache
    except ImportError:
        DynamicCache = None

    try:
        from transformers.cache_utils import MambaCache
    except ImportError:
        MambaCache = None

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

    return dict(DynamicCache=unregistered_dynamic_cache, MambaCache=unregistered_mamba_cache)


def _unregister(cls: type, verbose: int = 0):
    import optree
    import torch

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
        from transformers.cache_utils import MambaCache

        _unregister(MambaCache, verbose)
    elif verbose > 1:
        print("[_unregister_cache_serialization] skip unregister MambaCache")

    if undo.get("DynamicCache", False):
        from transformers.cache_utils import DynamicCache

        _unregister(DynamicCache, verbose)
    elif verbose > 1:
        print("[_unregister_cache_serialization] skip unregister DynamicCache")


@contextlib.contextmanager
def register_additional_serialization_functions(
    patch_transformers: bool = False, verbose: int = 0
) -> Callable:
    """The necessary modifications to run the fx Graph."""
    fct_callable = replacement_before_exporting if patch_transformers else (lambda x: x)
    done = _register_cache_serialization(verbose=verbose)
    try:
        yield fct_callable
    finally:
        _unregister_cache_serialization(done, verbose=verbose)


@contextlib.contextmanager
def bypass_export_some_errors(
    patch_sympy: bool = True,
    patch_torch: bool = True,
    patch_transformers: bool = False,
    catch_constraints: bool = True,
    stop_if_static: int = 0,
    verbose: int = 0,
    patch: bool = True,
    custom_patches: Optional[List[type["torch.nn.Module"]]] = None,  # noqa: F821
) -> Callable:
    """
    Tries to bypass some situations :func:`torch.export.export` does not support.

    :param patch_sympy: fix missing method ``name`` for IntegerConstant
    :param patch_torch: patches :epkg:`torch` with supported implementation
    :param patch_transformers: patches :epkg:`transformers` with supported implementation
    :param catch_constraints: catch constraints related to dynamic shapes,
        as a result, some dynamic dimension may turn into static ones,
        the environment variable ``SKIP_SOLVE_CONSTRAINTS=0``
        can be put to stop at that stage.
    :param stop_if_static: see example :ref:`l-plot-export-locale-issue`,
        to stop the export as soon as an issue is detected with dynamic shapes
        and show a stack trace indicating the exact location of the issue,
        ``if stop_if_static > 1``, more methods are replace to catch more
        issues
    :param patch: if False, disable all patches except the registration of
        serialization function
    :param custom_patches: to apply custom patches,
        every patched class must define static attributes
        ``_PATCHES_``, ``_PATCHED_CLASS_``
    :param verbose: to show which patches is applied

    The list of available patches.

    * ``torch.jit.isinstance``
    * ``torch._dynamo.mark_static_address``
    * ``torch._subclasses.fake_impls.infer_size``
    * fix missing method ``name`` for ``sympy.S.IntegerConstant``
    * ``AttentionMaskConverter._make_causal_mask``
    * Serialization of ``MambaCache`` (in :epkg:`transformers`)
    * Serialization of ``DynamicCache`` (in :epkg:`transformers`)
    * reduce errors due to shape inference
    * fixes some transformers classes,
      see :mod:`onnx_diagnostic.torch_export_patches.patches.patch_transformers`

    Serialization issues happen when a module takes one input or output
    has a type :func:`torch.export.export` cannot serialize.

    Examples:

    ::

        with bypass_export_some_errors(patch_transformers=True) as modificator:
            inputs = modificator(inputs)
            onx = to_onnx(..., inputs, ...)

    ::

        with bypass_export_some_errors(patch_transformers=True) as modificator:
            inputs = modificator(inputs)
            onx = torch.onnx.export(..., inputs, ...)

    It can be used as well to fix the torch export:

    ::

        with bypass_export_some_errors(patch_transformers=True) as modificator:
            inputs = modificator(inputs)
            ep = torch.export.export(..., inputs, ...)

    When running the model through the exported program, only the
    serialization functions need to be restored:

    ::

        with register_additional_serialization_functions() as modificator:
            inputs = modificator(inputs)
            ep = torch.export.export(..., inputs, ...)

    When exporting a model with a cache, the following error message
    may appear ``AssertionError: Mutating module attribute _seen_tokens during export.``.
    It can be avoided by setting ``strict=False`` when call :func:`torch.export.export`.
    """
    if not patch:
        fct_callable = lambda x: x  # noqa: E731
        done = _register_cache_serialization(verbose=verbose)
        try:
            yield fct_callable
        finally:
            _unregister_cache_serialization(done, verbose=verbose)
    else:
        import torch
        import torch._export.non_strict_utils  # produce_guards_and_solve_constraints
        import torch.jit

        if verbose:
            print(
                "[bypass_export_some_errors] replace torch.jit.isinstance, "
                "torch._dynamo.mark_static_address"
            )

        ########
        # caches
        ########

        cache_done = _register_cache_serialization(verbose=verbose)

        #############
        # patch sympy
        #############

        if patch_sympy:
            import sympy

            f_sympy_name = getattr(sympy.core.numbers.IntegerConstant, "name", None)

            if verbose:
                print(f"[bypass_export_some_errors] sympy.__version__={sympy.__version__!r}")
                print("[bypass_export_some_errors] patch sympy")

            sympy.core.numbers.IntegerConstant.name = lambda self: f"IntCst{str(self)}"

        ###############
        # patch pytorch
        ###############

        if patch_torch:
            from .patches.patch_torch import (
                patched_infer_size,
                patched__broadcast_shapes,
                _catch_produce_guards_and_solve_constraints,
                patch__check_input_constraints_for_graph,
            )

            if verbose:
                print(f"[bypass_export_some_errors] torch.__version__={torch.__version__!r}")
                print(f"[bypass_export_some_errors] stop_if_static={stop_if_static!r}")
                print("[bypass_export_some_errors] patch pytorch")

            # torch.jit.isinstance
            f_jit_isinstance = torch.jit.isinstance
            torch.jit.isinstance = isinstance

            # torch._dynamo.mark_static_address
            f_mark_static_address = torch._dynamo.mark_static_address
            torch._dynamo.mark_static_address = lambda *_, **y_: None

            # torch._subclasses.fake_impls.infer_size
            f_infer_size = torch._subclasses.fake_impls.infer_size
            torch._subclasses.fake_impls.infer_size = patched_infer_size

            # torch._refs._broadcast_shapes
            f__broadcast_shapes = torch._refs._broadcast_shapes
            torch._refs._broadcast_shapes = patched__broadcast_shapes
            torch._meta_registrations._broadcast_shapes = patched__broadcast_shapes

        # torch._export.non_strict_utils.produce_guards_and_solve_constraints
        if catch_constraints:
            if verbose:
                print("[bypass_export_some_errors] modifies shape constraints")
            f_produce_guards_and_solve_constraints = (
                torch._export.non_strict_utils.produce_guards_and_solve_constraints
            )
            f__check_input_constraints_for_graph = (
                torch._export.utils._check_input_constraints_for_graph
            )
            torch._export.non_strict_utils.produce_guards_and_solve_constraints = (
                lambda *args, **kwargs: _catch_produce_guards_and_solve_constraints(
                    f_produce_guards_and_solve_constraints, *args, verbose=verbose, **kwargs
                )
            )
            torch._export.utils._check_input_constraints_for_graph = (
                lambda *args, **kwargs: patch__check_input_constraints_for_graph(
                    f__check_input_constraints_for_graph, *args, verbose=verbose, **kwargs
                )
            )

        if stop_if_static:
            from torch.fx.experimental.symbolic_shapes import ShapeEnv
            from .patches.patch_torch import patched_ShapeEnv

            ShapeEnv._log_guard_remember = ShapeEnv._log_guard

            if verbose:
                print(
                    "[bypass_export_some_errors] assert when a dynamic dimension turns static"
                )
                print("[bypass_export_some_errors] replaces ShapeEnv._set_replacement")

            f_shape_env__set_replacement = ShapeEnv._set_replacement
            ShapeEnv._set_replacement = patched_ShapeEnv._set_replacement

            if verbose:
                print("[bypass_export_some_errors] replaces ShapeEnv._log_guard")
            f_shape_env__log_guard = ShapeEnv._log_guard
            ShapeEnv._log_guard = patched_ShapeEnv._log_guard

            if stop_if_static > 1:
                if verbose:
                    print("[bypass_export_some_errors] replaces ShapeEnv._check_frozen")
                f_shape_env__check_frozen = ShapeEnv._check_frozen
                ShapeEnv._check_frozen = patched_ShapeEnv._check_frozen

        ####################
        # patch transformers
        ####################

        if patch_transformers:
            if verbose:
                import transformers

                print(
                    f"[bypass_export_some_errors] transformers.__version__="
                    f"{transformers.__version__!r}"
                )
            revert_patches_info = patch_module_or_classes(
                patch_transformers_list, verbose=verbose
            )

        if custom_patches:
            if verbose:
                print("[bypass_export_some_errors] applies custom patches")
            revert_custom_patches_info = patch_module_or_classes(
                custom_patches, verbose=verbose
            )

        ########
        # export
        ########

        fct_callable = replacement_before_exporting if patch_transformers else (lambda x: x)

        if verbose:
            print("[bypass_export_some_errors] done patching")

        try:
            yield fct_callable
        finally:
            #######
            # sympy
            #######

            if verbose:
                print("[bypass_export_some_errors] remove patches")

            if patch_sympy:
                # tracked by https://github.com/pytorch/pytorch/issues/143494
                if f_sympy_name:
                    sympy.core.numbers.IntegerConstant.name = f_sympy_name
                else:
                    delattr(sympy.core.numbers.IntegerConstant, "name")

                if verbose:
                    print("[bypass_export_some_errors] restored sympy functions")

            #######
            # torch
            #######

            if patch_torch:
                # this should disappear when torch.jit is removed
                torch.jit.isinstance = f_jit_isinstance
                torch._dynamo.mark_static_address = f_mark_static_address
                # tracked by https://github.com/pytorch/pytorch/issues/143495
                torch._subclasses.fake_impls.infer_size = f_infer_size
                torch._refs._broadcast_shapes = f__broadcast_shapes
                torch._meta_registrations._broadcast_shapes = f__broadcast_shapes

                if verbose:
                    print("[bypass_export_some_errors] restored pytorch functions")

            if stop_if_static:
                if verbose:
                    print("[bypass_export_some_errors] restored ShapeEnv._set_replacement")

                ShapeEnv._set_replacement = f_shape_env__set_replacement

                if verbose:
                    print("[bypass_export_some_errors] restored ShapeEnv._log_guard")

                ShapeEnv._log_guard = f_shape_env__log_guard

                if stop_if_static > 1:
                    if verbose:
                        print("[bypass_export_some_errors] restored ShapeEnv._check_frozen")
                    ShapeEnv._check_frozen = f_shape_env__check_frozen

            if catch_constraints:
                # to catch or skip dynamic_shapes issues
                torch._export.non_strict_utils.produce_guards_and_solve_constraints = (
                    f_produce_guards_and_solve_constraints
                )
                torch._export.utils._check_input_constraints_for_graph = (
                    f__check_input_constraints_for_graph
                )
                if verbose:
                    print("[bypass_export_some_errors] restored shape constraints")

            if custom_patches:
                if verbose:
                    print("[bypass_export_some_errors] unpatch custom patches")
                unpatch_module_or_classes(
                    custom_patches, revert_custom_patches_info, verbose=verbose
                )

            ##############
            # transformers
            ##############

            if patch_transformers:
                if verbose:
                    print("[bypass_export_some_errors] unpatch transformers")
                unpatch_module_or_classes(
                    patch_transformers_list, revert_patches_info, verbose=verbose
                )

            ########
            # caches
            ########

            _unregister_cache_serialization(cache_done, verbose=verbose)


def replacement_before_exporting(args: Any) -> Any:
    """
    Does replacements on the given inputs if needed.
    """
    if args is None:
        return None
    if isinstance(args, (int, float)):
        return args
    if isinstance(args, dict):
        return {k: replacement_before_exporting(v) for k, v in args.items()}
    if isinstance(args, tuple):
        return tuple(replacement_before_exporting(v) for v in args)
    if isinstance(args, list):
        return [replacement_before_exporting(v) for v in args]

    return args
