import functools
import importlib
import contextlib
import re
from typing import Any, Callable, Dict, List, Optional, Tuple
from .onnx_export_serialization import (
    register_cache_serialization,
    unregister_cache_serialization,
)
from .patches import patch_transformers as patch_transformers_list


def get_function(name: str) -> Tuple[type, Callable]:
    """Returns the module and the function based on its name."""
    spl = name.split(".")
    module_name = ".".join(spl[:-1])
    fname = spl[-1]
    mod = importlib.import_module(module_name)
    if not hasattr(mod, fname):
        return None, None
    return mod, getattr(mod, fname)


@functools.lru_cache
def get_patches(mod, verbose: int = 0) -> Tuple[str, List[Any]]:
    """Returns the list of patches to make for a specific module."""
    to_patch = []
    for k in dir(mod):
        if k.startswith("patched_"):
            v = getattr(mod, k)
            if hasattr(v, "_PATCHED_CLASS_") and hasattr(v, "_PATCHES_"):
                to_patch.append(v)
            else:
                # a function
                doc = v.__doc__.lstrip()
                if doc.startswith("manual patch"):
                    continue
                reg = re.compile("[\\[]patch:([a-z_A-Z.]+)[\\]]")
                fall = reg.findall(doc)
                assert (
                    len(fall) == 1
                ), f"Unable to find patching information for {v} in \n{doc}"
                fmod, f = get_function(fall[0])
                if fmod is None and f is None:
                    # The function does not exist in this version of transformers.
                    # No patch is needed.
                    continue
                to_patch.append({"module": fmod, "function": f, "patch": v})

    name = mod.__name__
    return name, to_patch


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
        name, to_patch = get_patches(mod, verbose)

    res = {}
    for cls in to_patch:
        if isinstance(cls, dict):
            # a function
            keep = {}
            original = cls["module"]
            f = cls["function"]
            res[f] = f
            if verbose:
                print(f"[patch_module_or_classes] function: {original.__name__}.{f.__name__}")
            setattr(original, f.__name__, cls["patch"])
            continue

        original = cls._PATCHED_CLASS_
        methods = cls._PATCHES_
        if verbose:
            print(f"[patch_module_or_classes] {name}.{cls.__name__}: {', '.join(methods)}")

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
        name, to_patch = get_patches(mod, verbose)

    set_patch_cls = {i for i in to_patch if not isinstance(i, dict)}
    dict_patch_fct = {i["function"]: i for i in to_patch if isinstance(i, dict)}

    for cls, methods in info.items():
        if cls in set_patch_cls:
            if verbose:
                print(
                    f"[unpatch_module_or_classes] {name}.{cls.__name__}: {', '.join(methods)}"
                )
            original = cls._PATCHED_CLASS_
            for n, v in methods.items():
                if v is None:
                    # The method did not exist. We remove it.
                    delattr(original, n)
                else:
                    setattr(original, n, v)
            continue
        assert cls in dict_patch_fct, (
            f"No patch registered for {cls} in {mod} "
            f"(found {set_patch_cls} and {set(dict_patch_fct)})"
        )
        patch = dict_patch_fct[cls]
        if verbose:
            print(
                f"[unpatch_module_or_classes] function "
                f"{patch['module'].__name__}.{cls.__name__}"
            )
        setattr(patch["module"], cls.__name__, patch["function"])


@contextlib.contextmanager
def register_additional_serialization_functions(
    patch_transformers: bool = False, patch_diffusers: bool = False, verbose: int = 0
) -> Callable:
    """The necessary modifications to run the fx Graph."""
    fct_callable = (
        replacement_before_exporting
        if patch_transformers or patch_diffusers
        else (lambda x: x)
    )
    done = register_cache_serialization(
        patch_transformers=patch_transformers, patch_diffusers=patch_diffusers, verbose=verbose
    )
    try:
        yield fct_callable
    finally:
        unregister_cache_serialization(done, verbose=verbose)


@contextlib.contextmanager
def torch_export_patches(
    patch_sympy: bool = True,
    patch_torch: bool = True,
    patch_transformers: bool = False,
    patch_diffusers: bool = False,
    catch_constraints: bool = True,
    stop_if_static: int = 0,
    verbose: int = 0,
    patch: bool = True,
    custom_patches: Optional[List[type["torch.nn.Module"]]] = None,  # noqa: F821
    rewrite: Optional[List[Callable]] = None,
    dump_rewriting: Optional[str] = None,
) -> Callable:
    """
    Tries to bypass some situations :func:`torch.export.export` does not support.
    See also :ref:`l-patches-explained` and :ref:`l-patch-coverage`.

    :param patch_sympy: fix missing method ``name`` for IntegerConstant
    :param patch_torch: patches :epkg:`torch` with supported implementation
    :param patch_transformers: patches :epkg:`transformers` with supported implementation
    :param patch_diffusers: patches :epkg:`diffusers` with supported implementation
    :param catch_constraints: catch constraints related to dynamic shapes,
        as a result, some dynamic dimension may turn into static ones,
        the environment variable ``SKIP_SOLVE_CONSTRAINTS=0``
        can be put to stop at that stage.
    :param stop_if_static: see example :ref:`l-plot-export-locale-issue`,
        to stop the export as soon as an issue is detected with dynamic shapes
        and show a stack trace indicating the exact location of the issue,
        ``if stop_if_static > 1``, more methods are replace to catch more
        issues
    :param patch: if False, disable all patches but keeps the registration of
        serialization functions if other patch functions are enabled
    :param custom_patches: to apply custom patches,
        every patched class must define static attributes
        ``_PATCHES_``, ``_PATCHED_CLASS_``
    :param rewrite: list of methods to automatically rewrite
        before exporting, methods with control flow need to be rewritten
        before being exported if the execution path depends on the inputs,
        this is done by function :func:`transform_method
        <onnx_diagnostic.torch_export_patches.patch_module.transform_method>`,
        its documentation provides possible values
    :param dump_rewriting: dumps rewriting information in file beginning with that prefix
    :param verbose: to show which patches is applied

    The list of available patches.

    * ``torch.jit.isinstance``
    * ``torch._dynamo.mark_static_address``
    * ``torch._subclasses.fake_impls.infer_size``
    * ``torch.vmap``
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

    .. code-block:: python

        with torch_export_patches(patch_transformers=True) as modificator:
            inputs = modificator(inputs)
            onx = to_onnx(..., inputs, ...)

    .. code-block:: python

        with torch_export_patches(patch_transformers=True) as modificator:
            inputs = modificator(inputs)
            onx = torch.onnx.export(..., inputs, ...)

    It can be used as well to fix the torch export:

    .. code-block:: python

        with torch_export_patches(patch_transformers=True) as modificator:
            inputs = modificator(inputs)
            ep = torch.export.export(..., inputs, ...)

    When running the model through the exported program, only the
    serialization functions need to be restored:

    .. code-block:: python

        with register_additional_serialization_functions() as modificator:
            inputs = modificator(inputs)
            ep = torch.export.export(..., inputs, ...)

    When exporting a model with a cache, the following error message
    may appear ``AssertionError: Mutating module attribute _seen_tokens during export.``.
    It can be avoided by setting ``strict=False`` when call :func:`torch.export.export`.
    """
    if rewrite:
        from .patch_module import torch_export_rewrite

        with torch_export_rewrite(
            rewrite=rewrite, dump_rewriting=dump_rewriting, verbose=verbose
        ), torch_export_patches(  # type: ignore[var-annotated]
            patch_sympy=patch_sympy,
            patch_torch=patch_torch,
            patch_transformers=patch_transformers,
            patch_diffusers=patch_diffusers,
            catch_constraints=catch_constraints,
            stop_if_static=stop_if_static,
            verbose=verbose,
            patch=patch,
            custom_patches=custom_patches,
        ) as f:
            try:
                yield f
            finally:
                pass
    elif not patch:
        fct_callable = lambda x: x  # noqa: E731
        done = register_cache_serialization(
            patch_transformers=patch_transformers,
            patch_diffusers=patch_diffusers,
            verbose=verbose,
        )
        try:
            yield fct_callable
        finally:
            unregister_cache_serialization(done, verbose=verbose)
    else:
        import torch
        import torch._export.non_strict_utils  # produce_guards_and_solve_constraints
        import torch.jit

        if verbose:
            print(
                "[torch_export_patches] replace torch.jit.isinstance, "
                "torch._dynamo.mark_static_address"
            )

        ########
        # caches
        ########

        cache_done = register_cache_serialization(
            patch_transformers=patch_transformers,
            patch_diffusers=patch_diffusers,
            verbose=verbose,
        )

        #############
        # patch sympy
        #############

        if patch_sympy:
            import sympy

            f_sympy_name = getattr(sympy.core.numbers.IntegerConstant, "name", None)

            if verbose:
                print(f"[torch_export_patches] sympy.__version__={sympy.__version__!r}")
                print("[torch_export_patches] patch sympy")

            sympy.core.numbers.IntegerConstant.name = lambda self: f"IntCst{str(self)}"

        ###############
        # patch pytorch
        ###############

        if patch_torch:
            from .patches.patch_torch import (
                patched_infer_size,
                patched_vmap,
                patched__broadcast_shapes,
                _catch_produce_guards_and_solve_constraints,
                patch__check_input_constraints_for_graph,
            )

            if verbose:
                print(f"[torch_export_patches] torch.__version__={torch.__version__!r}")
                print(f"[torch_export_patches] stop_if_static={stop_if_static!r}")
                print("[torch_export_patches] patch pytorch")

            # torch.vmap
            f_vmap = torch.vmap
            torch.vmap = patched_vmap

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
                print("[torch_export_patches] modifies shape constraints")
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
                print("[torch_export_patches] assert when a dynamic dimension turns static")
                print("[torch_export_patches] replaces ShapeEnv._set_replacement")

            f_shape_env__set_replacement = ShapeEnv._set_replacement
            ShapeEnv._set_replacement = patched_ShapeEnv._set_replacement

            if verbose:
                print("[torch_export_patches] replaces ShapeEnv._log_guard")
            f_shape_env__log_guard = ShapeEnv._log_guard
            ShapeEnv._log_guard = patched_ShapeEnv._log_guard

            if stop_if_static > 1:
                if verbose:
                    print("[torch_export_patches] replaces ShapeEnv._check_frozen")
                f_shape_env__check_frozen = ShapeEnv._check_frozen
                ShapeEnv._check_frozen = patched_ShapeEnv._check_frozen

        ####################
        # patch transformers
        ####################

        if patch_transformers:
            try:
                import transformers.masking_utils as masking_utils
            except ImportError:
                masking_utils = None

            if verbose:
                import transformers

                print(
                    f"[torch_export_patches] transformers.__version__="
                    f"{transformers.__version__!r}"
                )
            revert_patches_info = patch_module_or_classes(
                patch_transformers_list, verbose=verbose
            )

            if (
                masking_utils
                and patch_transformers_list.patch_masking_utils
                and hasattr(masking_utils, "_vmap_for_bhqkv")
            ):
                if verbose:
                    print(
                        "[torch_export_patches] patches "
                        "transformers.masking_utils._vmap_for_bhqkv"
                    )
                f_transformers__vmap_for_bhqkv = masking_utils._vmap_for_bhqkv
                masking_utils._vmap_for_bhqkv = patch_transformers_list.patched__vmap_for_bhqkv

            if (
                masking_utils
                and patch_transformers_list.patch_masking_utils
                and hasattr(masking_utils, "eager_mask")
            ):
                if verbose:
                    print(
                        "[torch_export_patches] patches "
                        "transformers.masking_utils.eager_mask"
                    )
                f_transformers_eager_mask = masking_utils.eager_mask
                masking_utils.eager_mask = patch_transformers_list.patched_eager_mask
                if (
                    "eager" in masking_utils.ALL_MASK_ATTENTION_FUNCTIONS
                    and masking_utils.ALL_MASK_ATTENTION_FUNCTIONS["eager"]
                    == f_transformers_eager_mask
                ):
                    masking_utils.ALL_MASK_ATTENTION_FUNCTIONS["eager"] = (
                        patch_transformers_list.patched_eager_mask
                    )

        if custom_patches:
            if verbose:
                print("[torch_export_patches] applies custom patches")
            revert_custom_patches_info = patch_module_or_classes(
                custom_patches, verbose=verbose
            )

        ########
        # export
        ########

        fct_callable = replacement_before_exporting if patch_transformers else (lambda x: x)

        if verbose:
            print("[torch_export_patches] done patching")

        try:
            yield fct_callable
        finally:
            #######
            # sympy
            #######

            if verbose:
                print("[torch_export_patches] remove patches")

            if patch_sympy:
                # tracked by https://github.com/pytorch/pytorch/issues/143494
                if f_sympy_name:
                    sympy.core.numbers.IntegerConstant.name = f_sympy_name
                else:
                    delattr(sympy.core.numbers.IntegerConstant, "name")

                if verbose:
                    print("[torch_export_patches] restored sympy functions")

            #######
            # torch
            #######

            if patch_torch:
                # this should disappear when torch.jit is removed
                torch.vmap = f_vmap
                torch.jit.isinstance = f_jit_isinstance
                torch._dynamo.mark_static_address = f_mark_static_address
                # tracked by https://github.com/pytorch/pytorch/issues/143495
                torch._subclasses.fake_impls.infer_size = f_infer_size
                torch._refs._broadcast_shapes = f__broadcast_shapes
                torch._meta_registrations._broadcast_shapes = f__broadcast_shapes

                if verbose:
                    print("[torch_export_patches] restored pytorch functions")

            if stop_if_static:
                if verbose:
                    print("[torch_export_patches] restored ShapeEnv._set_replacement")

                ShapeEnv._set_replacement = f_shape_env__set_replacement

                if verbose:
                    print("[torch_export_patches] restored ShapeEnv._log_guard")

                ShapeEnv._log_guard = f_shape_env__log_guard

                if stop_if_static > 1:
                    if verbose:
                        print("[torch_export_patches] restored ShapeEnv._check_frozen")
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
                    print("[torch_export_patches] restored shape constraints")

            if custom_patches:
                if verbose:
                    print("[torch_export_patches] unpatches custom patches")
                unpatch_module_or_classes(
                    custom_patches, revert_custom_patches_info, verbose=verbose
                )

            ##############
            # transformers
            ##############

            if patch_transformers:
                try:
                    import transformers.masking_utils as masking_utils
                except ImportError:
                    masking_utils = None
                if verbose:
                    print("[torch_export_patches] unpatches transformers")
                unpatch_module_or_classes(
                    patch_transformers_list, revert_patches_info, verbose=verbose
                )

                if (
                    masking_utils
                    and patch_transformers_list.patch_masking_utils
                    and hasattr(masking_utils, "_vmap_for_bhqkv")
                ):
                    masking_utils._vmap_for_bhqkv = f_transformers__vmap_for_bhqkv
                    if verbose:
                        print(
                            "[torch_export_patches] restored "
                            "transformers.masking_utils._vmap_for_bhqkv"
                        )

                if (
                    masking_utils
                    and patch_transformers_list.patch_masking_utils
                    and hasattr(masking_utils, "eager_mask")
                ):
                    f_transformers_eager_mask = masking_utils.eager_mask
                    masking_utils.eager_mask = f_transformers_eager_mask
                    if (
                        "eager" in masking_utils.ALL_MASK_ATTENTION_FUNCTIONS
                        and masking_utils.ALL_MASK_ATTENTION_FUNCTIONS["eager"]
                        == patch_transformers_list.patched_eager_mask
                    ):
                        masking_utils.ALL_MASK_ATTENTION_FUNCTIONS["eager"] = (
                            f_transformers_eager_mask
                        )
                    if verbose:
                        print(
                            "[torch_export_patches] restored "
                            "transformers.masking_utils.eager_mask"
                        )

            ########
            # caches
            ########

            unregister_cache_serialization(cache_done, verbose=verbose)


def replacement_before_exporting(args: Any) -> Any:
    """
    Does replacements on the given inputs if needed.
    """
    if args is None:
        return None
    if isinstance(args, (int, float)):
        return args
    if type(args) not in {dict, tuple, list}:
        # BaseModelOutput is a dict
        return args
    if isinstance(args, dict):
        return {k: replacement_before_exporting(v) for k, v in args.items()}
    if isinstance(args, tuple):
        return tuple(replacement_before_exporting(v) for v in args)
    if isinstance(args, list):
        return [replacement_before_exporting(v) for v in args]

    return args
