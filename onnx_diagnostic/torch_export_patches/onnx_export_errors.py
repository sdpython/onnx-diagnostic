import functools
import importlib
import inspect
import contextlib
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from .onnx_export_serialization import (
    register_cache_serialization,
    unregister_cache_serialization,
)
from .patches import patch_transformers as patch_transformers_list
from .patch_details import PatchDetails


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
            elif v.__doc__:
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


def patch_module_or_classes(
    mod, verbose: int = 0, patch_details: Optional[PatchDetails] = None
) -> Dict[type, Dict[type, Callable]]:
    """
    Applies all patches defined in classes prefixed by ``patched_``
    ``cls._PATCHED_CLASS_`` defines the class to patch,
    ``cls._PATCHES_`` defines the method to patch.
    The returns information needs to be sent to :func:`unpatch_module_or_classes`
    to revert the changes.

    :param mod: module of list of clsses to patch
    :param verbose: verbosity
    :param patch_details: used to store information about the applied patches
    :return: patch info
    """
    if isinstance(mod, list):
        to_patch = mod
        name = "list"
        list_name = "auto/list"
    else:
        name, to_patch = get_patches(mod, verbose)
        list_name = f"auto/{mod.__name__.split('.')[-1]}"

    res = {}
    for cls in to_patch:
        if isinstance(cls, dict):
            # a function
            keep = {}
            original = cls["module"]
            f = cls["function"]
            assert not f.__name__.startswith("patched_"), (
                f"The function {f} was already patched or the patch was not removed, "
                f"original={original}"
            )
            res[f] = f
            if verbose:
                print(f"[patch_module_or_classes] function: {original.__name__}.{f.__name__}")
            if patch_details:
                patch_details.append(list_name, getattr(original, f.__name__), cls["patch"])
            setattr(original, f.__name__, cls["patch"])
            continue

        original = cls._PATCHED_CLASS_
        methods = [_ for _ in cls._PATCHES_ if _ is not None]
        if verbose:
            print(f"[patch_module_or_classes] {name}.{cls.__name__}: {', '.join(methods)}")

        keep = {n: getattr(original, n, None) for n in methods}
        for n in methods:
            if patch_details:
                if hasattr(original, n):
                    p = patch_details.append(list_name, getattr(original, n), getattr(cls, n))
                else:
                    p = patch_details.append(
                        list_name, f"{original.__name__}{n}", getattr(cls, n)
                    )
                if "@patched_dynamic_rope_update" in inspect.getsource(getattr(cls, n)):
                    # a tweak to include that patch.
                    f = patch_details.find("patched_dynamic_rope_update")
                    if f is not None:
                        p.add_dependency(f)
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


def _patch_sympy(verbose: int, patch_details: PatchDetails) -> Tuple[Optional[Callable], ...]:
    import sympy

    f_sympy_name = getattr(sympy.core.numbers.IntegerConstant, "name", None)

    if verbose:
        print(f"[torch_export_patches] sympy.__version__={sympy.__version__!r}")
        print("[torch_export_patches] patch sympy")

    sympy.core.numbers.IntegerConstant.name = lambda self: f"IntCst{str(self)}"
    if patch_details:
        patch_details.append(
            "sympy",
            f_sympy_name or "sympy.core.numbers.IntegerConstant.name",
            sympy.core.numbers.IntegerConstant.name,
        )
    return (f_sympy_name,)


def _unpatch_sympy(verbose: int, f_sympy_name: Optional[Callable]):
    # tracked by https://github.com/pytorch/pytorch/issues/143494
    import sympy

    if f_sympy_name:
        sympy.core.numbers.IntegerConstant.name = f_sympy_name
    else:
        delattr(sympy.core.numbers.IntegerConstant, "name")

    if verbose:
        print("[torch_export_patches] restored sympy functions")


def _patch_torch(
    verbose: int,
    patch_details: PatchDetails,
    patch_torch: int,
    catch_constraints: bool,
    stop_if_static: int,
) -> Tuple[Optional[Callable], ...]:
    import packaging.version as pv
    import torch
    import torch.jit
    import torch._export.non_strict_utils  # produce_guards_and_solve_constraints
    from torch.fx.experimental.symbolic_shapes import ShapeEnv
    from .patches.patch_torch import (
        patched_infer_size,
        patched_vmap,
        patched__broadcast_shapes,
        patched__constrain_user_specified_dimhint_range,
        _catch_produce_guards_and_solve_constraints,
        patch__check_input_constraints_for_graph,
        patched__broadcast_in_dim_meta,
        patched__broadcast_in_dim_meta_level_2,
        patched__maybe_broadcast,
        patched_ShapeEnv,
    )

    if pv.Version(torch.__version__) >= pv.Version("2.9.99"):
        from .patches.patch_torch import patched_DynamicDimConstraintPrinter
    else:
        patched_DynamicDimConstraintPrinter = None

    f___constrain_user_specified_dimhint_range = None
    f__broadcast_in_dim_meta = None
    f__broadcast_shapes = None
    f__check_input_constraints_for_graph = None
    f__maybe_broadcast = None
    f_broadcast_in_dim = None
    f_infer_size = None
    f_jit_isinstance = None
    f_mark_static_address = None
    f_produce_guards_and_solve_constraints = None
    f_shape_env__check_frozen = None
    f_shape_env__evaluate_expr = None
    f_shape_env__log_guard = None
    f_shape_env__set_replacement = None
    f_vmap = None

    if verbose:
        print(f"[torch_export_patches] torch.__version__={torch.__version__!r}")
        print(f"[torch_export_patches] stop_if_static={stop_if_static!r}")
        print("[torch_export_patches] patch pytorch")

    # torch.tx.experimental.symbolic_shapes.DynamicDimConstraintPrinter._print_Symbol
    if patched_DynamicDimConstraintPrinter is not None:
        f__print_symbol = (
            torch.fx.experimental.symbolic_shapes.DynamicDimConstraintPrinter._print_Symbol
        )
        torch.fx.experimental.symbolic_shapes.DynamicDimConstraintPrinter._print_Symbol = (
            patched_DynamicDimConstraintPrinter._print_Symbol
        )
    else:
        f__print_symbol = None

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
    if patch_details:
        patch_details.append("torch", f_infer_size, patched_infer_size)

    # torch._refs._broadcast_shapes
    f__broadcast_shapes = torch._refs._broadcast_shapes
    torch._refs._broadcast_shapes = patched__broadcast_shapes
    torch._meta_registrations._broadcast_shapes = patched__broadcast_shapes
    if patch_details:
        patch_details.append("torch", f__broadcast_shapes, patched__broadcast_shapes)

    # torch._export.non_strict_utils._constrain_user_specified_dimhint_range
    f___constrain_user_specified_dimhint_range = (
        torch._export.non_strict_utils._constrain_user_specified_dimhint_range
    )
    torch._export.non_strict_utils._constrain_user_specified_dimhint_range = (
        patched__constrain_user_specified_dimhint_range
    )
    if patch_details:
        patch_details.append(
            "torch",
            f___constrain_user_specified_dimhint_range,
            patched__constrain_user_specified_dimhint_range,
        )

    # torch._prims._broadcast_in_dim_meta
    f_broadcast_in_dim = torch._prims.broadcast_in_dim
    f__broadcast_in_dim_meta = torch._prims._broadcast_in_dim_meta
    _patched_dim_f = (
        patched__broadcast_in_dim_meta_level_2
        if patch_torch == 2
        else patched__broadcast_in_dim_meta
    )
    torch._prims._broadcast_in_dim_meta = _patched_dim_f
    torch._prims.broadcast_in_dim = _patched_dim_f
    if patch_details:
        patch_details.append("torch", f__broadcast_in_dim_meta, _patched_dim_f)

    # torch._refs._maybe_broadcast
    f__maybe_broadcast = torch._refs._maybe_broadcast
    torch._refs._maybe_broadcast = patched__maybe_broadcast
    if patch_details:
        patch_details.append("torch", f__maybe_broadcast, patched__maybe_broadcast)

    # ShapeEnv
    f_shape_env__evaluate_expr = ShapeEnv._evaluate_expr
    ShapeEnv._evaluate_expr = patched_ShapeEnv._evaluate_expr
    if patch_details:
        patch_details.append(
            "torch", f_shape_env__evaluate_expr, patched_ShapeEnv._evaluate_expr
        )

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

    if patch_torch and stop_if_static:
        ShapeEnv._log_guard_remember = ShapeEnv._log_guard

        if verbose:
            print("[torch_export_patches] assert when a dynamic dimension turns static")
            print("[torch_export_patches] replaces ShapeEnv._set_replacement")

        f_shape_env__set_replacement = ShapeEnv._set_replacement
        ShapeEnv._set_replacement = patched_ShapeEnv._set_replacement
        if patch_details:
            patch_details.append(
                "torch", f_shape_env__set_replacement, patched_ShapeEnv._set_replacement
            )

        if verbose:
            print("[torch_export_patches] replaces ShapeEnv._log_guard")
        f_shape_env__log_guard = ShapeEnv._log_guard
        ShapeEnv._log_guard = patched_ShapeEnv._log_guard
        if patch_details:
            patch_details.append("torch", f_shape_env__log_guard, patched_ShapeEnv._log_guard)

        if stop_if_static > 1:
            if verbose:
                print("[torch_export_patches] replaces ShapeEnv._check_frozen")
            f_shape_env__check_frozen = ShapeEnv._check_frozen
            ShapeEnv._check_frozen = patched_ShapeEnv._check_frozen
            if patch_details:
                patch_details.append(
                    "torch", f_shape_env__check_frozen, ShapeEnv._check_frozen
                )
    return (
        f___constrain_user_specified_dimhint_range,
        f__broadcast_in_dim_meta,
        f__broadcast_shapes,
        f__check_input_constraints_for_graph,
        f__maybe_broadcast,
        f_broadcast_in_dim,
        f_infer_size,
        f_jit_isinstance,
        f_mark_static_address,
        f_produce_guards_and_solve_constraints,
        f_shape_env__check_frozen,
        f_shape_env__evaluate_expr,
        f_shape_env__log_guard,
        f_shape_env__set_replacement,
        f_vmap,
        f__print_symbol,
    )


def _unpatch_torch(
    verbose: int,
    _patch_details: PatchDetails,
    patch_torch: int,
    catch_constraints: bool,
    stop_if_static: int,
    f___constrain_user_specified_dimhint_range: Optional[Callable],
    f__broadcast_in_dim_meta: Optional[Callable],
    f__broadcast_shapes: Optional[Callable],
    f__check_input_constraints_for_graph: Optional[Callable],
    f__maybe_broadcast: Optional[Callable],
    f_broadcast_in_dim: Optional[Callable],
    f_infer_size: Optional[Callable],
    f_jit_isinstance: Optional[Callable],
    f_mark_static_address: Optional[Callable],
    f_produce_guards_and_solve_constraints: Optional[Callable],
    f_shape_env__check_frozen: Optional[Callable],
    f_shape_env__evaluate_expr: Optional[Callable],
    f_shape_env__log_guard: Optional[Callable],
    f_shape_env__set_replacement: Optional[Callable],
    f_vmap: Optional[Callable],
    f__print_symbol: Optional[Callable],
):
    import torch
    import torch.jit
    import torch._export.non_strict_utils  # produce_guards_and_solve_constraints
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    # this should disappear when torch.jit is removed
    if f__print_symbol is not None:
        torch.fx.experimental.symbolic_shapes.DynamicDimConstraintPrinter._print_Symbol = (
            f__print_symbol
        )
    torch.vmap = f_vmap
    torch.jit.isinstance = f_jit_isinstance
    torch._dynamo.mark_static_address = f_mark_static_address
    # tracked by https://github.com/pytorch/pytorch/issues/143495
    torch._subclasses.fake_impls.infer_size = f_infer_size
    torch._refs._broadcast_shapes = f__broadcast_shapes
    torch._meta_registrations._broadcast_shapes = f__broadcast_shapes
    torch._export.non_strict_utils._constrain_user_specified_dimhint_range = (
        f___constrain_user_specified_dimhint_range
    )
    torch._prims._broadcast_in_dim_meta = f__broadcast_in_dim_meta
    torch._prims.broadcast_in_dim = f_broadcast_in_dim
    torch._refs._maybe_broadcast = f__maybe_broadcast
    ShapeEnv._evaluate_expr = f_shape_env__evaluate_expr

    if verbose:
        print("[torch_export_patches] restored pytorch functions")

    if patch_torch and stop_if_static:
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

    if patch_torch and catch_constraints:
        # to catch or skip dynamic_shapes issues
        torch._export.non_strict_utils.produce_guards_and_solve_constraints = (
            f_produce_guards_and_solve_constraints
        )
        torch._export.utils._check_input_constraints_for_graph = (
            f__check_input_constraints_for_graph
        )
        if verbose:
            print("[torch_export_patches] restored shape constraints")


def _patch_transformers(
    verbose: int, patch_details: PatchDetails
) -> Tuple[Optional[Callable], ...]:
    import transformers

    try:
        import transformers.masking_utils as masking_utils
    except ImportError:
        masking_utils = None

    try:
        import transformers.integrations.sdpa_attention as sdpa_attention
    except ImportError:
        sdpa_attention = None

    try:
        import transformers.modeling_utils as modeling_utils
    except ImportError:
        modeling_utils = None

    try:
        import transformers.modeling_rope_utils as modeling_rope_utils
    except ImportError:
        modeling_rope_utils = None

    if (
        patch_details
        and modeling_rope_utils
        and hasattr(modeling_rope_utils, "dynamic_rope_update")
    ):
        patch_details.append(
            "patch_transformers",
            modeling_rope_utils.dynamic_rope_update,
            patch_transformers_list.patched_dynamic_rope_update,
        )

    if verbose:
        print(f"[torch_export_patches] transformers.__version__={transformers.__version__!r}")
    assert not sdpa_attention.sdpa_attention_forward.__name__.startswith("patched_"), (
        f"Function 'sdpa_attention.sdpa_attention_forward' is already patched, "
        f"sdpa_attention.sdpa_attention_forward={sdpa_attention.sdpa_attention_forward}"
    )

    f_transformers__vmap_for_bhqkv = None
    f_transformers_eager_mask = None
    f_transformers_sdpa_attention_forward = None
    f_transformers_sdpa_mask = None
    f_transformers_sdpa_mask_recent_torch = None

    if (  # vmap
        masking_utils
        and patch_transformers_list.patch_masking_utils
        and hasattr(masking_utils, "_vmap_for_bhqkv")
    ):
        if verbose:
            print("[torch_export_patches] patches transformers.masking_utils._vmap_for_bhqkv")
        f_transformers__vmap_for_bhqkv = masking_utils._vmap_for_bhqkv
        masking_utils._vmap_for_bhqkv = patch_transformers_list.patched__vmap_for_bhqkv
        if patch_details:
            patch_details.append(
                "transformers",
                f_transformers__vmap_for_bhqkv,
                patch_transformers_list.patched__vmap_for_bhqkv,
            )

        if verbose:
            print(
                "[torch_export_patches] patches "
                "transformers.masking_utils.sdpa_mask_recent_torch"
            )
        f_transformers_sdpa_mask_recent_torch = masking_utils.sdpa_mask_recent_torch
        masking_utils.sdpa_mask_recent_torch = (
            patch_transformers_list.patched_sdpa_mask_recent_torch
        )
        if patch_details:
            patch_details.append(
                "transformers",
                f_transformers_sdpa_mask_recent_torch,
                patch_transformers_list.patched_sdpa_mask_recent_torch,
            )
        if masking_utils.sdpa_mask == f_transformers_sdpa_mask_recent_torch:
            if verbose:
                print("[torch_export_patches] patches transformers.masking_utils.sdpa_mask")
            f_transformers_sdpa_mask = masking_utils.sdpa_mask
            masking_utils.sdpa_mask = patch_transformers_list.patched_sdpa_mask_recent_torch
            if patch_details:
                patch_details.append(
                    "transformers",
                    f_transformers_sdpa_mask,
                    patch_transformers_list.patched_sdpa_mask_recent_torch,
                )
        else:
            f_transformers_sdpa_mask = None

    if (  # eager_mask
        masking_utils
        and patch_transformers_list.patch_masking_utils
        and hasattr(masking_utils, "eager_mask")
    ):
        if verbose:
            print("[torch_export_patches] patches transformers.masking_utils.eager_mask")
        f_transformers_eager_mask = masking_utils.eager_mask
        masking_utils.eager_mask = patch_transformers_list.patched_eager_mask
        if patch_details:
            patch_details.append(
                "transformers",
                f_transformers_eager_mask,
                patch_transformers_list.patched_eager_mask,
            )
        if (
            "eager" in masking_utils.ALL_MASK_ATTENTION_FUNCTIONS
            and masking_utils.ALL_MASK_ATTENTION_FUNCTIONS["eager"]
            == f_transformers_eager_mask
        ):
            if verbose:
                print(
                    "[torch_export_patches] patches "
                    "transformers.masking_utils.eager_mask "
                    "in ALL_MASK_ATTENTION_FUNCTIONS"
                )
            masking_utils.ALL_MASK_ATTENTION_FUNCTIONS["eager"] = (
                patch_transformers_list.patched_eager_mask
            )

    if (  # sdpa_mask
        masking_utils
        and patch_transformers_list.patch_masking_utils
        and hasattr(masking_utils, "sdpa_mask")
        and f_transformers_sdpa_mask is not None
    ):
        if verbose:
            print(
                "[torch_export_patches] patches "
                "transformers.masking_utils.sdpa_mask "
                "in ALL_MASK_ATTENTION_FUNCTIONS"
            )
        if (
            "sdpa" in masking_utils.ALL_MASK_ATTENTION_FUNCTIONS
            and masking_utils.ALL_MASK_ATTENTION_FUNCTIONS["sdpa"] == f_transformers_sdpa_mask
        ):
            masking_utils.ALL_MASK_ATTENTION_FUNCTIONS["sdpa"] = (
                patch_transformers_list.patched_sdpa_mask_recent_torch
            )

    if (  # sdpa_attention_forward
        sdpa_attention is not None
        and modeling_utils is not None
        and hasattr(sdpa_attention, "sdpa_attention_forward")
        and hasattr(sdpa_attention, "use_gqa_in_sdpa")
        and hasattr(modeling_utils, "AttentionInterface")
    ):
        if verbose:
            print(
                "[torch_export_patches] patches "
                "transformers.integrations.sdpa_attention.sdpa_attention_forward"
            )
        f_transformers_sdpa_attention_forward = sdpa_attention.sdpa_attention_forward
        assert not f_transformers_sdpa_attention_forward.__name__.startswith("patched_"), (
            f"Function 'sdpa_attention.sdpa_attention_forward' is already patched, "
            f"sdpa_attention.sdpa_attention_forward={f_transformers_sdpa_attention_forward}"
        )
        sdpa_attention.sdpa_attention_forward = (
            patch_transformers_list.patched_sdpa_attention_forward
        )
        modeling_utils.sdpa_attention_forward = (
            patch_transformers_list.patched_sdpa_attention_forward
        )
        modeling_utils.AttentionInterface._global_mapping["sdpa"] = (
            patch_transformers_list.patched_sdpa_attention_forward
        )
        if patch_details:
            patch_details.append(
                "transformers",
                f_transformers_sdpa_attention_forward,
                patch_transformers_list.patched_sdpa_attention_forward,
            )

    revert_patches_info = patch_module_or_classes(
        patch_transformers_list, verbose=verbose, patch_details=patch_details
    )

    return (
        f_transformers__vmap_for_bhqkv,
        f_transformers_eager_mask,
        f_transformers_sdpa_attention_forward,
        f_transformers_sdpa_mask,
        f_transformers_sdpa_mask_recent_torch,
        revert_patches_info,
    )


def _unpatch_transformers(
    verbose: int,
    _patch_details: PatchDetails,
    f_transformers__vmap_for_bhqkv: Optional[Callable],
    f_transformers_eager_mask: Optional[Callable],
    f_transformers_sdpa_attention_forward: Optional[Callable],
    f_transformers_sdpa_mask: Optional[Callable],
    f_transformers_sdpa_mask_recent_torch: Optional[Callable],
    revert_patches_info: Optional[Callable],
):

    try:
        import transformers.masking_utils as masking_utils
    except ImportError:
        masking_utils = None

    try:
        import transformers.integrations.sdpa_attention as sdpa_attention
    except ImportError:
        sdpa_attention = None

    try:
        import transformers.modeling_utils as modeling_utils
    except ImportError:
        modeling_utils = None

    try:
        import transformers.masking_utils as masking_utils
    except ImportError:
        masking_utils = None
    if verbose:
        print("[torch_export_patches] unpatches transformers")

    if (  # vmap
        masking_utils
        and patch_transformers_list.patch_masking_utils
        and hasattr(masking_utils, "_vmap_for_bhqkv")
    ):
        assert f_transformers__vmap_for_bhqkv.__name__ == "_vmap_for_bhqkv", (
            f"corrupted function '_vmap_for_bhqkv', its name is "
            f"{f_transformers__vmap_for_bhqkv.__name__!r}"
        )
        masking_utils._vmap_for_bhqkv = f_transformers__vmap_for_bhqkv

        if verbose:
            print("[torch_export_patches] restored transformers.masking_utils._vmap_for_bhqkv")

        assert f_transformers_sdpa_mask_recent_torch.__name__ == "sdpa_mask_recent_torch", (
            f"corrupted function 'sdpa_mask_recent_torch', its name is "
            f"{f_transformers_sdpa_mask_recent_torch.__name__!r}"
        )
        masking_utils.sdpa_mask_recent_torch = f_transformers_sdpa_mask_recent_torch

        if verbose:
            print(
                "[torch_export_patches] restored "
                "transformers.masking_utils.sdpa_mask_recent_torch"
            )

        if f_transformers_sdpa_mask is not None:
            assert f_transformers_sdpa_mask.__name__ in (
                "sdpa_mask",
                "sdpa_mask_recent_torch",
            ), (
                f"corrupted function 'sdpa_mask', its name is "
                f"{f_transformers_sdpa_mask.__name__!r}"
            )
            masking_utils.sdpa_mask = f_transformers_sdpa_mask
            if verbose:
                print("[torch_export_patches] restored transformers.masking_utils.sdpa_mask")

    if (  # eager_mask
        masking_utils
        and patch_transformers_list.patch_masking_utils
        and hasattr(masking_utils, "eager_mask")
    ):
        assert f_transformers_eager_mask.__name__ == "eager_mask", (
            f"corrupted function 'eager_mask', its name is "
            f"{f_transformers_eager_mask.__name__!r}"
        )
        masking_utils.eager_mask = f_transformers_eager_mask
        if verbose:
            print("[torch_export_patches] restored transformers.masking_utils.eager_mask")
        assert masking_utils.eager_mask.__name__ == "eager_mask", (
            f"corrupted function 'eager_mask', its name is "
            f"{masking_utils.eager_mask.__name__!r}"
        )
        if (
            "eager" in masking_utils.ALL_MASK_ATTENTION_FUNCTIONS
            and masking_utils.ALL_MASK_ATTENTION_FUNCTIONS["eager"]
            == patch_transformers_list.patched_eager_mask
        ):
            masking_utils.ALL_MASK_ATTENTION_FUNCTIONS["eager"] = f_transformers_eager_mask
            if verbose:
                print(
                    "[torch_export_patches] restored "
                    "transformers.masking_utils.eager_mask "
                    "in ALL_MASK_ATTENTION_FUNCTIONS"
                )
        assert masking_utils.eager_mask.__name__ == "eager_mask", (
            f"corrupted function 'eager_mask', its name is "
            f"{masking_utils.eager_mask.__name__!r}"
        )

    if (  # sdpa_mask
        masking_utils
        and patch_transformers_list.patch_masking_utils
        and hasattr(masking_utils, "sdpa_mask")
    ):
        if (
            "sdpa" in masking_utils.ALL_MASK_ATTENTION_FUNCTIONS
            and masking_utils.ALL_MASK_ATTENTION_FUNCTIONS["sdpa"]
            == patch_transformers_list.patched_sdpa_mask_recent_torch
        ):
            masking_utils.ALL_MASK_ATTENTION_FUNCTIONS["sdpa"] = f_transformers_sdpa_mask
            if verbose:
                print(
                    "[torch_export_patches] restored "
                    "transformers.masking_utils.sdpa_mask "
                    "in ALL_MASK_ATTENTION_FUNCTIONS"
                )

    if (  # sdpa_attention_forward
        sdpa_attention is not None
        and modeling_utils is not None
        and hasattr(sdpa_attention, "sdpa_attention_forward")
        and hasattr(sdpa_attention, "use_gqa_in_sdpa")
        and hasattr(modeling_utils, "AttentionInterface")
    ):
        sdpa_attention.sdpa_attention_forward = f_transformers_sdpa_attention_forward
        modeling_utils.sdpa_attention_forward = f_transformers_sdpa_attention_forward
        modeling_utils.AttentionInterface._global_mapping["sdpa"] = (
            f_transformers_sdpa_attention_forward
        )
        if verbose:
            print(
                "[torch_export_patches] restored "
                "transformers.integrations.sdpa_attention."
                "sdpa_attention_forward"
            )

    unpatch_module_or_classes(patch_transformers_list, revert_patches_info, verbose=verbose)


@contextlib.contextmanager
def torch_export_patches(
    patch_sympy: bool = True,
    patch_torch: Union[bool, int] = True,
    patch_transformers: bool = False,
    patch_diffusers: bool = False,
    catch_constraints: bool = True,
    stop_if_static: int = 0,
    verbose: int = 0,
    patch: bool = True,
    custom_patches: Optional[List[type["torch.nn.Module"]]] = None,  # noqa: F821
    rewrite: Optional[List[Callable]] = None,
    dump_rewriting: Optional[str] = None,
    patch_details: Optional[PatchDetails] = None,
    profile: Optional[str] = None,
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
    :param dump_rewriting: dumps rewriting information in file beginning with that prefix,
        this only applied on the automated rewritings
    :param patch_details: if specified, this class is used to stored every applied rewriting.
    :param verbose: to show which patches is applied
    :param profile: starts profiling whatever is called inside the context manager,
        output the profiling into a text file

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
    if verbose:
        print(f"[torch_export_patches] patch_sympy={patch_sympy!r}")
        print(f"                     . patch_torch={patch_torch!r}")
        print(f"                     . patch_transformers={patch_transformers!r}")
        print(f"                     . patch_diffusers={patch_diffusers!r}")
        print(f"                     . catch_constraints={catch_constraints!r}")
        print(f"                     . stop_if_static={stop_if_static!r}")
        print(f"                     . patch={patch!r}")
        print(f"                     . custom_patches={custom_patches!r}")
        print(f"[torch_export_patches] dump_rewriting={dump_rewriting!r}")

    if rewrite:
        from .patch_module import torch_export_rewrite

        with (
            torch_export_rewrite(
                rewrite=rewrite,
                dump_rewriting=dump_rewriting,
                verbose=verbose,
                patch_details=patch_details,
            ),
            torch_export_patches(  # type: ignore[var-annotated]
                patch_sympy=patch_sympy,
                patch_torch=patch_torch,
                patch_transformers=patch_transformers,
                patch_diffusers=patch_diffusers,
                catch_constraints=catch_constraints,
                stop_if_static=stop_if_static,
                verbose=verbose,
                patch=patch,
                custom_patches=custom_patches,
                patch_details=patch_details,
            ) as f,
        ):
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
        if verbose:
            print(
                "[torch_export_patches] replace torch.jit.isinstance, "
                "torch._dynamo.mark_static_address"
            )

        # caches

        cache_done = register_cache_serialization(
            patch_transformers=patch_transformers,
            patch_diffusers=patch_diffusers,
            verbose=verbose,
        )

        # patches

        if patch_sympy:
            (f_sympy_name,) = _patch_sympy(verbose, patch_details)

        if patch_torch:
            (
                f___constrain_user_specified_dimhint_range,
                f__broadcast_in_dim_meta,
                f__broadcast_shapes,
                f__check_input_constraints_for_graph,
                f__maybe_broadcast,
                f_broadcast_in_dim,
                f_infer_size,
                f_jit_isinstance,
                f_mark_static_address,
                f_produce_guards_and_solve_constraints,
                f_shape_env__check_frozen,
                f_shape_env__evaluate_expr,
                f_shape_env__log_guard,
                f_shape_env__set_replacement,
                f_vmap,
                f__print_Symbol,
            ) = _patch_torch(
                verbose, patch_details, patch_torch, catch_constraints, stop_if_static
            )

        if patch_transformers:
            (
                f_transformers__vmap_for_bhqkv,
                f_transformers_eager_mask,
                f_transformers_sdpa_attention_forward,
                f_transformers_sdpa_mask,
                f_transformers_sdpa_mask_recent_torch,
                revert_patches_info,
            ) = _patch_transformers(verbose, patch_details)

        if custom_patches:
            if verbose:
                print("[torch_export_patches] applies custom patches")
            revert_custom_patches_info = patch_module_or_classes(
                custom_patches, verbose=verbose, patch_details=patch_details
            )

        # export

        fct_callable = replacement_before_exporting if patch_transformers else (lambda x: x)

        if verbose:
            print("[torch_export_patches] done patching")

        if profile:
            from pyinstrument import Profiler

            profiler = Profiler()
            profiler.start()
        else:
            profiler = None

        try:
            yield fct_callable
        finally:

            if profiler:
                profiler.stop()
                with open(profile, "w") as f:
                    f.write(profiler.output_html())

            # unpatch

            if verbose:
                print("[torch_export_patches] remove patches")

            if patch_sympy:
                _unpatch_sympy(verbose, f_sympy_name)

            if patch_torch:
                _unpatch_torch(
                    verbose,
                    patch_details,
                    patch_torch,
                    catch_constraints,
                    stop_if_static,
                    f___constrain_user_specified_dimhint_range,
                    f__broadcast_in_dim_meta,
                    f__broadcast_shapes,
                    f__check_input_constraints_for_graph,
                    f__maybe_broadcast,
                    f_broadcast_in_dim,
                    f_infer_size,
                    f_jit_isinstance,
                    f_mark_static_address,
                    f_produce_guards_and_solve_constraints,
                    f_shape_env__check_frozen,
                    f_shape_env__evaluate_expr,
                    f_shape_env__log_guard,
                    f_shape_env__set_replacement,
                    f_vmap,
                    f__print_Symbol,
                )

            if patch_transformers:
                _unpatch_transformers(
                    verbose,
                    patch_details,
                    f_transformers__vmap_for_bhqkv,
                    f_transformers_eager_mask,
                    f_transformers_sdpa_attention_forward,
                    f_transformers_sdpa_mask,
                    f_transformers_sdpa_mask_recent_torch,
                    revert_patches_info,
                )

            if custom_patches:
                if verbose:
                    print("[torch_export_patches] unpatches custom patches")
                unpatch_module_or_classes(
                    custom_patches, revert_custom_patches_info, verbose=verbose
                )

            ########
            # caches
            ########

            unregister_cache_serialization(cache_done, verbose=verbose)


def replacement_before_exporting(args: Any) -> Any:
    """Does replacements on the given inputs if needed."""
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
