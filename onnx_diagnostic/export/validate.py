import inspect
import itertools
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from ..helpers import string_type, max_diff, string_diff
from ..helpers.torch_test_helper import torch_deepcopy
from .dynamic_shapes import CoupleInputsDynamicShapes


def compare_modules(
    modep: torch.nn.Module,
    mod: Optional[torch.nn.Module] = None,
    args: Optional[Tuple[Any, ...]] = None,
    kwargs: Optional[Dict[str, Any]] = None,
    copy: bool = False,
    exc: bool = True,
    verbose: int = 0,
    atol: float = 1e-2,
    rtol: float = 1e-1,
) -> Dict[str, Any]:
    """
    Compares two torch modules, usually one coming from an exported program,
    the other being the origin model.

    :param model: first module
    :param mod: second module (it produces the expected values)
    :param args: positional arguments
    :param kwargs: named arguments
    :param copy: copy the inputs before executing the model (they may modify them inplace)
    :param exc: raise exception if discrepancies are too high
    :param verbose: verbosity level
    :param atol: absolute tolerance
    :param rtol: relative tolerance
    :return: dictionary with inputs, outputs and tolerance

    Example:

    .. runpython::
        :showcode:

        import torch
        from onnx_diagnostic.export import validate_ep, CoupleInputsDynamicShapes

        class Model(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        model = Model()
        x = torch.randn((5, 6))
        y = torch.randn((1, 6))
        model(x, y)  # to make it is running

        ds = ({0: "a", 1: "b"}, {1: "b"})
        cpl = CoupleInputsDynamicShapes((x, y), {}, ds)
        ep = torch.export.export(model, (x, y), dynamic_shapes=cpl.replace_string_by())
        validate_ep(
            ep,
            model,
            args=(x, y),
            verbose=2,
            copy=True,
            dynamic_shapes=ds,
            values_to_try={"a": [5, 10], "b": [10, 20]},
        )

    """
    args = args or ()
    kwargs = kwargs or {}

    def _get(a):
        return torch_deepcopy(a) if copy else a

    if verbose:
        begin = time.perf_counter()
        print(
            f"[compare_modules] check ep with "
            f"args={string_type(args, with_shape=True, with_device=True)}, "
            f"kwargs={string_type(kwargs, with_shape=True, with_device=True)}..."
        )
    got = modep(*_get(args), **_get(kwargs))
    if verbose:
        d = time.perf_counter() - begin
        print(f"[compare_modules] done in {d} with output={string_type(got, with_shape=True)}")
    if mod:
        if verbose:
            begin = time.perf_counter()
            print("[compare_modules] run torch module...")
        expected = mod(*_get(args), **_get(kwargs))
        diff = max_diff(expected, got)
        if verbose:
            d = time.perf_counter() - begin
            print(
                f"[compare_modules] done in {d} with "
                f"output={string_type(expected, with_shape=True)}"
            )
            print(f"[compare_modules] discrepancies={string_diff(diff)}")
        assert not exc or (
            diff["abs"] <= atol and diff["rel"] <= rtol
        ), f"Discrepancies={string_diff(diff)} higher than expected."
        return dict(args=args, kwargs=kwargs, expected=expected, got=got, diff=diff)
    return dict(args=args, kwargs=kwargs, got=got)


def validate_ep(
    ep: Union[torch.nn.Module, torch.export.ExportedProgram],
    mod: Optional[torch.nn.Module] = None,
    args: Optional[Tuple[Any, ...]] = None,
    kwargs: Optional[Dict[str, Any]] = None,
    copy: bool = False,
    dynamic_shapes: Optional[Any] = None,
    values_to_try: Optional[Dict[str, List[int]]] = None,
    exc: bool = True,
    verbose: int = 0,
    atol: float = 1e-2,
    rtol: float = 1e-1,
) -> List[Dict[str, Any]]:
    """
    Validates an exported program.

    :param model: first module
    :param mod: second module (it produces the expected values)
    :param args: positional arguments
    :param kwargs: named arguments
    :param copy: copy the inputs before executing the model (they may modify them inplace)
    :param dynamic_shapes: dynamic shapes, string should be used not ``torch.export.Dim``
    :param values_to_try: dictionary with the values to try for every dynamic dimension
    :param exc: raise exception if discrepancies are too high
    :param verbose: verbosity level
    :param atol: absolute tolerance
    :param rtol: relative tolerance
    :return: dictionary with inputs, outputs and tolerance
    """
    modep = ep.module() if isinstance(ep, torch.export.ExportedProgram) else ep

    results = [
        compare_modules(
            modep, mod, args, kwargs, copy=copy, verbose=verbose, atol=atol, rtol=rtol
        )
    ]

    assert (dynamic_shapes and values_to_try) or (
        not dynamic_shapes and not values_to_try
    ), "Either both dynamic_shapes and values_to_try are specified, either none."
    if not dynamic_shapes or not values_to_try:
        return results

    items = list(values_to_try.items())
    keys = [_[0] for _ in items]
    values = [_[1] for _ in items]
    all_vals = list(itertools.product(*values))
    cpl = CoupleInputsDynamicShapes(
        args or (),
        kwargs or {},
        dynamic_shapes,
        args_names=(
            list(inspect.signature(modep.forward).parameters) if args and kwargs else None
        ),
    )
    for i, vals in enumerate(all_vals):
        change_dims = dict(zip(keys, vals))
        if verbose:
            print(f"[validate_ep] try {i}/{len(all_vals)}: {change_dims}")
        new_params = cpl.change_dynamic_dimensions(change_dims, args_kwargs=True)
        na, nkw = new_params
        c = compare_modules(
            modep, mod, na, nkw, copy=copy, verbose=max(verbose - 1, 0), atol=atol, rtol=rtol
        )
        results.append(c)
    return results
