from typing import Callable, Set
import torch
from ..helpers.torch_test_helper import is_torchdynamo_exporting


def make_undefined_dimension(i: int) -> torch.SymInt:
    """
    Uses for a custom op when a new dimension must be introduced to bypass
    some verification. The following function creates a dummy output
    with a dimension based on the content.

    .. code-block:: python

        def symbolic_shape(x, y):
            return torch.empty(
                x.shape[0],
                make_undefined_dimension(min(x.shape[1], y[0])),
            )
    """
    try:
        ti = int(i)
    except:  # noqa: E722
        ti = 10
    t = torch.ones((ti * 2,))
    t[:ti] = 0
    res = torch.nonzero(t).shape[0]
    return res


def _patched_float_arange(
    start: torch.Tensor, end: torch.Tensor, step: torch.Tensor
) -> torch.Tensor:
    """Float arange."""
    return torch.arange(
        float(start.item()),
        float(end.item()),
        float(step.item()),
        dtype=start.dtype,
        device=start.device,
    )


def _patched_float_arange_shape(start, end, step):
    # Fails because:
    # Did you accidentally call new_dynamic_size() or item()
    # more times than you needed to in your fake implementation?
    # try:
    #     n = math.ceil(((end - start) / step).item())
    # except:  # noqa: E722
    #     n = 10
    n = 10
    return torch.empty((make_undefined_dimension(n),), dtype=start.dtype, device=start.device)


def _iterate_patched_expressions():
    glo = globals().copy()
    for k, _v in glo.items():
        if k.startswith("_patched_") and not k.endswith("_shape"):
            name = k
            yield k[len("_patched_") :], glo[name], glo[f"{name}_shape"]


_registered: Set[str] = set()


def _register_patched_expression(
    fct: Callable, fct_shape: Callable, namespace: str, fname: str
):
    schema_str = torch.library.infer_schema(fct, mutates_args=())
    custom_def = torch.library.CustomOpDef(namespace, fname, schema_str, fct)
    custom_def.register_kernel("cpu")(fct)
    custom_def._abstract_fn = fct_shape


def register_patched_expressions(namespace: str = "patched"):
    """
    Registers as custom ops known expressions failing due to dynamic shapes.

    .. runpython::
        :showcode:

        import pprint
        from onnx_diagnostic.torch_export_patches.patch_expressions import (
            _iterate_patched_expressions,
        )

        pprint.pprint([name for name, _f, _fsh in _iterate_patched_expressions()])
    """
    for name, f, fsh in _iterate_patched_expressions():
        if name not in _registered:
            _register_patched_expression(f, fsh, namespace, name)
            _registered.add(name)


def patched_selector(fct: Callable, patched_fct: Callable) -> Callable:
    """
    Returns **fct** if the model is being executed or
    **patched_fct** if it is being exported.
    """
    return patched_fct if is_torchdynamo_exporting() else fct


def patched_float_arange(start, end, step):
    """Patched arange when start, end, step are floats."""
    if is_torchdynamo_exporting():
        return torch.ops.patched.float_arange(start, end, step)
    else:
        return torch.arange(start, end, step)
