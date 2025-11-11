import contextlib
import inspect
from typing import Callable, Optional, Tuple, Union
from ..helpers import string_type
import torch

_TEST_EXPORT = False


@contextlib.contextmanager
def enable_code_export_control_flow():
    """Enables the code means to be exported."""
    global _TEST_EXPORT
    old = _TEST_EXPORT
    _TEST_EXPORT = True
    try:
        yield
    finally:
        _TEST_EXPORT = old


def is_exporting():
    """
    Returns :func:`torch.compiler.is_exporting` or
    :func:`torch.compiler.is_compiling`
    """
    return _TEST_EXPORT or torch.compiler.is_exporting() or torch.compiler.is_compiling()


class CustomOpSchema:
    def __init__(self, name: str, model: torch.nn.Module, method_name: str = "forward"):
        self.name = name
        self.model = model
        self.method_name = method_name
        self.forward = getattr(model, method_name)

        self.inputs = []
        self.outputs = []
        self.signature = inspect.signature(self.forward)
        self.forward_parameter_names = set(
            p.name
            for p in self.signature.parameters.values()
            if p.kind not in {p.VAR_POSITIONAL, p.VAR_KEYWORD}
        )
        self.forward_ordered_parameter_names = list(self.signature.parameters)
        self.forward_positioned_parameter_names = [
            p.name
            for p in self.signature.parameters.values()
            if p.kind in (p.VAR_POSITIONAL, p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        names = [
            p.name for p in self.signature.parameters.values() if p.kind == p.VAR_POSITIONAL
        ]
        self.forward_args = names[0] if names else None
        names = [p.name for p in self.signature.parameters.values() if p.kind == p.VAR_KEYWORD]
        self.forward_kwargs = names[0] if names else None
        self.forward_custom_op_schema = None
        self.forward_need_serialization = False
        self.forward_fill_kwargs = bool(self.forward_kwargs)
        assert not isinstance(
            model, (torch.nn.ModuleList, torch.nn.ModuleDict)
        ), f"ModuleList or ModuleDict should not be traced: {type(model)}"

    def _annotation_from_type(self, obj) -> str:
        if isinstance(obj, torch.Tensor):
            return "Tensor"
        if isinstance(obj, (tuple, list)) and all(isinstance(t, torch.Tensor) for t in obj):
            return ["Tensor" for _t in obj]
        if isinstance(obj, dict) and all(isinstance(t, torch.Tensor) for t in obj.values()):
            return ["Tensor" for _t in obj]
        if obj is None:
            # Let's assume it is a tensor. It should not matter anyway.
            # Unless it becomes None in another call.
            return "Tensor?"
        if isinstance(obj, bool):
            return "bool"
        if isinstance(obj, float):
            return "float"
        if isinstance(obj, int):
            return "int"

        # Let's use torch to flatten the list.
        flat, _spec = torch.utils._pytree.tree_flatten(obj)
        return ["Tensor" for _ in flat]

    def _annotated_input(self, name):
        args, kwargs = self.inputs[0]
        if name in kwargs:
            o = kwargs[name]
            annotated = self._annotation_from_type(o)
        else:
            index = self.forward_ordered_parameter_names.index(name)
            assert index < len(args), (
                f"{self.full_name}: unexpected index={index} for name={name!r}, "
                f"forward_ordered_parameter_names={self.forward_ordered_parameter_names}, "
                f"args={string_type(args, with_shape=True)}"
            )
            o = args[index]
            annotated = self._annotation_from_type(o)
        if isinstance(annotated, str):
            return f"{annotated} {name}"
        assert isinstance(
            annotated, list
        ), f"unexpected type {type(annotated)} for name={name!r}"
        return ", ".join(
            [f"{t} {name}_n{len(annotated)}_{i}" for i, t in enumerate(annotated)]
        )

    def _annotated_output(self):
        outputs = []
        for o in self.outputs[0]:
            annotated = self._annotation_from_type(o)
            if isinstance(annotated, str):
                outputs.append(annotated)
                continue
            assert isinstance(
                annotated, list
            ), f"unexpected type {type(annotated)} for name={o!r}"
            outputs.extend(annotated)
        unique = set(outputs)
        assert unique == {
            "Tensor"
        }, f"{self.full_name}: no other tyoe than Tensor is supported, types={unique}"
        return "Tensor" if len(outputs) == 1 else "Tensor[]"

    def build_c_schema(self, verbose: int = 0) -> str:
        """Returns a schema for the C function."""
        # schema_str = return f"({', '.join(params)}) -> {ret}"
        args = []
        for p in self.forward_ordered_parameter_names:
            if p in (self.forward_args, self.forward_kwargs):
                args.append(f"Tensor[] {p}")
            else:
                args.append(self._annotated_input(p))
        return f"({', '.join(args)}) -> {self._annotated_output()}"


def loop_for(
    body: Callable[..., Tuple[torch.Tensor]],
    n_iter: Union[torch.SymInt, torch.Tensor],
    *args: torch.Tensor,
    reduction_dim: Optional[Tuple[int]] = None,
) -> Tuple[torch.Tensor, ...]:
    """
    Helpers to export loop.
    """
    assert args, "The function should have at least one arg."
    if is_exporting():
        raise NotImplementedError()

    res = []
    for i in torch.range(0, n_iter, dtype=n_iter.dtype):
        r = body(i, *args)
        assert isinstance(
            r, tuple
        ), f"Unexpected type {r} for function {body}, it must be a tuple."
        assert not res or len(r) == len(res[-1]), (
            f"Unexpected number of results {len(r)} for function {body}, "
            f"expected {len(res[-1])}"
        )
        res.append(r)

    if not res:
        return torch.empty(tuple(), dtype=torch.float32, device=args[0].device)
    n_res = len(res[0])
    return [
        torch.cat(
            [r[i] for r in res],
            dim=0 if reduction_dim is None or i >= len(reduction_dim) else reduction_dim[i],
        )
        for i in range(n_res)
    ]
