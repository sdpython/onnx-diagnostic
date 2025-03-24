import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from ..helpers import string_type


class ModelInputs:
    """ """

    def __init__(
        self,
        model: torch.nn.Module,
        inputs: Union[
            List[Tuple[Any, ...]],
            List[Dict[str, Any]],
            List[Tuple[Tuple[Any, ...], Dict[str, Any]]],
        ],
        level: int = 0,
        method_name: str = "forward",
        name: str = "main",
    ):
        assert isinstance(model, torch.nn.Module) or inspect.ismodule(
            model
        ), f"unexpected type for model={type(model)}, it must be a torch.nn.Module"
        assert name, (
            f"name={name!r} cannot be empty this string is used to "
            f"display meaningfull error messages"
        )
        self.name = name
        self.model = model
        self.level = level
        self.method_name = method_name
        self.forward = getattr(model, method_name)
        self.signature = inspect.signature(self.forward)

        # information about the signature
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

        # process the inputs
        self.inputs = self.process_inputs(inputs)

    def process_inputs(
        self,
        inputs: Union[
            List[Tuple[Any, ...]],
            List[Dict[str, Any]],
            List[Tuple[Tuple[Any, ...], Dict[str, Any]]],
        ],
    ) -> List[Tuple[Tuple[Any, ...], Dict[str, Any]]]:
        """ """
        if not isinstance(inputs, list):
            raise ValueError(
                f"inputs should be specifed as a list of sets of "
                f"inputs but type(inputs) is {type(inputs)}"
            )
        new_inputs = []
        for i, inp in enumerate(inputs):
            if (
                isinstance(inp, tuple)
                and len(inp) == 2
                and isinstance(inp[0], tuple)
                and isinstance(inp[1], dict)
            ):
                new_inputs.append(inp)
                continue
            if isinstance(inp, tuple):
                new_inputs.append((inp, {}))
                continue
            if isinstance(inp, dict):
                new_inputs.append(((), inp))
                continue
            raise ValueError(f"Unable to interpret inputs {i}: {string_type(inp)}")
        return new_inputs

    @property
    def true_model_name(self):
        "Returns class name or module name."
        return (
            self.model.__class__.__name__
            if isinstance(self.model, torch.nn.Module)
            else self.model.__name__
        )

    @property
    def full_name(self):
        "Returns a name and class name."
        if self.method_name == "forward":
            return f"{self.name}:{self.true_model_name}"
        return f"{self.name}:{self.true_model_name}.{self.method_name}"

    def guess_dynamic_dimensions(self, *tensors) -> Any:
        """Infers the dynamic dimension from multiple shapes."""
        if len(tensors) == 1:
            return {}
        shapes = [t.shape for t in tensors]
        set_length = set(len(s) for s in shapes)
        assert len(set_length) == 1, (
            f"Shapes can be different but not ranks possible shapes={set_length} "
            f"shapes={shapes} for module {self.name!r}, "
            f"class={self.true_model_name!r}"
        )
        dynamic = torch.export.Dim.DYNAMIC
        rk = set_length.pop()
        res = {}
        for i in range(rk):
            set_dim = set(s[i] for s in shapes)
            if len(set_dim) > 1:
                res[i] = dynamic
                continue
            if set_dim == {0}:
                # It is unexpected to find a null dimension. Let's replace it by a dynamic one.
                res[i] = dynamic
                continue
        return res

    def guess_dynamic_shape_object(self, *objs: Any, msg: Optional[Callable] = None) -> Any:
        """Guesses the dynamic shapes for one argument."""
        if len(objs) == 0:
            return None
        set_types = set(type(o) for o in objs)
        assert (
            len(set_types) == 1
        ), f"Unexpected variety of input type {set_types}{msg() if msg else ''})"
        obj = objs[0]
        if obj is None:
            return None
        if isinstance(obj, (bool, int, float, str)):
            return None
        if isinstance(obj, (torch.Tensor, np.ndarray)):
            return self.guess_dynamic_dimensions(*objs)

        if isinstance(obj, tuple):
            kl = set(len(o) for o in objs)
            assert (
                len(kl) == 1
            ), f"Unexpected variety of tuple lengths {kl}{msg() if msg else ''}"
            shapes = []
            for i in range(kl.pop()):
                shapes.append(self.guess_dynamic_shape_object(*[o[i] for o in objs]))
            return tuple(shapes)

        if isinstance(obj, list):
            kl = set(len(o) for o in objs)
            assert (
                len(kl) == 1
            ), f"Unexpected variety of list lengths {kl}{msg() if msg else ''}"
            shapes = []
            for i in range(kl.pop()):
                shapes.append(self.guess_dynamic_shape_object(*[o[i] for o in objs]))
            return shapes

        if obj.__class__.__name__ in ("DynamicCache", "patched_DynamicCache"):
            kc = set(len(o.key_cache) for o in objs)
            assert (
                len(kc) == 1
            ), f"All attribute 'key_cache' should have the same length but found {kc}"
            vc = set(len(o.value_cache) for o in objs)
            assert (
                len(vc) == 1
            ), f"All attribute 'value_cache' should have the same length but found {vc}"
            key_cache = []
            for i in range(kc.pop()):
                key_cache.append(
                    self.guess_dynamic_dimensions(*[o.key_cache[i] for o in objs])
                )
            value_cache = []
            for i in range(vc.pop()):
                value_cache.append(
                    self.guess_dynamic_dimensions(*[o.value_cache[i] for o in objs])
                )
            return [key_cache, value_cache]

        raise NotImplementedError(
            f"Unable to build dynamic shapes for type {set_types.pop()}: "
            f"{string_type(objs)}{msg() if msg else ''} in {self.module_name_type}"
        )

    def guess_dynamic_shapes(self) -> Any:
        """
        Guesses the dynamic shapes for that module from two execution.
        If there is only one execution, then that would be static dimensions.
        """
        if len(self.inputs) == 0:
            # No inputs, unable to guess.
            return None
        if len(self.inputs) == 1:
            # No dynamic shapes.
            return tuple(self.guess_dynamic_shape_object(a) for a in self.inputs[0][0]), {
                k: self.guess_dynamic_shape_object(v) for k, v in self.inputs[0][1]
            }

        # Otherwise.
        s1 = set(len(i[0]) for i in self.inputs)
        assert (
            len(s1) == 1
        ), f"Different numbers of positional arguments {s1} for {self.full_name}"
        s2 = set(tuple(sorted(set(i[1]))) for i in self.inputs)
        assert len(s2) == 1, f"Different named arguments {s2} for {self.full_name}"
        args = []
        kwargs = {}
        for i in range(s1.pop()):
            objs = [_[0][i] for _ in self.inputs]
            args.append(
                self.guess_dynamic_shape_object(*objs, msg=lambda i=i: f" failing input {i}")
            )
        names = s2.pop()
        for name in names:
            assert name not in {"_diag", "verbose"}, (
                f"{self.full_name}: unexpected parameter {name!r}, names={names}"
                f"\ninputs[0]={string_type(self.inputs[0], with_shape=True)}"
                f"\ninputs[1]={string_type(self.inputs[1], with_shape=True)}"
            )

            objs = [_[1][name] for _ in self.inputs]
            kwargs[name] = self.guess_dynamic_shape_object(
                *objs, msg=lambda name=name: f" failing input {name!r}"
            )
        return tuple(args), kwargs

    def move_to_kwargs(
        self, args: Tuple[Any, ...], kwargs: Dict[str, Any], dynamic_shapes: Any
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any], Dict[str, Any]]:
        """
        Uses the signatures to move positional arguments (args) to named arguments (kwargs)
        with the corresponding dynamic shapes.
        *kwargs*, *dynamic_shapes* are modified inplace.
        """
        sig = self.signature
        arg_dyn, kw_dyn = dynamic_shapes
        for i, p in enumerate(sig.parameters):
            if i >= len(arg_dyn):
                break
            kw_dyn[p] = arg_dyn[i]
        if self.forward_kwargs:
            kdw = {}
            for k, v in kw_dyn.items():
                if k not in self.forward_parameter_names:
                    kdw[k] = v
            if kdw:
                for k in kdw:
                    del kw_dyn[k]
                kw_dyn[self.forward_kwargs] = kdw

            # Let's reorder as it seems to matter later
            # in the shape inference algorithm.
            _kwargs = kwargs
            kwargs = {}
            _kw_dyn = kw_dyn
            kw_dyn = {}
            for name in self.forward_ordered_parameter_names:
                if name in _kwargs:
                    kwargs[name] = _kwargs[name]
                if name in _kw_dyn:
                    kw_dyn[name] = _kw_dyn[name]
            for k in _kwargs:
                if k not in kwargs:
                    # Then it is part of **kwargs.
                    kwargs[k] = _kwargs[k]
            assert len(kw_dyn) == len(_kw_dyn), (
                f"{self.full_name}: unexpected mismatch between _kw_dyn={set(_kw_dyn)} "
                f"and kw_dyn={set(kw_dyn)}, "
                f"forward_ordered_parameter_names={self.forward_ordered_parameter_names}"
            )
            assert len(kwargs) == len(_kwargs), (
                f"{self.full_name}: unexpected mismatch between _kwargs={set(_kwargs)} "
                f"and kwargs={set(kwargs)}, "
                f"forward_ordered_parameter_names={self.forward_ordered_parameter_names}"
            )
        return args, kwargs, (tuple(), kw_dyn)
