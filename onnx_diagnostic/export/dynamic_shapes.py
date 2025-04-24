import inspect
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import torch
from ..helpers import string_type
from ..helpers.cache_helper import flatten_unflatten_for_dynamic_shapes

DYNAMIC_SHAPES = Tuple[Tuple[Any, ...], Dict[str, Any]]


def flatten_dynamic_shapes(ds: Any) -> Any:
    """Flattens the dynamic shapes."""
    if isinstance(ds, list):
        return _flat_list([flatten_dynamic_shapes(t) for t in ds])
    if isinstance(ds, tuple):
        return tuple(_flat_list([flatten_dynamic_shapes(t) for t in ds]))
    if isinstance(ds, dict):
        if all(isinstance(i, int) for i in ds):
            # That's a dynamic shape
            return ds
        return _flat_list([flatten_dynamic_shapes(t) for t in ds.values()])
    raise AssertionError(f"Not implemented for {type(ds)}: {ds}")


def _flat_list(li: List[Any]) -> List[Dict[int, str]]:
    res = []
    for t in li:
        if isinstance(t, dict):
            res.append(t)
        else:
            res.extend(t)
    return res


class CoupleInputsDynamicShapes:
    """
    Pair inputs / dynamic shapes.

    :param args: positional arguments
    :param kwargs: named arguments
    :param dynamic_shapes: dynamic shapes
    :param args_names: if both args and kwargs are not empty, then
        dynamic shapes must be a dictionary, and positional must be added
        to the named arguments. Arguments names or a module must be given
        in that case.
    """

    def __init__(
        self,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        dynamic_shapes: DYNAMIC_SHAPES,
        args_names: Optional[Union[torch.nn.Module, List[str]]] = None,
    ):
        self.args = args
        self.kwargs = kwargs
        self.dynamic_shapes = dynamic_shapes
        self.args_names = args_names

    def __str__(self) -> str:
        return "\n".join(
            [
                f"{self.__class__.__name__}(",
                f"    args={string_type(self.args, with_shape=True)},"
                f"    kwargs={string_type(self.kwargs, with_shape=True)},"
                f"    dynamic_shapes={string_type(self.dynamic_shapes, with_shape=True)},"
                f")",
            ]
        )

    def replace_string_by(self, value: Any = None):
        """
        Replaces string by the value ``torch.export.Dim.DYNAMIC``
        (default) or any other value specified by value.

        Example:

        .. runpython::
            :showcode:

            import torch
            from onnx_diagnostic.export.dynamic_shapes import CoupleInputsDynamicShapes

            T3x1 = torch.rand((3, 1))
            T3x4 = torch.rand((3, 4))
            ds_batch = {0: "batch"}
            ds_batch_seq = {0: "batch", 1: "seq"}
            kwargs = {"A": T3x4, "B": (T3x1, T3x1)}
            ds = {"A": ds_batch, "B": (ds_batch, ds_batch_seq)}
            print(CoupleInputsDynamicShapes((), kwargs, ds).replace_string_by())
        """
        return self._generic_walker(
            lambda inputs, ds, value=value: self._replace_string_dim_tensor(
                inputs, ds, value=value
            ),
            flatten_unflatten=True,
        )

    @classmethod
    def _replace_string_dim_tensor(cls, inputs, ds, value=None):
        assert isinstance(inputs, torch.Tensor), f"unexpected type for inputs {type(inputs)}"
        assert isinstance(ds, dict) and all(isinstance(s, int) for s in ds), (
            f"Unexpected types, inputs is a Tensor but ds is {ds}, "
            f"a dictionary is expected to specify a dimension"
        )
        if value is None:
            value = torch.export.Dim.DYNAMIC
        new_ds = ds.copy()
        for i, v in ds.items():
            if isinstance(v, str):
                new_ds[i] = value
        return new_ds

    def replace_by_string(self):
        """
        Replaces dimensions by strings.

        Example:

        .. runpython::
            :showcode:

            import torch
            from onnx_diagnostic.export.dynamic_shapes import CoupleInputsDynamicShapes

            Dim = torch.export.Dim
            T3x1 = torch.rand((3, 1))
            T3x4 = torch.rand((3, 4))
            ds_batch = {0: Dim("batch")}
            ds_batch_seq = {0: Dim("batch"), 1: Dim("seq")}
            kwargs = {"A": T3x4, "B": (T3x1, T3x1)}
            ds = {"A": ds_batch, "B": (ds_batch, ds_batch_seq)}
            print(CoupleInputsDynamicShapes((), kwargs, ds).replace_by_string())
        """
        unique = set()
        return self._generic_walker(
            lambda inputs, ds, unique=unique: self._replace_dim_tensor_by_string(
                inputs, ds, unique=unique
            ),
            flatten_unflatten=True,
        )

    @classmethod
    def _replace_dim_tensor_by_string(cls, inputs, ds, unique: Set[str]):
        assert isinstance(inputs, torch.Tensor), f"unexpected type for inputs {type(inputs)}"
        assert isinstance(ds, dict) and all(isinstance(s, int) for s in ds), (
            f"Unexpected types, inputs is a Tensor but ds is {ds}, "
            f"a dictionary is expected to specify a dimension"
        )
        new_ds = ds.copy()
        for i, v in ds.items():
            if isinstance(v, str):
                unique.add(v)
                new_ds[i] = v
            elif v in (torch.export.Dim.DYNAMIC, torch.export.Dim.AUTO):
                name = f"Dim{len(unique)}"
                new_ds[i] = name
                unique.add(name)
            else:
                name = v.__name__
                unique.add(name)
                new_ds[i] = name
        return new_ds

    def invalid_dimensions_for_export(self):
        """
        Tells if the inputs are valid based on the dynamic shapes definition.
        The method assumes that all custom classes can be serialized.
        If some patches were applied to export, they should enabled while
        calling this method if the inputs contains such classes.

        The function checks that a dynamic dimension does not receive a value
        of 0 or 1. It returns the unexpected values in the same structure as
        the given dynamic shapes.

        Example:

        .. runpython::
            :showcode:

            import torch
            from onnx_diagnostic.export.dynamic_shapes import CoupleInputsDynamicShapes

            T3x1 = torch.rand((3, 1))
            T3x4 = torch.rand((3, 4))
            ds_batch = {0: "batch"}
            ds_batch_seq = {0: "batch", 1: "seq"}
            kwargs = {"A": T3x4, "B": (T3x1, T3x1)}
            ds = {"A": ds_batch, "B": (ds_batch, ds_batch_seq)}
            print(CoupleInputsDynamicShapes((), kwargs, ds).invalid_dimensions_for_export())

        In case it works, it shows:

        .. runpython::
            :showcode:

            import torch
            from onnx_diagnostic.export.dynamic_shapes import CoupleInputsDynamicShapes

            T3x2 = torch.rand((3, 2))
            T3x4 = torch.rand((3, 4))
            ds_batch = {0: "batch"}
            ds_batch_seq = {0: "batch", 1: "seq"}
            kwargs = {"A": T3x4, "B": (T3x2, T3x2)}
            ds = {"A": ds_batch, "B": (ds_batch, ds_batch_seq)}
            print(CoupleInputsDynamicShapes((), kwargs, ds).invalid_dimensions_for_export())
        """
        return self._generic_walker(self._valid_shapes_tensor, flatten_unflatten=True)

    @classmethod
    def _valid_shapes_tensor(cls, inputs, ds):
        assert isinstance(inputs, torch.Tensor), f"unexpected type for inputs {type(inputs)}"
        assert isinstance(ds, dict) and all(isinstance(s, int) for s in ds), (
            f"Unexpected types, inputs is a Tensor but ds is {ds}, "
            f"a dictionary is expected to specify a dimension dimension"
        )
        issues = {}
        for i, d in enumerate(inputs.shape):
            if i in ds and not isinstance(ds[i], int):
                # dynamic then
                if d in {0, 1}:
                    # export issues for sure
                    issues[i] = f"d=[{d}]"
        return issues if issues else None

    def _generic_walker(
        self, processor: Callable, args_kwargs: bool = False, flatten_unflatten: bool = False
    ):
        """
        Generic deserializator walking through inputs and dynamic_shapes all along.
        The function returns a result with the same structure as the dynamic shapes.
        """
        if not self.args:
            assert isinstance(self.kwargs, dict) and isinstance(self.dynamic_shapes, dict), (
                f"Type mismatch, args={string_type(self.args)} and "
                f"dynamic_shapes={self.dynamic_shapes} should have the same type."
            )
            res = self._generic_walker_step(
                processor,
                self.kwargs,
                self.dynamic_shapes,
                flatten_unflatten=flatten_unflatten,
            )
            return (tuple(), res) if args_kwargs else res

        if not self.kwargs:
            assert isinstance(self.args, tuple) and isinstance(self.dynamic_shapes, tuple), (
                f"Type mismatch, args={string_type(self.args)} and "
                f"dynamic_shapes={self.dynamic_shapes} should have the same type."
            )
            res = self._generic_walker_step(
                processor, self.args, self.dynamic_shapes, flatten_unflatten=flatten_unflatten
            )
            return (res, {}) if args_kwargs else res

        assert isinstance(self.dynamic_shapes, dict), (
            f"Both positional and named arguments (args and kwargs) are filled. "
            f"dynamic shapes must a dictionary not {type(self.dynamic_shapes)}"
        )
        if not self.args_names and set(self.dynamic_shapes) & set(self.kwargs) == set(
            self.dynamic_shapes
        ):
            # No dynamic shapes for the positional arguments.
            return self._generic_walker_step(
                processor,
                self.kwargs,
                self.dynamic_shapes,
                flatten_unflatten=flatten_unflatten,
            )

        if isinstance(self.args_names, list):
            if not set(self.args_names) & set(self.dynamic_shapes):
                # No dynamic shapes for the positional arguments.
                return self._generic_walker_step(
                    processor,
                    self.kwargs,
                    self.dynamic_shapes,
                    flatten_unflatten=flatten_unflatten,
                )

            assert self.args_names, (
                "args and kwargs are filled, then args_names must be specified in "
                "the constructor to move positional arguments to named arguments."
            )
            assert len(self.args) <= len(self.args_names), (
                f"There are {len(self.args)} positional arguments "
                f"but only {len(self.args_names)} names. "
                f"args={string_type(self.args, with_shape=True)}, args_name={self.args_names}"
            )
            kwargs = dict(zip(self.args_names, self.args))
            kwargs.update(self.kwargs)
            res = self._generic_walker_step(
                processor, kwargs, self.dynamic_shapes, flatten_unflatten=flatten_unflatten
            )
            if args_kwargs:
                pgs = [None for _ in range(len(self.args))]
                kws = {}
                for k, v in res.items():
                    if k not in self.kwargs:
                        pgs[self.args_names.index(k)] = v
                    else:
                        kws[k] = v
                return pgs, kws
            return res

        raise NotImplementedError(
            f"Not yet implemented when args is filled, "
            f"kwargs as well but args_names is {type(self.args_names)}"
        )

    @classmethod
    def _generic_walker_step(
        cls, processor: Callable, inputs, ds, flatten_unflatten: bool = False
    ):
        if isinstance(inputs, torch.Tensor):
            return processor(inputs, ds)
        if isinstance(inputs, (int, float, str)):
            return None
        if type(inputs) in (tuple, list, dict):
            # Type must be strict, some custom classes can inherit from those.
            assert type(inputs) is type(ds), (
                f"Input type and dynamic shape type mush match but "
                f"type(inputs)={type(inputs)}, type(ds)={type(ds)}, "
                f"inputs={string_type(inputs, with_shape=True)}, ds={ds}"
            )
            assert len(ds) == len(inputs), (
                f"Length mismatch between inputs {len(inputs)} "
                f"and ds={len(ds)}\n"
                f"inputs={string_type(inputs, with_shape=True)}, ds={ds}"
            )
            if type(inputs) in (tuple, list):
                value = []
                for i, d in zip(inputs, ds):
                    value.append(
                        cls._generic_walker_step(
                            processor, i, d, flatten_unflatten=flatten_unflatten
                        )
                    )
                return (
                    (value if isinstance(ds, list) else tuple(value))
                    if any(v is not None for v in value)
                    else None
                )
            assert type(inputs) is dict, f"Unexpected type for inputs {type(inputs)}"
            assert set(inputs) == set(ds), (
                f"Keys mismatch between inputs {set(inputs)} and ds={set(ds)}, "
                f"inputs={string_type(inputs, with_shape=True)}, ds={ds}"
            )
            dvalue = {}
            for k, v in inputs.items():
                t = cls._generic_walker_step(
                    processor, v, ds[k], flatten_unflatten=flatten_unflatten
                )
                if t is not None:
                    dvalue[k] = t
            return dvalue if dvalue else None

        # A custom class.
        assert inputs.__class__ in torch.utils._pytree.SUPPORTED_NODES, (
            f"Class {inputs.__class__.__name__!r} was not registered using "
            f"torch.utils._pytree.register_pytree_node, it is not possible to "
            f"map this class with the given dynamic shapes."
        )
        if flatten_unflatten:
            flatunflat = flatten_unflatten_for_dynamic_shapes(inputs)
            res = cls._generic_walker_step(
                processor, flatunflat, ds, flatten_unflatten=flatten_unflatten
            )
            # Should we restore the original class?
            return res
        flat, spec = torch.utils._pytree.tree_flatten(inputs)
        if all(isinstance(t, torch.Tensor) for t in flat):
            # We need to flatten dynamic shapes as well
            ds = flatten_dynamic_shapes(ds)
        res = cls._generic_walker_step(
            processor, flat, ds, flatten_unflatten=flatten_unflatten
        )
        # Then we restore the original class.
        return torch.utils._pytree.tree_unflatten(res, spec)

    class ChangeDimensionProcessor:
        def __init__(self, desired_values):
            self.mapping = desired_values or {}

        def _build_new_shape(
            self, shape: Tuple[int, ...], ds: Dict[int, Any]
        ) -> Tuple[int, ...]:
            new_shape = list(shape)
            for i in range(len(shape)):
                if i in ds:
                    if isinstance(ds[i], str):
                        d = ds[i]
                    elif isinstance(
                        ds[i],
                        (
                            torch.export.dynamic_shapes._DerivedDim,
                            torch.export.dynamic_shapes._Dim,
                        ),
                    ):
                        d = str(ds[i])
                    elif not isinstance(ds[i], int):
                        raise NotImplementedError(f"Unable to handle type {ds[i]} in {ds}")
                    if d in self.mapping:
                        new_dim = self.mapping[d]
                    else:
                        new_dim = shape[i] + 1
                        self.mapping[d] = new_dim
                    new_shape[i] = new_dim
            return tuple(new_shape)

        def _build_new_tensor(self, tensor: torch.Tensor, new_shape: Tuple[int, ...]):
            rank = len(tensor.shape)
            for i in range(len(tensor.shape)):
                d0 = tensor.shape[i]
                d1 = new_shape[i]
                if d0 == d1:
                    continue
                alt_shape = list(tensor.shape)
                alt_shape[i] = d1
                new_tensor = torch.zeros(
                    tuple(alt_shape), dtype=tensor.dtype, device=tensor.device
                )
                mind = min(d0, d1)
                indices: List[Union[slice, int]] = [slice(None) for _ in range(rank)]
                indices[i] = slice(0, mind)
                ind = tuple(indices)
                new_tensor[ind] = tensor[ind]
                if d1 > mind:
                    for k in range(d1 - mind):
                        indices0: List[Union[slice, int]] = [slice(None) for _ in range(rank)]
                        indices1: List[Union[slice, int]] = [slice(None) for _ in range(rank)]
                        indices1[i] = mind + k
                        indices0[i] = k % mind
                        new_tensor[tuple(indices1)] = tensor[tuple(indices0)]
                tensor = new_tensor
            return tensor

        def __call__(self, inputs, ds):
            assert isinstance(
                inputs, torch.Tensor
            ), f"unexpected type for inputs {type(inputs)}"
            assert isinstance(ds, dict) and all(isinstance(s, int) for s in ds), (
                f"Unexpected types, inputs is a Tensor but ds is {ds}, "
                f"a dictionary is expected to specify a dimension dimension"
            )
            new_shape = self._build_new_shape(inputs.shape, ds)
            return self._build_new_tensor(inputs, new_shape)

    def change_dynamic_dimensions(
        self, desired_values: Optional[Dict[str, int]] = None, args_kwargs: bool = False
    ):
        """
        A model exported with dynamic shapes is not necessarily dynamic
        just because the user specified dynamic shapes. The algorithm
        may discover that a dimension cannot be dynamic and then continues
        the export making the assumption it is static. That may lead a wrong
        model. This function produces a new set of inputs with different values
        for the dimension than the first ones, assuming they were used to export
        the model.

        :param desired_values: to fixed named dimension to have the desired value
        :param args_kwargs: return both args, kwargs even if empty
        :return: new inputs

        Example:

        .. runpython::
            :showcode:

            import torch
            from onnx_diagnostic.helpers import string_type
            from onnx_diagnostic.export.dynamic_shapes import CoupleInputsDynamicShapes

            T3x15 = torch.rand((3, 15))
            T3x20 = torch.rand((3, 20))
            T3x4 = torch.rand((3, 4))
            ds_batch = {0: "batch"}
            ds_batch_seq = {0: "batch", 1: "seq"}
            kwargs = {"A": T3x4, "B": (T3x15, T3x20)}
            ds = {"A": ds_batch, "B": (ds_batch, ds_batch_seq)}
            new_kwargs = CoupleInputsDynamicShapes((), kwargs, ds).change_dynamic_dimensions()
            print("before:", string_type(kwargs, with_shape=True))
            print("-after:", string_type(new_kwargs, with_shape=True))
        """
        return self._generic_walker(
            self.ChangeDimensionProcessor(desired_values), args_kwargs=args_kwargs
        )


class ModelInputs:
    """
    Wraps a model and a couple of sets of valid inputs.
    Based on that information, the class is able to infer the dynamic shapes
    for :func:`torch.export.export`.

    :param model: model to export
    :param inputs: list of valid set of inputs
    :param level: if this module is a submodule, it is the level of submodule
    :param method_name: by default, the forward method is processed but it
        could be another one
    :param name: a name, mostly for debugging purposes

    Examples:

    **args**

    .. runpython::
        :showcode:

        import pprint
        import torch
        from onnx_diagnostic.export import ModelInputs


        class Model(torch.nn.Module):
            def forward(self, x, y):
                return x + y


        model = Model()
        x = torch.randn((5, 6))
        y = torch.randn((1, 6))
        model(x, y)  # to check it works

        inputs = [(x, y), (torch.randn((7, 8)), torch.randn((1, 8)))]
        mi = ModelInputs(Model(), inputs)
        ds = mi.guess_dynamic_shapes()
        pprint.pprint(ds)

    **kwargs**

    .. runpython::
        :showcode:

        import pprint
        import torch
        from onnx_diagnostic.export import ModelInputs

        class Model(torch.nn.Module):
            def forward(self, x, y):
                return x + y


        model = Model()
        x = torch.randn((5, 6))
        y = torch.randn((1, 6))
        model(x=x, y=y)  # to check it works

        inputs = [dict(x=x, y=y), dict(x=torch.randn((7, 8)), y=torch.randn((1, 8)))]
        mi = ModelInputs(Model(), inputs)
        ds = mi.guess_dynamic_shapes()
        pprint.pprint(ds)

    **args and kwargs**

    .. runpython::
        :showcode:

        import pprint
        import torch
        from onnx_diagnostic.export import ModelInputs

        class Model(torch.nn.Module):
            def forward(self, x, y):
                return x + y


        model = Model()
        x = torch.randn((5, 6))
        y = torch.randn((1, 6))
        model(x, y=y)  # to check it works

        inputs = [((x,), dict(y=y)), ((torch.randn((7, 8)),), dict(y=torch.randn((1, 8))))]
        mi = ModelInputs(Model(), inputs)
        ds = mi.guess_dynamic_shapes()
        pprint.pprint(ds)

    :func:`torch.export.export` does not like dynamic shapes defined both as args and kwargs.
    kwargs must be used. ``move_to_kwargs`` modifies the inputs and the dynamic shapes
    to make the model and the given inputs exportable.

    .. runpython::
        :showcode:

        import pprint
        import torch
        from onnx_diagnostic.export import ModelInputs
        from onnx_diagnostic.helpers import string_type


        class Model(torch.nn.Module):
            def forward(self, x, y):
                return x + y


        model = Model()
        x = torch.randn((5, 6))
        y = torch.randn((1, 6))
        model(x, y=y)  # to check it works

        inputs = [((x,), dict(y=y)), ((torch.randn((7, 8)),), dict(y=torch.randn((1, 8))))]
        mi = ModelInputs(Model(), inputs)
        ds = mi.guess_dynamic_shapes()

        a, kw, nds = mi.move_to_kwargs(*mi.inputs[0], ds)
        print("moved args:", string_type(a, with_shape=True))
        print("moved kwargs:", string_type(kw, with_shape=True))
        print("dynamic shapes:")
        pprint.pprint(nds)
    """

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
            f"display meaningful error messages"
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
        """
        Transforms a list of valid inputs, list of args, list of kwargs or list of both
        into a list of (args, kwargs).
        """
        if not isinstance(inputs, list):
            raise ValueError(
                f"inputs should be specified as a list of sets of "
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
    def true_model_name(self) -> str:
        "Returns class name or module name."
        return (
            self.model.__class__.__name__
            if isinstance(self.model, torch.nn.Module)
            else self.model.__name__
        )

    @property
    def full_name(self) -> str:
        "Returns a name and class name."
        if self.method_name == "forward":
            return f"{self.name}:{self.true_model_name}"
        return f"{self.name}:{self.true_model_name}.{self.method_name}"

    @property
    def module_name_type(self):
        "Returns name and module type."
        if self.method_name == "forward":
            return f"type({self.name})={self.true_model_name}"
        return f"type({self.name})={self.true_model_name}.{self.method_name}"

    def guess_dynamic_dimensions(
        self, *tensors, auto: bool = False
    ) -> Optional[Dict[int, Any]]:
        """
        Infers the dynamic dimension from multiple shapes.
        If auto is True, it returns ``torch.export.Dim.AUTO`` for every dimension
        which cannot be guessed. Two tensors with the same value for one dimension
        can be guessed, but if there is only 1, it cannot.
        """
        if len(tensors) == 1:
            if isinstance(tensors[0], (int, float)):
                return None
            assert isinstance(tensors[0], torch.Tensor), (
                f"Unexpected type for tensors {string_type(tensors, with_shape=True)}, "
                f"Only tensors are allowed."
            )
            return (
                {i: torch.export.Dim.AUTO for i in range(len(tensors[0].shape))}  # noqa: C420
                if auto
                else {}
            )
        shapes = [t.shape for t in tensors]
        set_length = set(len(s) for s in shapes)
        assert len(set_length) == 1, (
            f"Shapes can be different but not ranks possible shapes={set_length} "
            f"shapes={shapes} for module {self.name!r}, "
            f"class={self.true_model_name!r}"
        )
        dynamic: Any = torch.export.Dim.DYNAMIC  # type: ignore
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

    def guess_dynamic_shape_object(
        self, *objs: Any, auto: bool = False, msg: Optional[Callable] = None
    ) -> Any:
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
            return self.guess_dynamic_dimensions(*objs, auto=auto)

        if isinstance(obj, tuple):
            kl = set(len(o) for o in objs)
            assert (
                len(kl) == 1
            ), f"Unexpected variety of tuple lengths {kl}{msg() if msg else ''}"
            shapes: Any = []
            for i in range(kl.pop()):
                shapes.append(
                    self.guess_dynamic_shape_object(*[o[i] for o in objs], auto=auto, msg=msg)
                )
            return tuple(shapes)

        if isinstance(obj, list):
            kl = set(len(o) for o in objs)
            assert (
                len(kl) == 1
            ), f"Unexpected variety of list lengths {kl}{msg() if msg else ''}"
            shapes = []
            for i in range(kl.pop()):
                shapes.append(
                    self.guess_dynamic_shape_object(*[o[i] for o in objs], auto=auto, msg=msg)
                )
            return shapes

        if isinstance(obj, dict):
            kl = set(len(o) for o in objs)
            assert (
                len(kl) == 1
            ), f"Unexpected variety of dict lengths {kl}{msg() if msg else ''}"
            shapes = {}
            for i in obj:
                shapes[i] = self.guess_dynamic_shape_object(
                    *[o[i] for o in objs], auto=auto, msg=msg
                )
            return shapes

        if obj.__class__ in torch.utils._pytree.SUPPORTED_NODES:
            kcl = set(o.__class__ for o in objs)
            assert len(kcl) == 1, (
                f"All instances of argument {i} are not of the same class but {kcl}, "
                f"types should be the same."
            )
            col_args = [flatten_unflatten_for_dynamic_shapes(o) for o in objs]
            kc = set(len(o) for o in col_args)
            assert len(kc) == 1, (
                f"All instances of type {kcl.pop()} are not serialized into the same number "
                f"of arguments, it should be the same."
            )
            values = []
            for i in range(kc.pop()):
                values.append(
                    self.guess_dynamic_shape_object(
                        *[ca[i] for ca in col_args], auto=auto, msg=msg
                    )
                )
            return values

        # In case DynamicCache is not registered.
        if obj.__class__.__name__ == "DynamicCache":
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
                    self.guess_dynamic_dimensions(*[o.key_cache[i] for o in objs], auto=auto)
                )
            value_cache = []
            for i in range(vc.pop()):
                value_cache.append(
                    self.guess_dynamic_dimensions(*[o.value_cache[i] for o in objs], auto=auto)
                )
            return [key_cache, value_cache]

        raise NotImplementedError(
            f"Unable to build dynamic shapes for type {set_types.pop()}: "
            f"{string_type(objs)}{msg() if msg else ''} in {self.module_name_type}, "
            f"this object needs serialization function to be registered."
        )

    def guess_dynamic_shapes(self, auto: bool = False) -> DYNAMIC_SHAPES:
        """
        Guesses the dynamic shapes for that module from two execution.
        If there is only one execution, then that would be static dimensions.

        :param auto: if auto is True, use ``torch.export.Dim.AUTO`` for any
            dimension if the number of inputs is one
        """
        if len(self.inputs) == 0:
            # No inputs, unable to guess.
            return (tuple(), {})
        if len(self.inputs) == 1:
            # No dynamic shapes.
            return tuple(
                self.guess_dynamic_shape_object(a, auto=auto) for a in self.inputs[0][0]
            ), {
                k: self.guess_dynamic_shape_object(v, auto=auto)
                for k, v in self.inputs[0][1].items()
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
                self.guess_dynamic_shape_object(
                    *objs, auto=auto, msg=lambda i=i: f" failing input {i}"
                )
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
                *objs, auto=auto, msg=lambda name=name: f" failing input {name!r}"
            )
        return tuple(args), kwargs

    def move_to_kwargs(
        self,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        dynamic_shapes: Tuple[Tuple[Any, ...], Dict[str, Any]],
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any], DYNAMIC_SHAPES]:
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

    def validate_inputs_for_export(
        self, dynamic_shapes: Optional[DYNAMIC_SHAPES] = None
    ) -> List[List[Union[int, str]]]:
        """
        Validates the inputs the class contains for the given dynamic shapes.
        If not specified, the dynamic_shapes are guessed.

        :param dynamic_shapes: dynamic shapes to validate
        :return: a list of lists, every list contains the path the invalid dimension
        """
        if dynamic_shapes is None:
            if len(self.inputs) == 1:
                return []
            dyn_shapes = self.guess_dynamic_shapes()
        return [
            CoupleInputsDynamicShapes(*i, dyn_shapes).invalid_dimensions_for_export()
            for i in self.inputs
        ]
