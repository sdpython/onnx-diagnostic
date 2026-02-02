import contextlib
import inspect
import time
from typing import Any, Callable, Sequence
import onnx
import torch
from ..helpers import max_diff
from ..reference import OnnxruntimeEvaluator


def _flatten_unflatten_for_dynamic_shapes(
    obj: Any,
    use_dict: bool = True,
    change_function: Callable[[torch.Tensor], Any] | None = None,
) -> Any:
    """Returns the object in a different structure similar to what
    the definition of the dynamic shapes should use.

    Args:
        obj:
            object from a custom class
        use_dict:
            closer to the original result but
            :func:`torch.export.export` only considers the values,
            the context gives the dictionary keys but it is not expressed
            in the dynamic shapes, these specifications seems to be different
            for the strict and non strict mode. It also preserves tuple.
        change_function:
            to modify the tensor in the structure itself,
            like replace them by a shape

    Returns:
        the serialized object
    """
    if isinstance(obj, torch.Tensor):
        return change_function(obj) if change_function else obj
    flat, spec = torch.utils._pytree.tree_flatten(obj)
    start = 0
    end = 0
    subtrees = []
    for subspec in (spec.children() if hasattr(spec, "children") else spec.children_specs):
        end += subspec.num_leaves
        value = subspec.unflatten(flat[start:end])
        value = _flatten_unflatten_for_dynamic_shapes(
            value, use_dict=use_dict, change_function=change_function
        )
        subtrees.append(value)
        start = end
    if use_dict:
        if spec.type is dict:
            # This is a dictionary.
            return dict(zip(spec.context, subtrees))
        if spec.type is tuple:
            return tuple(subtrees)
        if spec.type is list:
            return list(subtrees)
        if spec.type is None and not subtrees:
            return None
        if spec.context:
            # This is a custom class with attributes.
            # It is returned as a list.
            return list(subtrees)
        raise ValueError(
            f"Unable to interpret spec type {spec.type} "
            f"(type is {type(spec.type)}, context is {spec.context}), "
            f"spec={spec}, subtrees={subtrees}"
        )
    # This is a list.
    return subtrees


def _infer_dynamic_dimensions(
    shape_list: Sequence[tuple[int, ...]], set_batch_dimension: bool = False
) -> list[int]:
    """Returns the list of dynamic dimensions given a list of shapes
    corresponding to the same tensor.

    Args:
        shape_list:
            list of shapes, they must all have the same length
        set_batch_dimension:
            make the first dimension dynamic if it is not

    Returns:
        list of dynamic dimensions
    """
    unique_ranks = {len(shape) for shape in shape_list}
    torch._check(
        len(unique_ranks) == 1, lambda: "all shapes in shape_list must have the same rank"
    )
    rank = unique_ranks.pop()
    dynamic = []
    for i in range(rank):
        dims = [shape[i] for shape in shape_list]
        if len(set(dims)) > 1 or (i == 0 and set_batch_dimension):
            dynamic.append(i)
    return dynamic


class InputCandidate:
    """Steals forward method to collect inputs and outputs.
    This information is used to infer dynamic shapes and
    export arguments.

    Examples
    --------
    >>> input_observer = InputObserver()
    >>> with input_observer(model):
    >>>     model(x1, y1)
    >>>     model(x2, y2)
    >>> ep = torch.export.export(  # or torch.onnx.export
    >>>     model,
    >>>     input_observer.infer_arguments(),
    >>>     dynamic_shapes.input_observer.infer_dynamic_shapes(),
    >>> )

    With LLM:
    >>> input_observer = InputObserver()
    >>> with input_observer(model):
    >>>     model.generate(input_ids)
    >>> ep = torch.export.export(  # or torch.onnx.export
    >>>     model,
    >>>     ()
    >>>     kwargs=input_observer.infer_arguments(),
    >>>     dynamic_shapes.input_observer.infer_dynamic_shapes(),
    >>> )

    See example :ref:`l-plot-tiny-llm-export-input-observer`.
    """

    def __init__(self, args: tuple[Any, ...], kwargs: dict[str, Any], clone: bool):
        self.args = args
        self.kwargs = kwargs
        self.flat_list, self.spec = torch.utils._pytree.tree_flatten((args, kwargs))
        self.n_tensors = sum(t is not None for t in self.flat_list)
        self._position_to_args_kwargs: list[int | str] | None = None
        self._n_tensors_for_args_kwargs: dict[int | str, int] | None = None

        if clone:
            self.flat_list = [
                (None if not isinstance(t, torch.Tensor) else t.clone().detach())
                for t in self.flat_list
            ]
            self.args, self.kwargs = torch.utils._pytree.tree_unflatten(
                self.flat_list, self.spec
            )

        self.aligned_spec: torch.utils._pytree.PyTreeSpec | None = None
        self.aligned_flat_list: list[torch.Tensor | None] | None = None

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}({len(self.args)} args, "
            f"{len(self.kwargs)} kwargs, {len(self.flat_list)} tensors, "
            f"{len(self.aligned_flat_list or [])} aligned tensors)"
        )

    def __len__(self) -> int:
        """Returns the number of flattended tensors, None tensors are included."""
        return len(self.flat_list)

    def build_mappings(self):
        if self._position_to_args_kwargs is not None:
            return self._position_to_args_kwargs
        self._n_tensors_for_args_kwargs = {}

        flat_index_to_args: list[int | str] = []
        for index_args, a in enumerate(self.args):
            size = len(torch.utils._pytree.tree_flatten(a)[0])
            self._n_tensors_for_args_kwargs[index_args] = size
            flat_index_to_args.extend([index_args] * size)
        for k, v in self.kwargs.items():
            size = len(torch.utils._pytree.tree_flatten(v)[0])
            self._n_tensors_for_args_kwargs[k] = size
            flat_index_to_args.extend([k] * size)

        self._position_to_args_kwargs = flat_index_to_args
        return self._position_to_args_kwargs

    @property
    def position_to_args_kwargs(self) -> list[int | str]:
        """Returns the corresponding args or kwargs
        for every tensor in the flattened inputs.
        """
        if self._position_to_args_kwargs is None:
            self.build_mappings()
        # type checking is missing it
        assert self._position_to_args_kwargs is not None
        return self._position_to_args_kwargs

    @property
    def n_tensors_for_args_kwargs(self) -> dict[int | str, int]:
        """Returns the number of flat tensors in every args or kwargs."""
        if self._n_tensors_for_args_kwargs is None:
            self.build_mappings()
        # type checking is missing it
        assert self._n_tensors_for_args_kwargs is not None
        return self._n_tensors_for_args_kwargs

    def _set_aligned_flat_list(
        self,
        aligned_flat_list: list[torch.Tensor | None],
        aligned_spec: torch.utils._pytree.PyTreeSpec,
    ):
        self.aligned_flat_list = aligned_flat_list
        self.aligned_spec = aligned_spec

    def align_with(
        self, best_candidate: "InputCandidate", captured_inputs: dict[int | str, int]
    ):
        """Two candidates are considered as aligned if after being flattened,
        they have the same number of tensors (None allowed)."""
        flat = []
        for i in range(len(best_candidate.args)):
            if i < len(self.args) and (isinstance(self.args[i], torch.Tensor) or self.args[i]):
                ts = torch.utils._pytree.tree_flatten(self.args[i])[0]
                if i in captured_inputs and captured_inputs[i] != len(ts):
                    raise RuntimeError(
                        f"Positional argument {i} has {len(ts)} tensors "
                        f"but previously got {captured_inputs[i]} tensors. "
                        f"Inference is impossible in that case."
                    )
                captured_inputs[i] = len(ts)
                flat.extend(ts)
                continue
            # If the argument i is not specified or is None or an empty container.
            flat.extend([None for _ in range(best_candidate.n_tensors_for_args_kwargs[i])])

        for k in best_candidate.kwargs:
            if k in self.kwargs and (
                isinstance(self.kwargs[k], torch.Tensor) or self.kwargs[k]
            ):
                ts = torch.utils._pytree.tree_flatten(self.kwargs[k])[0]
                if k in captured_inputs and captured_inputs[k] != len(ts):
                    raise RuntimeError(
                        f"Named argument {k!r} has {len(ts)} tensors "
                        f"but previously got {captured_inputs[k]} tensors in "
                        f"kwargs={list(self.kwargs)}. "
                        f"Inference is impossible in that case."
                    )
                captured_inputs[k] = len(ts)
                flat.extend(ts)
                continue
            # If the argument k is not specified or is None or an empty container.
            flat.extend([None for _ in range(best_candidate.n_tensors_for_args_kwargs[k])])

        self._set_aligned_flat_list(flat, best_candidate.spec)

    @property
    def n_aligned_tensors(self) -> int:
        if self.aligned_flat_list is None:
            raise RuntimeError("This input was not aligned with the others.")
        return len(self.aligned_flat_list)


class InputObserverInfo:
    """Contains all the necessary information to infer dynamic shapes
    and the arguments to send to :func:`torch.export.export`.

    Args:
        signature_names: Names of the arguments of the method
            the collector tensors come from. They are used if it becomes
            necessary to move positional arguments to named ones.
            They are used a second time because :func:`torch.export.export`
            cares about the order in kwargs and dynamic shapes, it needs
            to be the same in the ordered dictionaries `add_inputs` receive.
    """

    def __init__(self, signature_names: list[str]):
        self.inputs: list[InputCandidate] = []
        self.outputs_specs: list[torch.utils._pytree.PyTreeSpec] = []
        self.flat_outputs: list[list[torch.Tensor]] = []
        self.latencies: list[float] = []
        self.signature_names = signature_names
        self._best_candidate: InputCandidate | None = None
        self._captured_inputs: dict[int | str, int] | None = None

    def __len__(self) -> int:
        """Returns the number of collected set of inputs/outputs."""
        return len(self.inputs)

    def add_inputs(self, args: tuple[Any, ...], kwargs: dict[str, Any]):
        """Stores one set of inputs. They are deepcopied.

        Args:
            args: Positional arguments.
            kwargs: Named arguments.
        """
        kwargs = {
            k: v
            for k, v in kwargs.items()
            if v is not None and not isinstance(v, (int, float, bool))
        }

        # kwargs may come in a different ordeer teach.
        # dictionaries are ordered and torch.export.export expects
        # dynamic shapes an kwargs to follow the same order.

        ordered_kwargs = {k: kwargs[k] for k in self.signature_names if k in kwargs}
        for k, v in kwargs.items():
            if k not in ordered_kwargs:
                ordered_kwargs[k] = v

        candidate = InputCandidate(args, ordered_kwargs, clone=True)
        self.inputs.append(candidate)
        if self._best_candidate is None or len(self._best_candidate) < len(candidate):
            self._best_candidate = candidate

    def add_outputs(self, res: torch.Tensor | tuple[torch.Tensor, ...], latency: float):
        """Stores outputs. They are deepcopied."""
        flat_res, spec = torch.utils._pytree.tree_flatten(res)
        self.outputs_specs.append(spec)
        self.flat_outputs.append([t.clone().detach() for t in flat_res])
        self.latencies.append(latency)

    def align_inputs_none_values(self):
        """Once the best candidate is chosen, this method aligns every set of inputs
        on the best candidate, it inserts None at the right position when
        optional inputs are not specified. We consider a set of inputs is aligned
        if this method does not change the original flattened inputs.
        """
        if not self.inputs or self._best_candidate is None:
            raise RuntimeError("No inputs were captured.")

        if all(candidate.aligned_flat_list is not None for candidate in self.inputs):
            # No new inputs, no alignment is necessary.
            return

        # Let's reprocess everything.
        self._captured_inputs = {}
        for candidate in self.inputs:
            if len(set(candidate.kwargs) | set(self._best_candidate.kwargs)) > len(
                self._best_candidate.kwargs
            ):
                raise RuntimeError(
                    "At least one call to the observed model "
                    "must contain all the named arguments."
                )
            candidate.align_with(self._best_candidate, self._captured_inputs)

    def infer_dynamic_shapes(
        self,
        set_batch_dimension_for: set[int | str] | bool | None = None,
        return_flat: bool = False,
    ) -> tuple[dict[int, Any], ...] | dict[str, dict[int, Any]]:
        """Infers dynamic shapes.  based on the collected tensors.
        Most of the time, models do support a batch dimension
        but this batch dimension has the same value for every input sample.
        Instead of running inference on new samples, argument `set_batch_dimension_for`
        can be used to tell the first dimension is a dynamic dimension for a particular
        set of inputs referenced by their name (str) or their position (int).

        `return_flat` tells the function to return a flat tuple instead of
        nested structured.
        """
        self.align_inputs_none_values()
        # type checking
        assert self._best_candidate is not None
        assert self._best_candidate.flat_list is not None
        assert self._best_candidate.aligned_flat_list is not None

        def _set_batch_dimension(name_or_position):
            if not set_batch_dimension_for:
                return False
            if (
                isinstance(set_batch_dimension_for, bool) and set_batch_dimension_for
            ) or name_or_position in set_batch_dimension_for:
                return True
            if isinstance(name_or_position, int):
                torch._check(
                    name_or_position < len(self.signature_names),
                    lambda: f"argument at position {name_or_position} is out of boundary",
                )
                if self.signature_names[name_or_position] in set_batch_dimension_for:
                    return True
            return False

        def _set_batch_dimension_for_flat_index(index):
            # type checking
            assert self._best_candidate is not None
            return _set_batch_dimension(self._best_candidate.position_to_args_kwargs[index])

        if len(self._best_candidate.flat_list) != len(self._best_candidate.aligned_flat_list):
            raise NotImplementedError(
                "infer_dynamic_shapes is not implemented "
                "when the best candidate is not 'aligned'."
                "This happens when there is not stored set inputs where "
                "all optional inputs showing in other sets are defined."
            )

        if len({inputs.n_aligned_tensors for inputs in self.inputs}) != 1:
            raise NotImplementedError(
                f"infer_dynamic_shapes is not implemented "
                f"when the number of input tensors are not the same in "
                f"every set of inputs "
                f"{[inputs.n_aligned_tensors for inputs in self.inputs]}."
            )
        shape_lists = [
            [(None if t is None else t.shape) for t in candidate.aligned_flat_list]
            for candidate in self.inputs
            if candidate.aligned_flat_list is not None
        ]
        n_tensors = len(shape_lists[0])
        dynamic_shapes = [
            _infer_dynamic_dimensions(
                [s for s in [shapes[index] for shapes in shape_lists] if s is not None],
                set_batch_dimension=_set_batch_dimension_for_flat_index(index),
            )
            for index in range(n_tensors)
        ]
        cst = torch.export.Dim.DYNAMIC
        flat_dynamic_shapes = [dict.fromkeys(dims, cst) for dims in dynamic_shapes]
        if return_flat:
            return tuple(flat_dynamic_shapes)
        if len(flat_dynamic_shapes) == len(self._best_candidate.args) + len(
            self._best_candidate.kwargs
        ):
            # It means forward method is called with tensors only.
            if not self._best_candidate.kwargs:
                # only positional arguments
                return tuple(flat_dynamic_shapes)
            if not self._best_candidate.args:
                # only named arguments
                return dict(zip(list(self._best_candidate.kwargs), flat_dynamic_shapes))
            # positional arguments needs to be moved to the named arguments
            n_args = len(self._best_candidate.args)
            pos_names = self.signature_names[:n_args]
            return {
                **dict(zip(pos_names, flat_dynamic_shapes[:n_args])),
                **dict(zip(list(self._best_candidate.kwargs), flat_dynamic_shapes[n_args:])),
            }

        # nested types, here comes the fun part because the shapes cannot be unflattened,
        # custom classes must appear in their flattened shape.
        # This does not work in all cases but every time every available argument is flattened
        # with the same number of tensors. The function does not check
        # if that assumption is true.
        flat_inputs, _max_spec = torch.utils._pytree.tree_flatten(
            (self._best_candidate.args, self._best_candidate.kwargs)
        )
        torch._check(
            len(flat_inputs) == len(flat_dynamic_shapes),
            (
                f"Length mismatch len(flat_inputs)={len(flat_inputs)}, "
                f"len(flat_dynamic_shapes)={len(flat_dynamic_shapes)}"
            ),
        )

        index = 0

        def change_function(t):
            nonlocal index
            if index >= len(flat_dynamic_shapes):
                raise RuntimeError(
                    f"Flattened {index} tensors when there are only "
                    f"{len(flat_dynamic_shapes)}."
                )
            res = flat_dynamic_shapes[index]
            index += 1
            return res

        ds_args, ds_kwargs = _flatten_unflatten_for_dynamic_shapes(
            (self._best_candidate.args, self._best_candidate.kwargs),
            change_function=change_function,
        )
        if not ds_kwargs:
            return tuple(ds_args)
        if not ds_args:
            return ds_kwargs
        pos_names = self.signature_names[: len(ds_args)]
        return {**dict(zip(pos_names, ds_args)), **ds_kwargs}

    def infer_arguments(
        self, index_or_candidate: InputCandidate | int | None = None, flat: bool = False
    ) -> list[torch.Tensor] | tuple[torch.Tensor, ...] | dict[str, torch.Tensor]:
        """Infers arguments based on the collected tensors."""
        # This is already checked by _build_inputs_completed_with_none_values
        # but this is not always well captured by tools checking types.
        self.align_inputs_none_values()
        torch._check(self._best_candidate is not None, lambda: "No input was captured.")
        # type checking
        assert self._best_candidate is not None
        candidate = None
        if index_or_candidate is None:
            for cand in self.inputs:
                args, kwargs = cand.args, cand.kwargs
                if len(args) == len(self._best_candidate.args) and len(kwargs) == len(
                    self._best_candidate.kwargs
                ):
                    candidate = cand
                    break
        elif isinstance(index_or_candidate, int):
            torch._check(
                index_or_candidate < len(self.inputs),
                lambda: (
                    f"No stored input set for index="
                    f"{index_or_candidate}<{len(self.inputs)}."
                ),
            )
            candidate = self.inputs[index_or_candidate]
        else:
            candidate = index_or_candidate

        torch._check(candidate is not None, "No input was captured.")
        # type checking
        assert candidate is not None
        if candidate.aligned_flat_list is None:
            raise RuntimeError(
                f"Candidate {candidate} has no aligned flat list of tensors, "
                f"index_or_candidate={index_or_candidate}. You should call "
                f"method 'align_with'."
            )

        aligned_flat_list = candidate.aligned_flat_list
        if any(t is None for t in aligned_flat_list):
            dynamic_shapes = self.infer_dynamic_shapes(return_flat=True)
            # type checking
            assert isinstance(dynamic_shapes, tuple)
            aligned_flat_list = aligned_flat_list.copy()
            for index in range(len(aligned_flat_list)):
                if aligned_flat_list[index] is not None:
                    continue
                shape = dynamic_shapes[index]
                all_non_empty_tensors = [
                    c.aligned_flat_list[index]
                    for c in self.inputs
                    if c.aligned_flat_list is not None
                ]
                all_non_empty_tensors_not_none = [
                    t for t in all_non_empty_tensors if t is not None
                ]
                if not all_non_empty_tensors_not_none:
                    raise RuntimeError(
                        f"There is no tensor at position {index} in any flattened inputs."
                    )
                tensor = all_non_empty_tensors_not_none.pop()
                if tensor.numel() == 0:
                    aligned_flat_list[index] = tensor
                    continue
                dim = max(shape)
                torch._check(
                    dim < tensor.ndim,
                    lambda index=index, shape=shape, tshape=tensor.shape: (
                        f"Tensor shape {tshape} does not match the "
                        f"dynamic shape {shape} at position {index}."
                    ),
                )
                new_shape = list(tensor.shape)
                new_shape[dim] = 0
                aligned_flat_list[index] = torch.empty(
                    tuple(new_shape), dtype=tensor.dtype, device=tensor.device
                )
        if flat:
            # type checking
            assert all(t is not None for t in aligned_flat_list)
            # pyrefly: ignore[bad-return]
            return aligned_flat_list
        # type checking
        assert candidate is not None
        assert candidate.aligned_spec is not None
        args, kwargs = torch.utils._pytree.tree_unflatten(
            aligned_flat_list, candidate.aligned_spec
        )
        if not kwargs:
            return args
        if not args:
            return kwargs
        # We need to move args to kwargs
        pos_names = self.signature_names[: len(args)]
        return {**dict(zip(pos_names, args)), **kwargs}


class InputObserver:
    """Steals forward method to collect inputs and outputs.
    This information is used to infer dynamic shapes and
    export arguments.

    Examples
    --------
    >>> input_observer = InputObserver()
    >>> with input_observer(model):
    >>>     model(x1, y1)
    >>>     model(x2, y2)
    >>> ep = torch.export.export(  # or torch.onnx.export
    >>>     model,
    >>>     input_observer.infer_arguments(),
    >>>     dynamic_shapes.input_observer.infer_dynamic_shapes(),
    >>> )

    With LLM:
    >>> input_observer = InputObserver()
    >>> with input_observer(model):
    >>>     model.generate(input_ids)
    >>> ep = torch.export.export(  # or torch.onnx.export
    >>>     model,
    >>>     ()
    >>>     kwargs=input_observer.infer_arguments(),
    >>>     dynamic_shapes.input_observer.infer_dynamic_shapes(),
    >>> )
    """

    def __init__(self):
        self.info: InputObserverInfo | None = None

    def _replaced_method(
        self,
        *args,
        _captured_method: Callable | None = None,
        _store_n_calls: int = 3,
        **kwargs,
    ):
        assert _captured_method is not None, "_captured_forward cannot be None"
        assert self.info is not None, "info cannot be None"
        n_stored = len(self.info)
        if n_stored < _store_n_calls:
            self.info.add_inputs(args, kwargs)
        begin = time.perf_counter()
        res = _captured_method(*args, **kwargs)
        duration = time.perf_counter() - begin
        if n_stored < _store_n_calls:
            self.info.add_outputs(res, latency=duration)
        return res

    @contextlib.contextmanager
    def __call__(
        self, model: torch.nn.Module, store_n_calls: int = 3, method_name: str = "forward"
    ):
        """Starts collecting inputs and outputs of a specific method.
        The model method is replaced by a new one collecting tensors
        before and after the inner one is called.
        The original method is restored after the collection.

        Args:
            model: Model
            store_n_calls: The collection stops after this many calls
                to avoid taking too much memory.
            method_name: Method name to spy on.
        """
        if self.info is not None:
            raise RuntimeError(
                "This class was already used to capture a model. Please create a new one."
            )
        if not hasattr(model, method_name):
            raise ValueError(f"Model type {model} does not have a method {method_name!r}")
        captured_method = getattr(model, method_name)
        self.info = InputObserverInfo(
            signature_names=list(inspect.signature(captured_method).parameters)
        )
        setattr(
            model,
            method_name,
            lambda *args, _cm=captured_method, _snc=store_n_calls, **kwargs: self._replaced_method(  # noqa: E501
                *args,
                _captured_method=_cm,
                _store_n_calls=_snc,
                **kwargs,
            ),
        )
        try:
            yield self
        finally:
            setattr(model, method_name, captured_method)

    def _check_captured(self):
        if self.info is None:
            raise RuntimeError("No inputs were captured.")

    def infer_dynamic_shapes(
        self, set_batch_dimension_for: set[int | str] | bool | None = None
    ) -> tuple[dict[int, Any], ...] | dict[str, dict[int, Any]]:
        """
        Infers dynamic shapes. Most of the time, models do support a batch dimension
        but this batch dimension has the same value for every input sample.
        Instead of running inference on new samples, argument `set_batch_dimension_for`
        can be used to tell the first dimension is a dynamic dimension for a particular
        set of inputs referenced by their name (str) or their position (int).
        """
        self._check_captured()
        assert self.info is not None  # missed by type checking
        return self.info.infer_dynamic_shapes(set_batch_dimension_for=set_batch_dimension_for)

    def infer_arguments(
        self,
        index_or_args_or_kwargs: tuple[Any] | dict[str, Any] | int | None = None,
        flat: bool = False,
    ) -> list[torch.Tensor] | tuple[torch.Tensor, ...] | dict[str, torch.Tensor]:
        """Infers arguments based on the collected tensors.

        Args:
            index_or_args_or_kwargs: If missing, the method selects one set of inputs
                among the available ones, usually this inputs containing
                the set of stored inputs with the highest number of tensors.
                The then replaces None values and missing tensors by empty tensors.
                If not missing, it can be an integer to fetch one of the stored set
                or some inputs.
            flat: If True, it returns a flattened list of tensors,
                if False, it returns a tuple or a dictionary preserving
                the nested structures.

        Returns:
            Inferred arguments, every optional tensor is replaced by a empty tensor.
        """
        self._check_captured()
        assert self.info is not None  # missed by type checking
        index_or_candidate: int | InputCandidate | None = None
        if index_or_args_or_kwargs is None or isinstance(index_or_args_or_kwargs, int):
            index_or_candidate = index_or_args_or_kwargs
        else:
            if isinstance(index_or_args_or_kwargs, tuple):
                index_or_candidate = InputCandidate(
                    args=index_or_args_or_kwargs, kwargs={}, clone=False
                )
            elif isinstance(index_or_args_or_kwargs, dict):
                index_or_candidate = InputCandidate(
                    args=(), kwargs=index_or_args_or_kwargs, clone=False
                )
            else:
                raise ValueError(
                    f"Unexpected type {type(index_or_args_or_kwargs)} "
                    f"for index_or_args_or_kwargs."
                )
            self.info.align_inputs_none_values()
            # type checking
            assert self.info._best_candidate is not None
            assert self.info._captured_inputs is not None
            index_or_candidate.align_with(
                self.info._best_candidate, self.info._captured_inputs
            )
        return self.info.infer_arguments(index_or_candidate=index_or_candidate, flat=flat)

    def check_discrepancies(
        self,
        onnx_model: str | onnx.ModelProto,
        atol: float = 1e-4,
        rtol: float = 0.1,
        hist=(0.1, 0.01),
        progress_bar: bool = False,
    ) -> list[dict[str, str | int | float]]:
        """Computes the discrepancies between the saved inputs and outputs
        with the saved onnx model.

        Args:
            onnx_model: ONNX Model to verify.
            atol: Absolute tolerance, recommended values, 1e-4 for float, 1e-2 flot float16.
            rtol: Relative tolerance.
            hist: Thresholds, the function determines the number of discrepancies
                above these thresholds.
            progress_bar: Shows a progress bar (requires :epkg:`tqdm`).

        Returns:
            A list of dictionaries, ready to be consumed by a dataframe.
        """
        sess = OnnxruntimeEvaluator(onnx_model, whole=True)
        input_names = sess.input_names
        self._check_captured()
        # type checking
        assert self.info is not None
        assert self.info.inputs is not None
        assert self.info.flat_outputs is not None
        assert self.info.latencies is not None
        io_sets = list(zip(self.info.inputs, self.info.flat_outputs, self.info.latencies))
        if progress_bar:
            from tqdm import tqdm

            loop = tqdm(io_sets)
        else:
            loop = io_sets
        lhist = list(hist)
        data: list[dict[str, Any]] = []
        for inputs, outputs, latency in loop:
            # type checking
            assert inputs.aligned_flat_list is not None
            if len(input_names) != len(inputs.aligned_flat_list):
                raise RuntimeError(
                    f"There are ({len(inputs.aligned_flat_list)}) "
                    f"tensors but the model expects {len(input_names)}."
                )

            feeds = dict(zip(input_names, self.info.infer_arguments(inputs, flat=True)))

            begin = time.perf_counter()
            ort_outputs = sess.run(None, feeds)
            duration = time.perf_counter() - begin
            diff = max_diff(outputs, ort_outputs, hist=lhist)
            if "rep" in diff and isinstance(diff["rep"], dict):
                diff.update(diff["rep"])
                del diff["rep"]
            diff["SUCCESS"] = (
                isinstance(diff["abs"], float)
                and isinstance(diff["rel"], float)
                and diff["abs"] < atol
                and diff["rel"] < rtol
            )
            diff.update(
                dict(
                    index=len(diff),
                    duration_torch=latency,
                    ort_duration=duration,
                    n_inputs=len(input_names),
                )
            )
            data.append(diff)
        return data
