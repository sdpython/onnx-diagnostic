import inspect
import os
import textwrap
import time
from collections.abc import Mapping, Iterable
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union
import torch
from .dynamic_shapes import ModelInputs
from .onnx_plug import EagerDirectReplacementWithOnnx
from ..helpers import flatten_object, max_diff, string_diff, string_type
from ..helpers.torch_helper import torch_deepcopy
from ..helpers.rt_helper import make_feeds
from ..reference import OnnxruntimeEvaluator


def get_main_dispatcher(
    use_control_flow_dispatcher: bool = False,
    onnx_plugs: Optional[List[EagerDirectReplacementWithOnnx]] = None,
) -> Any:  # Dispatcher
    """Creates a custom dispatcher for the custom exporter."""
    from experimental_experiment.torch_interpreter import Dispatcher

    if use_control_flow_dispatcher:
        from .control_flow_onnx import create_global_dispatcher

        control_flow_dispatcher = create_global_dispatcher()
    else:
        control_flow_dispatcher = None

    class MainDispatcher(Dispatcher):
        def __init__(self, previous_dispatcher=None):
            super().__init__({})
            self.previous_dispatcher = previous_dispatcher

        @property
        def supported(self):
            if self.previous_dispatcher:
                return set(self.registered_functions) | self.previous_dispatcher.supported
            return set(self.registered_functions)

        def find_function(self, name: Any):
            if self.previous_dispatcher:
                find = self.previous_dispatcher.find_function(name)
                if find:
                    return find
            return Dispatcher.find_function(self, name)

        def find_method(self, name: Any):
            if self.previous_dispatcher:
                find = self.previous_dispatcher.find_method(name)
                if find:
                    return find
            return Dispatcher.find_method(self, name)

    main_dispatcher = MainDispatcher(control_flow_dispatcher)
    if onnx_plugs:
        for plug in onnx_plugs:
            main_dispatcher.registered_functions[plug.target_name] = plug.custom_converter()
    return main_dispatcher


def to_onnx(
    mod: Union["torch.nn.Module", "torch.fx.GraphModule"],  # noqa: F821
    args: Optional[Sequence["torch.Tensor"]] = None,  # noqa: F821
    kwargs: Optional[Dict[str, "torch.Tensor"]] = None,  # noqa: F821
    input_names: Optional[Sequence[str]] = None,
    target_opset: Optional[Union[int, Dict[str, int]]] = None,
    verbose: int = 0,
    dynamic_shapes: Optional[Union[Dict[str, Any], Tuple[Any]]] = None,
    filename: Optional[str] = None,
    output_names: Optional[List[str]] = None,
    output_dynamic_shapes: Optional[Union[Dict[str, Any], Tuple[Any]]] = None,
    exporter: str = "onnx-dynamo",
    exporter_kwargs: Optional[Dict[str, Any]] = None,
    save_ep: Optional[str] = None,
    optimize: bool = True,
    optimizer_for_ort: bool = True,
    use_control_flow_dispatcher: bool = False,
    onnx_plugs: Optional[List[EagerDirectReplacementWithOnnx]] = None,
    inline: bool = True,
) -> Any:
    """
    Exports one model into ONNX.
    Common API for exporters. By default, the models are optimized to use the
    most efficient kernels implemented in :epkg:`onnxruntime`.

    :param mod: torch model
    :param args: unnamed arguments
    :param kwargs: named arguments
    :param input_names: input names for the onnx model (optional)
    :param target_opset: opset to target, if not specified, each converter
        keeps its default value
    :param verbose: verbosity level
    :param dynamic_shapes: dynamic shapes, usually a nested structure
        included a dictionary for each tensor
    :param filename: output filename
    :param output_names: to change the output of the onnx model
    :param output_dynamic_shapes: to overwrite the dynamic shapes names
    :param exporter: exporter to use (``onnx-dynamo``, ``modelbuilder``, ``custom``)
    :param exporter_kwargs: additional parameters sent to the exporter
    :param save_ep: saves the exported program
    :param optimize: optimizes the model
    :param optimizer_for_ort: optimizes the model for onnxruntime
    :param use_control_flow_dispatcher: use the dispatcher created to supported
        custom loops (see :func:`onnx_diagnostic.export.control_flow_onnx.loop_for_onnx`)
    :param onnx_plugs: the code was modified to replace some parts with onnx translation
    :param inline: inline local functions
    :return: the output of the selected exporter, usually a structure including
        an onnx model

    A simple example:

    .. code-block:: python

        to_onnx(
            model,
            kwargs=inputs,
            dynamic_shapes=ds,
            exporter=exporter,
            filename=filename,
        )

    Some examples using control flows are available in
    :func:`onnx_diagnostic.export.control_flow_onnx.loop_for_onnx` or
    :class:`onnx_diagnostic.export.onnx_plug.EagerDirectReplacementWithOnnx`.
    """
    if exporter_kwargs and "inline" in exporter_kwargs:
        assert (
            inline == exporter_kwargs["inline"]
        ), f"Mismatch between inline={inline} and exporter_kwargs={exporter_kwargs}"
        exporter_kwargs.pop("inline")
    if exporter == "custom":
        from experimental_experiment.torch_interpreter import (
            to_onnx as _to_onnx,
            ExportOptions,
        )
        from experimental_experiment.xbuilder import OptimizationOptions

        options = None
        export_options = None
        if exporter_kwargs is not None:
            options = exporter_kwargs.pop("options", None)
            export_options = exporter_kwargs.pop("export_options", None)
        if export_options is None:
            export_options = ExportOptions(save_ep=save_ep)
        if options is None and optimize:
            options = OptimizationOptions(
                patterns="default+onnxruntime" if optimizer_for_ort else "default"
            )
        main_dispatcher = (
            get_main_dispatcher(use_control_flow_dispatcher, onnx_plugs)
            if onnx_plugs or use_control_flow_dispatcher
            else None
        )

        proto, opt_stats = _to_onnx(
            mod,
            args=args,
            kwargs=kwargs,
            input_names=input_names,
            output_names=output_names,
            target_opset=target_opset,
            verbose=verbose,
            filename=filename,
            dynamic_shapes=dynamic_shapes,
            large_model=True,
            output_dynamic_shapes=output_dynamic_shapes,
            export_options=export_options,
            options=options,
            inline=inline,
            dispatcher=main_dispatcher,
            optimize=optimize,
            return_optimize_report=True,
            **(exporter_kwargs or {}),
        )
        if opt_stats and filename and os.path.exists(filename):
            import pandas

            stat_filename = f"{os.path.splitext(filename)[0]}.opt.xlsx"
            pattern_stats = []
            for k, v in opt_stats.items():
                if "time" in k:
                    pattern_stats.append(dict(level="main", pattern=k, time_in=v))
            pattern_stats.extend(
                [{**obs, "level": "detailed"} for obs in opt_stats["optimization"]]
            )
            df = pandas.DataFrame(pattern_stats)
            df.to_excel(stat_filename, index=False)
            cols = [
                c
                for c in [
                    "level",
                    "pattern",
                    "time_in",
                    "iteration",
                    "inlined",
                    "removed",
                    "added",
                    "instances",
                    "changed",
                    "scale",
                ]
                if c in df.columns
            ]
            agg = {k: "sum" for k in cols if k not in ("level", "pattern")}
            agg.update(dict(iteration="max", instances="mean"))
            agg = {k: v for k, v in agg.items() if k in df.columns}
            stat_filename = f"{os.path.splitext(filename)[0]}.opt.agg.xlsx"
            df[cols].groupby(["level", "pattern"]).agg(agg).to_excel(stat_filename)

        return proto

    if exporter in ("dynamo", "onnx-dynamo"):
        from ..helpers import flatten_object
        import onnxscript.rewriter.ort_fusions as ort_fusions

        assert (
            not output_dynamic_shapes
        ), f"output_dynamic_shapes not supported for exporter={exporter!r}"
        assert (
            optimize
        ), f"torch.onnx.export always optimizes the model but optimize={optimize}"
        custom_translation_table = {}
        if onnx_plugs:
            for plug in onnx_plugs:
                custom_translation_table[plug.torch_op] = plug.onnx_dynamo_converter()
        epo = torch.onnx.export(
            mod,
            args=args or tuple(),
            kwargs=kwargs,
            input_names=input_names,
            output_names=output_names,
            opset_version=target_opset,
            dynamic_shapes=dynamic_shapes,
            dynamo=True,
            verbose=verbose,
            dump_exported_program=bool(save_ep),
            artifacts_dir=os.path.dirname(filename) if filename else ".",
            custom_translation_table=custom_translation_table,
            **(exporter_kwargs or {}),
        )
        if not inline and optimize and optimizer_for_ort:
            ort_fusions.optimize_for_ort(epo.model)

        if onnx_plugs:
            import onnx_ir as ir
            import onnx_ir.passes.common as common_passes

            opset = (
                18
                if target_opset is None
                else (target_opset if isinstance(target_opset, int) else target_opset[""])
            )

            irfunctions = [
                ir.from_proto(
                    plug.get_function_proto(
                        opset, *flatten_object((args, kwargs), drop_keys=True)
                    )
                )
                for plug in onnx_plugs
            ]
            for func in irfunctions:
                epo.model.functions[func.identifier()] = func
            if inline:
                common_passes.InlinePass()(epo.model)
                common_passes.RemoveUnusedOpsetsPass()(epo.model)

        if inline and optimize and optimizer_for_ort:
            ort_fusions.optimize_for_ort(epo.model)
        if filename:
            epo.save(filename, external_data=True)
        if save_ep:
            if isinstance(save_ep, tuple):
                save_ep = save_ep[0]
            torch.export.save(epo.exported_program, f"{save_ep}.pt2")
        return epo

    if exporter == "modelbuilder":
        from ..helpers import flatten_object, string_type
        from ..helpers.model_builder_helper import create_model_builder, save_model_builder

        assert filename, f"filename must be specified for exporter={exporter!r}"
        assert (
            not output_dynamic_shapes
        ), f"output_dynamic_shapes not supported for exporter={exporter!r}"
        assert hasattr(mod, "config"), f"configuration is missing in model class {type(mod)}"
        assert not args, f"only kwargs can be defined with exporter={exporter!r}"
        assert list(kwargs) == ["input_ids", "attention_mask", "past_key_values"], (  # type: ignore[arg-type]
            f"Only a specified set of inputs is supported for exporter={exporter!r}, "
            f"but it is {list(kwargs)}"  # type: ignore[arg-type]
        )
        assert optimizer_for_ort and optimize, (
            f"ModelBuilder only produces model optimized for onnxruntime but "
            f"optimizer_for_ort={optimizer_for_ort} and optimize={optimize}"
        )
        flat_inputs = flatten_object(kwargs, drop_keys=True)
        first = flat_inputs[0]
        first_float = [
            t
            for t in flat_inputs
            if t.dtype in {torch.float32, torch.double, torch.float16, torch.bfloat16}
        ]
        assert first_float, (
            f"Unable to find a float tensor in the inputs "
            f"{string_type(kwargs, with_shape=True)}"
        )
        onx = create_model_builder(
            mod.config,
            mod,
            precision=str(first_float[0].dtype).split(".")[-1],
            execution_provider="cuda" if first.is_cuda else "cpu",
            cache_dir=os.path.dirname(filename),
            **(exporter_kwargs or {}),
        )
        save_model_builder(onx, os.path.dirname(filename))
        return onx

    raise ValueError(f"Unknown exporter={exporter!r}")


class WrapperToExportMethodToOnnx(torch.nn.Module):
    """
    Wraps an existing models in order to spy on inputs.
    This is used by :func:`onnx_diagnostic.export.api.method_to_onnx`
    or :ref:`l-plot-tiny-llm-export-method-generate` for an example.
    """

    def __init__(
        self,
        mod: "torch.nn.Module",
        method_name: str = "forward",
        input_names: Optional[Sequence[str]] = None,
        target_opset: Optional[Union[int, Dict[str, int]]] = None,
        verbose: int = 0,
        filename: Optional[str] = None,
        output_names: Optional[List[str]] = None,
        output_dynamic_shapes: Optional[Union[Dict[str, Any], Tuple[Any]]] = None,
        exporter: str = "onnx-dynamo",
        exporter_kwargs: Optional[Dict[str, Any]] = None,
        save_ep: Optional[str] = None,
        optimize: bool = True,
        optimizer_for_ort: bool = True,
        use_control_flow_dispatcher: bool = False,
        onnx_plugs: Optional[List[EagerDirectReplacementWithOnnx]] = None,
        inline: bool = True,
        convert_after_n_calls: int = 2,
        patch_kwargs: Optional[Dict[str, Any]] = None,
        skip_kwargs_names: Optional[Set[str]] = None,
        dynamic_shapes: Optional[Union[Dict[str, Any], Tuple[Any]]] = None,
    ):
        super().__init__()
        self._model_to_call = mod
        self._method_name = method_name
        self._method_call = (
            self._model_to_call.forward
            if method_name == "forward"
            else getattr(mod, method_name)
        )
        self._inputs: List[Tuple[Tuple[Any, ...], Dict[str, Any]]] = []
        self._outputs: List[Any] = []
        self._convert_after_n_calls = convert_after_n_calls
        self._patch_kwargs = patch_kwargs
        self._method_src = None
        self.verbose = verbose
        self.skip_kwargs_names = skip_kwargs_names
        self.dynamic_shapes = dynamic_shapes
        self._to_onnx_kwargs = dict(
            input_names=input_names,
            target_opset=target_opset,
            verbose=verbose,
            filename=filename,
            output_names=output_names,
            output_dynamic_shapes=output_dynamic_shapes,
            exporter=exporter,
            exporter_kwargs=exporter_kwargs,
            save_ep=save_ep,
            optimize=optimize,
            optimizer_for_ort=optimizer_for_ort,
            use_control_flow_dispatcher=use_control_flow_dispatcher,
            onnx_plugs=onnx_plugs,
            inline=inline,
        )
        self._export_done = False
        self._serialization_classes: Set[type] = set()

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self._model_to_call.__class__.__name__}."
            f"{self._method_name})"
        )

    def _collect_classes(self, obj):
        if obj is None or isinstance(obj, torch.Tensor):
            return
        cls = type(obj)
        if cls.__module__ not in ("builtins",):
            self._serialization_classes.add(cls)
        if hasattr(obj, "__dict__"):
            for v in vars(obj).values():
                self._collect_classes(v)
            return
        if isinstance(obj, Mapping):
            for v in obj.values():
                self._collect_classes(v)
            return
        if isinstance(obj, Iterable) and not isinstance(obj, (str, bytes)):
            for v in obj:
                self._collect_classes(v)
            return

    def forward(self, *args, **kwargs):
        if not self._export_done:
            self._inputs.append(
                torch_deepcopy(
                    (
                        args,
                        (
                            kwargs
                            if not kwargs or not self.skip_kwargs_names
                            else {
                                k: v
                                for k, v in kwargs.items()
                                if k not in self.skip_kwargs_names
                            }
                        ),
                    )
                )
            )
            if self.verbose:
                print(
                    f"[method_to_onnx] input[{len(self._inputs)-1}]: "
                    f"{string_type(self._inputs[-1], with_shape=True)}"
                )
            if len(self._inputs) >= self._convert_after_n_calls:
                name = os.path.splitext(self._to_onnx_kwargs["filename"])[0]
                input_file = f"{name}.inputs.pt"
                self._input_file = input_file
                if self.verbose:
                    print(
                        f"[method_to_onnx] save {len(self._inputs)} inputs in {input_file!r}"
                    )
                torch.save(self._inputs, input_file)
                self._convert_method_to_onnx()
                self._export_done = True

        begin = time.perf_counter()
        res = self._method_call(*args, **kwargs)
        duration = time.perf_counter() - begin
        self._collect_classes([args, kwargs, res])
        if self._inputs:
            self._outputs.append((torch_deepcopy(res), duration))
            assert len(self._inputs) == len(self._outputs), (
                f"Number of inputs {len(self._inputs)} and "
                f"outputs {len(self._outputs)} are different."
            )
            if self._export_done:
                name = os.path.splitext(self._to_onnx_kwargs["filename"])[0]
                output_file = f"{name}.outputs.pt"
                if self.verbose:
                    print(
                        f"[method_to_onnx] save {len(self._outputs)} "
                        f"outputs in {output_file!r}"
                    )
                torch.save(self._outputs, output_file)
                self._output_file = output_file
                del self._inputs[:]
                del self._outputs[:]
        return res

    def _convert_method_to_onnx(self):
        for args, kwargs in self._inputs:
            self._serialization_classes |= {type(a) for a in args}
            self._serialization_classes |= {type(a) for a in kwargs.values()}

        def make_method(self):
            inner_sig = inspect.signature(self._method_call)
            params = [
                p.replace(annotation=inspect._empty) for p in inner_sig.parameters.values()
            ]
            simple_sig = inspect.Signature(params, return_annotation=inspect._empty)
            args = str(simple_sig)[1:-1]
            calls_args = ", ".join(f"{p}={p}" for p in simple_sig.parameters)
            src = textwrap.dedent(
                f"""
                def f(self, {args}):
                    return self._method_call({calls_args})
                """
            )
            self._method_src = src
            ns = {}
            try:
                exec(src, ns)
            except NameError as e:
                raise NameError(f"Unable to compile due to {e}\n{src}") from e
            return ns["f"]

        class WrapWithExactSignature(torch.nn.Module):
            def __init__(self, parent):
                super().__init__()
                self._model_to_call = parent._model_to_call
                self._method_call = parent._method_call

            forward = make_method(self)

        compiled_model = WrapWithExactSignature(self)

        if self.dynamic_shapes is None:
            mi = ModelInputs(compiled_model, self._inputs)
            ds = mi.guess_dynamic_shapes()
            if self.verbose:
                print(f"[method_to_onnx] guess_dynamic_shapes={string_type(ds)}")
            a, kw, nds = mi.move_to_kwargs(*self._inputs[-1], ds)
        else:
            a, kw = self._inputs[-1]
            nds = [self.dynamic_shapes]
        if self.verbose:
            print(f"[method_to_onnx] export args={string_type(a, with_shape=True)}")
            print(f"[method_to_onnx] export kwargs={string_type(kw, with_shape=True)}")
            print(f"[method_to_onnx] dynamic_shapes={string_type(nds)}")
        if self._patch_kwargs is None:
            to_onnx(
                compiled_model,
                args=a,
                kwargs=kw,
                dynamic_shapes=nds[-1],
                **self._to_onnx_kwargs,
            )
            return
        from ..torch_export_patches import torch_export_patches

        with torch_export_patches(**self._patch_kwargs):
            to_onnx(
                compiled_model,
                args=a,
                kwargs=kw,
                dynamic_shapes=nds[-1],
                **self._to_onnx_kwargs,
            )

    def check_discrepancies(
        self, atol: float = 1e-4, rtol: float = 0.1, hist=(0.1, 0.01), verbose: int = 0
    ) -> List[Dict[str, Union[str, int, float]]]:
        """
        Computes the discrepancies between the saved inputs and outputs
        with the saved onnx model.

        :param atol: absolute tolerance, recommended values, 1e-4 for float, 1e-2 flot float16
        :param rtol: relative tolerance
        :param hist: thresholds, the function determines the number of discrepancies
            above that threshold.
        :param verbose: verbosity
        :return: results, a list of dictionaries, ready to be consumed by a dataframe
        """
        assert self._export_done, "The onnx export was not done."
        assert os.path.exists(self._input_file), f"input file {self._input_file!r} not found"
        assert os.path.exists(
            self._output_file
        ), f"output file {self._output_file!r} not found"
        filename = self._to_onnx_kwargs["filename"]
        assert isinstance(filename, str) and os.path.exists(
            filename
        ), f"onnx file {filename!r} not found"
        classes = [
            cls
            for cls in self._serialization_classes
            if cls not in {int, float, bool, str, torch.Tensor, list, set, dict, torch.device}
        ]
        if verbose:
            print(f"[method_to_onnx.check_discrepancies] register classes {classes}")
            print(f"[method_to_onnx.check_discrepancies] load {self._input_file!r}")
        with torch.serialization.safe_globals(classes):
            inputs = torch.load(self._input_file)
        if verbose:
            print(f"[method_to_onnx.check_discrepancies] load {self._output_file!r}")
        with torch.serialization.safe_globals(classes):
            outputs = torch.load(self._output_file)
        assert len(inputs) == len(outputs), (
            f"Unexpected number of inputs {len(inputs)} and outputs {len(outputs)}, "
            f"inputs={string_type(inputs, with_shape=True)}, "
            f"outputs={string_type(outputs, with_shape=True)}"
        )
        if verbose:
            print(f"[method_to_onnx.check_discrepancies] create onnx session {filename!r}")
        sess = OnnxruntimeEvaluator(filename, whole=True)
        input_names = sess.input_names
        if verbose:
            print(f"[method_to_onnx.check_discrepancies] input_names={input_names}")
        data = []
        for i, (input, (output, latency)) in enumerate(zip(inputs, outputs)):
            if verbose:
                if verbose > 1:
                    print(
                        f"[method_to_onnx.check_discrepancies] process input {i}: "
                        f"{string_type(input, with_shape=True)}"
                    )
                    print(
                        f"[method_to_onnx.check_discrepancies] expects: "
                        f"{string_type(output, with_shape=True)}"
                    )
                else:
                    print(f"[method_to_onnx.check_discrepancies] process input {i}")

            flat_inputs = flatten_object(input, drop_keys=True)
            if len(flat_inputs) < len(input_names):
                # not implemented yet, it is caused by a missing cache,
                # which requires an empty cache instead
                data.append(dict(index=i, duration_torch=latency, n_inputs=len(flat_inputs)))
                continue
            assert len(flat_inputs) == len(input_names), (
                f"Length mismatch, expecting {len(input_names)} onnx inputs and got "
                f"{len(flat_inputs)} flat torch inputs"
            )
            feeds = make_feeds(input_names, flat_inputs)
            begin = time.perf_counter()
            ort_outputs = sess.run(None, feeds)
            duration = time.perf_counter() - begin
            diff = max_diff(output, ort_outputs, hist=hist)
            if "rep" in diff and isinstance(diff["rep"], dict):
                diff.update(diff["rep"])
                del diff["rep"]
            diff["SUCCESS"] = (
                isinstance(diff["abs"], float)
                and isinstance(diff["ref"], float)
                and diff["abs"] < atol
                and diff["rel"] < rtol
            )
            diff.update(
                dict(
                    index=i,
                    duration_torch=latency,
                    ort_duration=duration,
                    n_inputs=len(flat_inputs),
                )
            )
            if verbose > 1:
                print(
                    f"[method_to_onnx.check_discrepancies] ort output "
                    f"{string_type(ort_outputs, with_shape=True)}"
                )
                print(f"[method_to_onnx.check_discrepancies] diff {string_diff(diff)}")
            data.append(diff)
        if verbose:
            print("[method_to_onnx.check_discrepancies] done")
        return data


def method_to_onnx(
    mod: "torch.nn.Module",
    method_name: str = "forward",
    input_names: Optional[Sequence[str]] = None,
    target_opset: Optional[Union[int, Dict[str, int]]] = None,
    verbose: int = 0,
    filename: Optional[str] = None,
    output_names: Optional[List[str]] = None,
    output_dynamic_shapes: Optional[Union[Dict[str, Any], Tuple[Any]]] = None,
    exporter: str = "onnx-dynamo",
    exporter_kwargs: Optional[Dict[str, Any]] = None,
    save_ep: Optional[str] = None,
    optimize: bool = True,
    optimizer_for_ort: bool = True,
    use_control_flow_dispatcher: bool = False,
    onnx_plugs: Optional[List[EagerDirectReplacementWithOnnx]] = None,
    inline: bool = True,
    convert_after_n_calls: int = 2,
    patch_kwargs: Optional[Dict[str, Any]] = None,
    skip_kwargs_names: Optional[Set[str]] = None,
    dynamic_shapes: Optional[Union[Dict[str, Any], Tuple[Any]]] = None,
) -> Callable:
    """
    Exports one method into ONNX for a module into ONNX.
    It returns a new method which must be called by the user
    at least twice with different values for the dynamic dimension
    between triggering the conversion into ONNX.

    :param mod_meth: function to export into ONNX
    :param input_names: input names for the onnx model (optional)
    :param target_opset: opset to target, if not specified, each converter
        keeps its default value
    :param verbose: verbosity level
    :param filename: output filename, mandatory, the onnx model is saved on disk
    :param output_names: to change the output of the onnx model
    :param output_dynamic_shapes: to overwrite the dynamic shapes names
    :param exporter: exporter to use (``onnx-dynamo``, ``modelbuilder``, ``custom``)
    :param exporter_kwargs: additional parameters sent to the exporter
    :param save_ep: saves the exported program
    :param optimize: optimizes the model
    :param optimizer_for_ort: optimizes the model for onnxruntime
    :param use_control_flow_dispatcher: use the dispatcher created to supported
        custom loops (see :func:`onnx_diagnostic.export.control_flow_onnx.loop_for_onnx`)
    :param onnx_plugs: the code was modified to replace some parts with onnx translation
    :param inline: inline local functions
    :param convert_after_n_calls: converts the model after this number of calls.
    :param patch_kwargs: patch arguments
    :param skip_kwargs_names: use default values for these parameters part of
        the signature of the method to export
    :param dynamic_shapes: dynamic shapes to use if the guessed ones are not right
    :return: the output of the selected exporter, usually a structure including
        an onnx model

    See :ref:`l-plot-tiny-llm-export-method-generate` for an example.
    """
    wrapped_model = WrapperToExportMethodToOnnx(
        mod=mod,
        method_name=method_name,
        input_names=input_names,
        target_opset=target_opset,
        verbose=verbose,
        filename=filename,
        output_names=output_names,
        output_dynamic_shapes=output_dynamic_shapes,
        exporter=exporter,
        exporter_kwargs=exporter_kwargs,
        save_ep=save_ep,
        optimize=optimize,
        optimizer_for_ort=optimizer_for_ort,
        use_control_flow_dispatcher=use_control_flow_dispatcher,
        onnx_plugs=onnx_plugs,
        inline=inline,
        convert_after_n_calls=convert_after_n_calls,
        patch_kwargs=patch_kwargs,
        skip_kwargs_names=skip_kwargs_names,
        dynamic_shapes=dynamic_shapes,
    )
    return wrapped_model
