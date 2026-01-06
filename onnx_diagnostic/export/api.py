import inspect
import os
import textwrap
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import torch
from .dynamic_shapes import ModelInputs
from .onnx_plug import EagerDirectReplacementWithOnnx
from ..helpers import string_type


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


class _WrapperToExportMethodToOnnx(torch.nn.Module):
    """
    Wraps an existing models in order to spy on inputs.
    This is used by :func:`onnx_diagnostic.export.api.method_to_onnx`.
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
    ):
        super().__init__()
        self._model_to_call = mod
        self._method_name = method_name
        self._call = (
            self._model_to_call if method_name == "forward" else getattr(mod, method_name)
        )
        self._inputs: List[Tuple[Tuple[Any, ...], Dict[str, Any]]] = []
        self._convert_after_n_calls = convert_after_n_calls
        self._patch_kwargs = patch_kwargs
        self._method_src = None
        self.verbose = verbose
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

    def forward(self, *args, **kwargs):
        self._inputs.append((args, kwargs))
        if self.verbose:
            print(
                f"[method_to_onnx] input{len(self._inputs)}: "
                f"{string_type((args, kwargs), with_shape=True)}"
            )
        if len(self._inputs) >= self._convert_after_n_calls:
            self._convert_method_to_onnx()
        return self._call(*args, **kwargs)

    def _convert_method_to_onnx(self):

        def make_method(self):
            sig = inspect.signature(getattr(self._model_to_call, self._method_name))
            args = str(sig)[1:-1]
            calls_args = ", ".join(f"{p}={p}" for p in sig.parameters)
            src = textwrap.dedent(
                f"""
                def f(self, {args}):
                    return self._call({calls_args})
                """
            )
            self._method_src = src
            ns = {}
            exec(src, ns)
            return ns["f"]

        class WrapWithExactSignature(torch.nn.Module):
            def __init__(self, parent):
                super().__init__()
                self._model_to_call = parent._model_to_call
                self._call = parent._call

            forward = make_method(self)

        compiled_model = WrapWithExactSignature(self)
        mi = ModelInputs(compiled_model, self._inputs)
        ds = mi.guess_dynamic_shapes()
        if self.verbose:
            print(f"[method_to_onnx] guess_dynamic_shapes={string_type(ds)}")
        a, kw, nds = mi.move_to_kwargs(*self._inputs[-1], ds)
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
    :return: the output of the selected exporter, usually a structure including
        an onnx model
    """
    wrapped_model = _WrapperToExportMethodToOnnx(
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
    )
    return wrapped_model
