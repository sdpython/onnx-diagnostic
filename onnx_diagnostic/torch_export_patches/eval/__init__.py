import contextlib
import io
import itertools
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import onnx


def discover():
    """
    Discovers all model cases used to evaluate an exporter.

    .. runpython::
        :showcode:

        import pprint
        from onnx_diagnostic.torch_export_patches.eval import discover

        pprint.pprint(discover())
    """
    from . import model_cases

    res = {}
    for m in model_cases.__dict__.values():
        if m is None or isinstance(m, str):
            continue
        if not hasattr(m, "forward"):
            continue
        assert m.__name__ not in res, f"Case {m.__name__!r} is duplicated."
        assert hasattr(m, "_inputs"), f"Attribute '_inputs' is missing from class {m}"
        assert hasattr(m, "_dynamic"), f"Attribute '_dynamic' is missing from class {m}"
        res[m.__name__] = m
    return res


def evaluation(
    exporters: Tuple[str] = (
        "export-strict",
        "export-nostrict",
        "export-nostrict-decall",
        "export-strict-oblivious",
        "export-nostrict-oblivious",
        "export-nostrict-decall-oblivious",
    ),
    dynamic: Tuple[bool] = (False, True),
    cases: Optional[Union[str, Dict[str, type]]] = None,
    verbose: int = 0,
    quiet: bool = True,
) -> List[Dict[str, Any]]:
    """
    Evaluates exporter for a list of cases.

    :param exporters: exporters to evaluate
    :param dynamic: evaluate static shape and dynamic shapes
    :param cases: model cases to evaluate
    :param verbose: verbosity
    :param quiet: catch exception
    :return: results, list of dictionaries
    """
    if isinstance(exporters, str):
        exporters = (exporters,)
    if isinstance(dynamic, (bool, int)):
        dynamic = (dynamic,)

    if cases is None:
        cases = discover()
    elif cases in ("three", ["three"]):
        all_cases = discover()
        cases = dict(list(all_cases.items())[:3])
    elif isinstance(cases, str):
        cases = (cases,)

    if isinstance(cases, (list, tuple)):
        all_cases = discover()
        new_cases = []  # type: ignore[var-annotated]
        for c in cases:
            if "*" in c or "?" in c:
                # regex
                reg = re.compile(c)
                new_cases.extend(k for k in all_cases if reg.match(k))
            else:
                new_cases.append(c)
        cases = {k: v for k, v in all_cases.items() if k in set(new_cases)}

    sorted_cases = sorted(cases.items())
    loop = list(itertools.product(sorted_cases, dynamic, exporters))
    if verbose:
        try:
            import tqdm

            loop = tqdm.tqdm(loop)
        except ImportError:

            def _loop():
                for _ in loop:
                    print(f"[evaluation] {_}")
                    yield _

    assert len(loop) > 0, f"No case to test for cases={cases!r}."
    obs = []
    for case, dyn, exporter in loop:
        name, cls_model = case
        res = run_exporter(exporter, cls_model, dyn, quiet=quiet, verbose=max(0, verbose - 1))
        res.update(dict(name=name, dynamic=int(dyn), exporter=exporter))
        obs.append(res)
    return obs


def _flatten_inputs(x: Any) -> List["torch.Tensor"]:  # noqa: F821
    """Flatten inputs."""
    if x is None:
        return x
    import torch

    if isinstance(x, (list, tuple)):
        res = []
        for i in x:
            if i is None or isinstance(
                i,
                (
                    torch.Tensor,
                    torch.SymInt,
                    torch.SymFloat,
                    int,
                    float,
                ),
            ):
                res.append(i)
            else:
                res.extend(_flatten_inputs(i))
        return tuple(res) if isinstance(x, tuple) else res
    raise AssertionError(f"Unexpected type {type(x)} for x")


def _to_numpy(x):
    if hasattr(x, "numpy"):
        return x.numpy()
    if isinstance(x, int):
        # onnxruntime does not like scalar
        return np.array([x], dtype=np.int64)
    if isinstance(x, float):
        # onnxruntime does not like scalar
        return np.array([x], dtype=np.float32)
    if isinstance(x, list):
        return [_to_numpy(_) for _ in x]
    if isinstance(x, tuple):
        return tuple(_to_numpy(_) for _ in x)
    raise TypeError(f"Unable to convert type {type(x)}, x={x} into numpy")


def _make_feeds(names, args):
    if len(names) == len(args):
        return {k: _to_numpy(v) for k, v in zip(names, args)}
    if len(names) > len(args):
        flats = _flatten_inputs(args)
        return {k: _to_numpy(v) for k, v in zip(names, flats)}
    from ...helpers import string_type

    raise RuntimeError(
        f"Unable to handle names={names!r} and args={string_type(args, limit=20)}"
    )


def _clone(x):
    if hasattr(x, "clone"):
        return x.clone()
    if isinstance(x, (int, float)):
        return x
    if isinstance(x, list):
        return [_clone(_) for _ in x]
    if isinstance(x, tuple):
        return tuple(_clone(_) for _ in x)
    raise TypeError(f"Unable to clone type {type(x)}, x={x} into numpy")


def _wrap_torch_export(*args, backed_size_oblivious=False, **kwargs):
    import torch

    if backed_size_oblivious:
        with torch.fx.experimental._config.patch(backed_size_oblivious=True):
            return torch.export.export(*args, **kwargs)
    return torch.export.export(*args, **kwargs)


def _make_exporter_export(
    exporter: str,
    model: "torch.nn.Module",  # noqa: F821
    inputs: Tuple[Any, ...],
    dynamic_shapes: Optional[Any] = None,
    verbose: int = 0,
    quiet: bool = True,
) -> Union[Dict, Callable]:
    import torch

    backed_size_oblivious = "-oblivious" in exporter
    strict = "-nostrict" not in exporter

    if exporter in (
        "export-strict",
        "export-strict-oblivious",
        "export-nostrict",
        "export-nostrict-oblivious",
        "export-oblivious",
    ):
        try:
            if verbose >= 2:
                exported = _wrap_torch_export(
                    model,
                    inputs,
                    dynamic_shapes=dynamic_shapes,
                    strict=strict,
                    backed_size_oblivious=backed_size_oblivious,
                )
            else:
                with (
                    contextlib.redirect_stdout(io.StringIO()),
                    contextlib.redirect_stderr(io.StringIO()),
                ):
                    exported = _wrap_torch_export(
                        model,
                        inputs,
                        dynamic_shapes=dynamic_shapes,
                        strict=strict,
                        backed_size_oblivious=backed_size_oblivious,
                    )
        except Exception as e:
            if not quiet:
                raise
            return dict(error=str(e), success=0, error_step="export")
        if verbose >= 9:
            print("-- graph")
            print(exported.graph)
        return exported.module()

    if exporter in (
        "export-strict-dec",
        "export-strict-decall",
        "export-strict-dec-oblivious",
        "export-strict-decall-oblivious",
        "export-nostrict-dec",
        "export-nostrict-decall",
        "export-nostrict-dec-oblivious",
        "export-nostrict-decall-oblivious",
    ):
        try:
            if verbose >= 2:
                exported = _wrap_torch_export(
                    model,
                    inputs,
                    dynamic_shapes=dynamic_shapes,
                    strict=strict,
                    backed_size_oblivious=backed_size_oblivious,
                )
                if verbose >= 9:
                    print("-- graph before decomposition")
                    print(exported.graph)
                exported = (
                    exported.run_decompositions()
                    if "decall" in exporter
                    else exported.run_decompositions({})
                )
            else:
                with (
                    contextlib.redirect_stdout(io.StringIO()),
                    contextlib.redirect_stderr(io.StringIO()),
                ):
                    exported = _wrap_torch_export(
                        model,
                        inputs,
                        dynamic_shapes=dynamic_shapes,
                        strict=strict,
                        backed_size_oblivious=backed_size_oblivious,
                    )
                    if verbose >= 9:
                        print("-- graph before decomposition")
                        print(exported.graph)
                    exported = (
                        exported.run_decompositions()
                        if "decall" in exporter
                        else exported.run_decompositions({})
                    )
        except Exception as e:
            if not quiet:
                raise
            return dict(error=str(e), success=0, error_step="export")
        if verbose >= 9:
            print("-- graph after decomposition")
            print(exported.graph)
        return exported.module()

    if exporter == "export-tracing":
        from experimental_experiment.torch_interpreter.tracing import CustomTracer

        try:
            if verbose >= 2:
                graph = CustomTracer().trace(model)
                mod = torch.fx.GraphModule(model, graph)
            else:
                with (
                    contextlib.redirect_stdout(io.StringIO()),
                    contextlib.redirect_stderr(io.StringIO()),
                ):
                    graph = CustomTracer().trace(model)
                    mod = torch.fx.GraphModule(model, graph)
        except Exception as e:
            if not quiet:
                raise
            return dict(error=str(e), success=0, error_step="export")
        if verbose >= 9:
            print("-- graph")
            print(graph)
        return mod
    raise AssertionError(f"Unexpected exporter={exporter!r}")


def _make_exporter_onnx(
    exporter: str,
    model: "torch.nn.Module",  # noqa: F821
    inputs: Tuple[Any, ...],
    dynamic_shapes: Optional[Any] = None,
    verbose: int = 0,
    quiet: bool = True,
) -> Union[Dict, Tuple[onnx.ModelProto, Any]]:
    from ...helpers import string_type

    if exporter.startswith("custom"):
        from experimental_experiment.torch_interpreter import to_onnx, ExportOptions

        opts = {}
        opts["strict"] = "-strict" in exporter
        opts["fallback"] = "-fallback" in exporter
        opts["tracing"] = "-tracing" in exporter
        opts["jit"] = "-jit" in exporter
        if "-dec" in exporter:
            opts["decomposition_table"] = "all" if "-decall" in exporter else "default"
        try:
            if verbose >= 2:
                onx, builder = to_onnx(
                    model,
                    inputs,
                    dynamic_shapes=dynamic_shapes,
                    export_options=ExportOptions(**opts),
                    return_builder=True,
                )
            else:
                with (
                    contextlib.redirect_stdout(io.StringIO()),
                    contextlib.redirect_stderr(io.StringIO()),
                ):
                    onx, builder = to_onnx(
                        model,
                        inputs,
                        dynamic_shapes=dynamic_shapes,
                        export_options=ExportOptions(**opts),
                        return_builder=True,
                    )
        except Exception as e:
            if not quiet:
                raise RuntimeError(
                    f"Unable to convert model={model.__class__.__name__}, "
                    f"input={string_type(inputs[0], with_shape=True)}, "
                    f"dynamic_shapes={dynamic_shapes}, "
                    f"exporter={exporter!r}"
                ) from e
            return dict(error=str(e), success=0, error_step="export")
        return onx, builder

    if exporter == "dynamo":
        import torch

        try:
            if verbose >= 2:
                onx = torch.onnx.export(
                    model,
                    inputs,
                    dynamic_shapes=dynamic_shapes,
                    dynamo=True,
                    report=True,
                ).model_proto
            else:
                with (
                    contextlib.redirect_stdout(io.StringIO()),
                    contextlib.redirect_stderr(io.StringIO()),
                ):
                    onx = torch.onnx.export(
                        model,
                        inputs,
                        dynamic_shapes=dynamic_shapes,
                        dynamo=True,
                    ).model_proto
        except Exception as e:
            if not quiet:
                raise RuntimeError(
                    f"Unable to convert model={model.__class__.__name__}, "
                    f"input={string_type(inputs[0], with_shape=True)}, "
                    f"dynamic_shapes={dynamic_shapes}, "
                    f"exporter={exporter!r}"
                ) from e
            return dict(error=str(e), success=0, error_step="export")
        return onx, None

    if exporter == "dynamo-ir":
        import torch

        try:
            if verbose >= 2:
                ep = torch.onnx.export(
                    model,
                    inputs,
                    dynamic_shapes=dynamic_shapes,
                    dynamo=True,
                    report=True,
                )
                ep.optimize()
                onx = ep.model_proto
            else:
                with (
                    contextlib.redirect_stdout(io.StringIO()),
                    contextlib.redirect_stderr(io.StringIO()),
                ):
                    ep = torch.onnx.export(
                        model,
                        inputs,
                        dynamic_shapes=dynamic_shapes,
                        dynamo=True,
                    )
                    ep.optimize()
                    onx = ep.model_proto
        except Exception as e:
            if not quiet:
                raise RuntimeError(
                    f"Unable to convert model={model.__class__.__name__}, "
                    f"input={string_type(inputs[0], with_shape=True)}, "
                    f"dynamic_shapes={dynamic_shapes}, "
                    f"exporter={exporter!r}"
                ) from e
            return dict(error=str(e), success=0, error_step="export")
        return onx, None
    raise AssertionError(f"Unexpected exporter={exporter!r}")


def _compares_on_one_example(
    model: Callable, inputs: Tuple[Any, ...], mod: Callable, verbose: int, quiet: bool
) -> Tuple[Any, Any, Dict]:
    from onnx_diagnostic.helpers import max_diff, string_type

    try:
        expected = model(*_clone(inputs))
    except Exception as e:
        if not quiet:
            raise RuntimeError(
                f"eager mode failed=\n{string_type(inputs, with_shape=True)} "
                f"\nmodel=\n{type(model)}"
            ) from e
        res = dict(error=str(e), success=0, error_step="eager")
        return None, None, res
    try:
        got = mod(*inputs)
    except Exception as e:
        if not quiet:
            raise RuntimeError(
                f"onnxruntime failed, feeds=\n{string_type(inputs, with_shape=True)}"
            ) from e
        res = dict(error=str(e), success=0, error_step="run.0")
        return expected, None, res

    try:
        disc = max_diff(expected, got)
    except Exception as e:
        if not quiet:
            raise
        res = dict(error=str(e), success=0, error_step="discrepancy")
        return expected, got, res

    if verbose >= 5 and np.isinf(disc["abs"]):
        print("[run_exporter] comparison issues with")
        print(f"--   inputs={string_type(inputs[0], with_shape=True, limit=20)}")
        print(f"-- expected={string_type(expected, with_shape=True, limit=20)}")
        print(f"--      got={string_type(got, with_shape=True, limit=20)}")
    elif verbose >= 9:
        print("[run_exporter] inputs and outputs")
        print(
            f"--   inputs="
            f"{string_type(inputs[0], with_shape=True, with_min_max=True, limit=20)}"
        )
        print(
            f"-- expected="
            f"{string_type(expected, with_shape=True, with_min_max=True, limit=20)}"
        )
        print(f"--      got={string_type(got, with_shape=True, with_min_max=True, limit=20)}")
    del disc["n"]
    del disc["sum"]
    disc.update(
        dict(
            success=1 if disc["abs"] < 0.1 else 0,
            model_cls=model.__class__,  # type: ignore[dict-item]
            exported=mod,  # type: ignore[dict-item]
        )
    )
    if disc["abs"] >= 0.1:
        disc["error"] = "diff.0"
        disc["error_step"] = "diff.0"
        if verbose >= 9:
            max_diff(expected, got, verbose=verbose)
    else:
        disc["success"] = 1
    return expected, got, disc


def run_exporter(
    exporter: str,
    cls_model: type,
    dynamic: bool = False,
    quiet: bool = False,
    verbose: int = 0,
) -> Dict[str, Any]:
    """
    Runs an exporter and returns whether it fails or not.

    :param exporter: exporter
    :param cls_model: model class to create
    :param inputs: list of inputs to try
    :param dynamic: use dynamic shapes or not
    :param quiet: raise exception or not
    :param verbose: verbosity
    :return: results
    """
    from onnx_diagnostic.helpers import max_diff, string_type
    from onnx_diagnostic.helpers.onnx_helper import pretty_onnx

    assert hasattr(
        cls_model, "_inputs"
    ), f"Attribute '_inputs' is missing from class {cls_model}"

    model = cls_model()
    inputs = cls_model._inputs
    valid = getattr(cls_model, "_valid", None)
    if isinstance(inputs, tuple):
        inputs = [inputs]
    if dynamic:
        assert hasattr(
            cls_model, "_dynamic"
        ), f"Attribute '_inputs' is missing from class {cls_model}"
        dynamic_shapes = cls_model._dynamic
    else:
        dynamic_shapes = None

    base = dict(inputs=inputs, model=model, dynamic_shapes=dynamic_shapes)

    if verbose > 0:
        print(
            f"[run_exporter] exporter={exporter}, model={cls_model.__name__}, "
            f"dynamic={dynamic}, inputs={string_type(inputs, with_shape=True)}"
        )

    builder = None
    onx = None

    if exporter.startswith("export-"):
        mod = _make_exporter_export(
            exporter,
            model,
            inputs[0],
            dynamic_shapes=dynamic_shapes,
            verbose=verbose,
            quiet=quiet,
        )
        if isinstance(mod, dict):
            # something went wrong
            return mod
    else:
        res = _make_exporter_onnx(
            exporter,
            model,
            inputs[0],
            dynamic_shapes=dynamic_shapes,
            verbose=verbose,
            quiet=quiet,
        )
        if isinstance(res, dict):
            # something went wrong
            return res

        onx, builder = res
        base["onx"] = onx
        base["builder"] = builder
        if verbose >= 9:
            print("[run_exporter] onnx model")
            print(
                builder.pretty_text(add_fx_graph=True)
                if builder is not None
                else pretty_onnx(onx)
            )
        if verbose >= 2:
            onnx.save(onx, f"evaluation-{model.__class__.__name__}-{dynamic}-{exporter}.onnx")

        names = [i.name for i in onx.graph.input]
        flats = _flatten_inputs(inputs[0]) if len(names) > len(inputs[0]) else inputs[0]

        assert quiet or len(names) == len(flats), (
            f"Input mismatch, inputs[0]={string_type(inputs[0])} "
            f"inputs but names={names!r}, "
            f"model={cls_model.__name__}, export={exporter!r}"
        )
        if len(names) != len(flats):
            res = dict(
                error=f"Input mismatch, inputs[0]={string_type(inputs[0])} "
                f"but names={names!r}, model={cls_model.__name__}, export={exporter!r}",
                success=0,
                error_step="inputs",
            )
            res.update(base)
            return res

        import onnxruntime

        try:
            sess = onnxruntime.InferenceSession(
                onx.SerializeToString(), providers=["CPUExecutionProvider"]
            )
        except Exception as e:
            if not quiet:
                raise
            res = dict(error=str(e), success=0, error_step="ort-init")
            res.update(base)
            return res

        mod = lambda *args, names=names: sess.run(None, _make_feeds(names, args))  # noqa: E731

    # we need to clone for models modifying the inputs
    expected, got, disc = _compares_on_one_example(model, inputs[0], mod, verbose, quiet)
    if expected is not None:
        base["expected"] = expected
    if got is not None:
        base["obtained"] = got
    disc.update(base)
    disc["onnx"] = onx  # type: ignore[dict-item]

    if valid is not None:
        for valid_inputs in valid:
            expected, got, _disc = _compares_on_one_example(
                model, valid_inputs, mod, verbose, quiet
            )
            if "abs" not in disc and (np.isnan(disc["abs"]) or disc["abs"] > 1e-3):
                _disc["issue-abs"] = disc["abs"]
                _disc["issue-rel"] = disc["rel"]
                _disc["issue-inputs"] = string_type(
                    valid_inputs, with_shape=True, with_min_max=True
                )
                _disc["issue-expected"] = string_type(
                    expected, with_shape=True, with_min_max=True
                )
                _disc["issue-obtained"] = string_type(got, with_shape=True, with_min_max=True)
                if not quiet:
                    raise RuntimeError(
                        f"validation failed,"
                        f"\n-- inputs=\n{string_type(_disc['issue-inputs'])} "
                        f"\n-- exporter={exporter!r}\n-- dynamic_shapes={dynamic_shapes}, "
                        f"\n-- expected={_disc['issue-expected']}"
                        f"\n-- obtained={_disc['issue-obtained']}"
                    )
                break

    if dynamic and onx is not None:
        ds = []
        for i in onx.graph.input:
            if i.type.tensor_type:
                for di, dim in enumerate(i.type.tensor_type.shape.dim):
                    if dim.dim_param:
                        ds.append((i.name, di, dim.dim_param))
        if verbose >= 2:
            print(f"[run_exporter] dynamic dimension={ds}")
        if not ds:
            return dict(error="no dynamic shape", success=0, error_step="dynamic")

    if dynamic and len(inputs) > 1:
        for index, i in enumerate(inputs):
            if quiet:
                try:
                    expected = model(*_clone(i))
                except Exception as e:
                    return dict(error=str(e), success=0, error_step=f"run0.{index}")
            else:
                expected = model(*_clone(i))
            try:
                got = mod(*i)
            except Exception as e:
                if not quiet:
                    raise RuntimeError(
                        f"onnxruntime failed,\n-- feeds=\n{string_type(i, with_shape=True)} "
                        f"exporter={exporter!r}, dynamic_shapes={dynamic_shapes}"
                        f"\n-- model=\n{pretty_onnx(onx) if onx is not None else type(model)}"
                    ) from e
                return dict(error=str(e), success=0, error_step=f"run.{index}")

            try:
                d = max_diff(expected, got)
            except Exception as e:
                if not quiet:
                    raise
                return dict(error=str(e), success=0, error_step=f"discrepancy.{index}")

            if verbose >= 5 and np.isinf(d["abs"]):
                print(f"[run_exporter] comparison issues iteration {index}")
                print(f"--   inputs={string_type(i, with_shape=True)}")
                print(f"-- expected={string_type(expected, with_shape=True)}")
                print(f"--      got={string_type(got, with_shape=True)}")
            elif verbose >= 9:
                print(f"[run_exporter] inputs and outputs iteration {index}")
                print(f"--   inputs={string_type(i, with_shape=True, with_min_max=True)}")
                print(
                    f"-- expected={string_type(expected, with_shape=True, with_min_max=True)}"
                )
                print(f"--      got={string_type(got, with_shape=True, with_min_max=True)}")
            del d["n"]
            del d["sum"]
            if d["abs"] >= 0.1:
                d["error"] = f"diff.{index}"
                d["error_step"] = f"diff.{index}"
                d["success"] = 0
                disc.update(d)

    disc.update(base)
    return disc
