import ast
import enum
import inspect
from dataclasses import is_dataclass, fields
from typing import Any, Callable, Dict, List, Optional, Set
import numpy as np


def size_type(dtype: Any) -> int:
    """Returns the element size for an element type."""
    if isinstance(dtype, int):
        from onnx import TensorProto

        # It is a TensorProto.DATATYPE
        if dtype in {
            TensorProto.DOUBLE,
            TensorProto.INT64,
            TensorProto.UINT64,
            TensorProto.COMPLEX64,
        }:
            return 8
        if dtype in {TensorProto.FLOAT, TensorProto.INT32, TensorProto.UINT32}:
            return 4
        if dtype in {
            TensorProto.FLOAT16,
            TensorProto.BFLOAT16,
            TensorProto.INT16,
            TensorProto.UINT16,
        }:
            return 2
        if dtype in {
            TensorProto.INT8,
            TensorProto.UINT8,
            TensorProto.BOOL,
            TensorProto.FLOAT8E4M3FN,
            TensorProto.FLOAT8E4M3FNUZ,
            TensorProto.FLOAT8E5M2,
            TensorProto.FLOAT8E5M2FNUZ,
        }:
            return 1
        if dtype in {TensorProto.COMPLEX128}:
            return 16
        from .helpers.onnx_helper import onnx_dtype_name

        raise AssertionError(
            f"Unable to return the element size for type {onnx_dtype_name(dtype)}"
        )

    if dtype == np.float64 or dtype == np.int64:
        return 8
    if dtype == np.float32 or dtype == np.float32:
        return 4
    if dtype == np.float16 or dtype == np.int16:
        return 2
    if dtype == np.int32:
        return 4
    if dtype == np.int8:
        return 1
    if hasattr(np, "uint64"):
        # it fails on mac
        if dtype == np.uint64:
            return 8
        if dtype == np.uint32:
            return 4
        if dtype == np.uint16:
            return 2
        if dtype == np.uint8:
            return 1

    import torch

    if dtype in {torch.float64, torch.int64}:
        return 8
    if dtype in {torch.float32, torch.int32}:
        return 4
    if dtype in {torch.float16, torch.int16, torch.bfloat16}:
        return 2
    if dtype in {torch.int8, torch.uint8, torch.bool}:
        return 1
    if hasattr(torch, "uint64"):
        # it fails on mac
        if dtype in {torch.uint64}:
            return 8
        if dtype in {torch.uint32}:
            return 4
        if dtype in {torch.uint16}:
            return 2
    import ml_dtypes

    if dtype == ml_dtypes.bfloat16:
        return 2
    raise AssertionError(f"Unexpected dtype={dtype}")


def string_type(
    obj: Any,
    with_shape: bool = False,
    with_min_max: bool = False,
    with_device: bool = False,
    ignore: bool = False,
    limit: int = 20,
    verbose: int = 0,
) -> str:
    """
    Displays the types of an object as a string.

    :param obj: any
    :param with_shape: displays shapes as well
    :param with_min_max: displays information about the values
    :param with_device: display the device
    :param ignore: if True, just prints the type for unknown types
    :param verbose: verbosity (to show the path it followed to get that print)
    :return: str

    .. runpython::
        :showcode:

        from onnx_diagnostic.helpers import string_type

        print(string_type((1, ["r", 6.6])))

    With pytorch:

    .. runpython::
        :showcode:

        import torch
        from onnx_diagnostic.helpers import string_type

        inputs = (
            torch.rand((3, 4), dtype=torch.float16),
            [
                torch.rand((5, 6), dtype=torch.float16),
                torch.rand((5, 6, 7), dtype=torch.float16),
            ]
        )

        # with shapes
        print(string_type(inputs, with_shape=True))

        # with min max
        print(string_type(inputs, with_shape=True, with_min_max=True))
    """
    if obj is None:
        if verbose:
            print(f"[string_type] A:{type(obj)}")
        return "None"

    # tuple
    if isinstance(obj, tuple):
        if len(obj) == 1:
            s = string_type(
                obj[0],
                with_shape=with_shape,
                with_min_max=with_min_max,
                with_device=with_device,
                ignore=ignore,
                limit=limit,
                verbose=verbose,
            )
            if verbose:
                print(f"[string_type] C:{type(obj)}")
            return f"({s},)"
        if len(obj) < limit:
            js = ",".join(
                string_type(
                    o,
                    with_shape=with_shape,
                    with_min_max=with_min_max,
                    with_device=with_device,
                    ignore=ignore,
                    limit=limit,
                    verbose=verbose,
                )
                for o in obj
            )
            if verbose:
                print(f"[string_type] D:{type(obj)}")
            return f"({js})"
        tt = string_type(
            obj[0],
            with_shape=with_shape,
            with_min_max=with_min_max,
            with_device=with_device,
            ignore=ignore,
            limit=limit,
            verbose=verbose,
        )
        if with_min_max and all(isinstance(_, (int, float, bool)) for _ in obj):
            mini, maxi, avg = min(obj), max(obj), sum(float(_) for _ in obj) / len(obj)
            if verbose:
                print(f"[string_type] E:{type(obj)}")
            return f"#{len(obj)}({tt},...)[{mini},{maxi}:A[{avg}]]"
        if verbose:
            print(f"[string_type] F:{type(obj)}")
        return f"#{len(obj)}({tt},...)"
    # list
    if isinstance(obj, list):
        if len(obj) < limit:
            js = ",".join(
                string_type(
                    o,
                    with_shape=with_shape,
                    with_min_max=with_min_max,
                    with_device=with_device,
                    ignore=ignore,
                    limit=limit,
                    verbose=verbose,
                )
                for o in obj
            )
            if verbose:
                print(f"[string_type] G:{type(obj)}")
            return f"#{len(obj)}[{js}]"
        tt = string_type(
            obj[0],
            with_shape=with_shape,
            with_min_max=with_min_max,
            with_device=with_device,
            ignore=ignore,
            limit=limit,
            verbose=verbose,
        )
        if with_min_max and all(isinstance(_, (int, float, bool)) for _ in obj):
            mini, maxi, avg = min(obj), max(obj), sum(float(_) for _ in obj) / len(obj)
            if verbose:
                print(f"[string_type] H:{type(obj)}")
            return f"#{len(obj)}[{tt},...][{mini},{maxi}:{avg}]"
        if verbose:
            print(f"[string_type] I:{type(obj)}")
        return f"#{len(obj)}[{tt},...]"
    # set
    if isinstance(obj, set):
        if len(obj) < 10:
            js = ",".join(
                string_type(
                    o,
                    with_shape=with_shape,
                    with_min_max=with_min_max,
                    with_device=with_device,
                    ignore=ignore,
                    limit=limit,
                    verbose=verbose,
                )
                for o in obj
            )
            if verbose:
                print(f"[string_type] J:{type(obj)}")
            return f"{{{js}}}"
        if with_min_max and all(isinstance(_, (int, float, bool)) for _ in obj):
            mini, maxi, avg = min(obj), max(obj), sum(float(_) for _ in obj) / len(obj)
            if verbose:
                print(f"[string_type] K:{type(obj)}")
            return f"{{...}}#{len(obj)}[{mini},{maxi}:A{avg}]"
        if verbose:
            print(f"[string_type] L:{type(obj)}")
        return f"{{...}}#{len(obj)}" if with_shape else "{...}"
    # dict
    if isinstance(obj, dict) and type(obj) is dict:
        if len(obj) == 0:
            if verbose:
                print(f"[string_type] M:{type(obj)}")
            return "{}"

        import torch

        if all(isinstance(k, int) for k in obj) and all(
            isinstance(
                v,
                (
                    str,
                    torch.export.dynamic_shapes._Dim,
                    torch.export.dynamic_shapes._DerivedDim,
                    torch.export.dynamic_shapes._DimHint,
                ),
            )
            for v in obj.values()
        ):
            # This is dynamic shapes
            rows = []
            for k, v in obj.items():
                if isinstance(v, str):
                    rows.append(f"{k}:DYN({v})")
                else:
                    rows.append(f"{k}:{string_type(v, verbose=verbose)}")
            if verbose:
                print(f"[string_type] DS0:{type(obj)}")
            return f"{{{','.join(rows)}}}"

        kws = dict(
            with_shape=with_shape,
            with_min_max=with_min_max,
            with_device=with_device,
            ignore=ignore,
            limit=limit,
            verbose=verbose,
        )
        s = ",".join(f"{kv[0]}:{string_type(kv[1],**kws)}" for kv in obj.items())
        if all(isinstance(k, int) for k in obj):
            if verbose:
                print(f"[string_type] N:{type(obj)}")
            return f"{{{s}}}"
        if verbose:
            print(f"[string_type] O:{type(obj)}")
        return f"dict({s})"
    # array
    if isinstance(obj, np.ndarray):
        from .onnx_helper import np_dtype_to_tensor_dtype

        if with_min_max:
            s = string_type(obj, with_shape=with_shape)
            if len(obj.shape) == 0:
                return f"{s}={obj}"
            if obj.size == 0:
                return f"{s}[empty]"
            n_nan = np.isnan(obj.reshape((-1,))).astype(int).sum()
            if n_nan > 0:
                nob = obj.ravel()
                nob = nob[~np.isnan(nob)]
                if nob.size == 0:
                    if verbose:
                        print(f"[string_type] A1:{type(obj)}")
                    return f"{s}[N{n_nan}nans]"
                if verbose:
                    print(f"[string_type] A2:{type(obj)}")
                return f"{s}[{nob.min()},{nob.max()}:A{nob.astype(float).mean()}N{n_nan}nans]"
            if verbose:
                print(f"[string_type] A3:{type(obj)}")
            return f"{s}[{obj.min()},{obj.max()}:A{obj.astype(float).mean()}]"
        i = np_dtype_to_tensor_dtype(obj.dtype)
        if not with_shape:
            if verbose:
                print(f"[string_type] A4:{type(obj)}")
            return f"A{i}r{len(obj.shape)}"
        if verbose:
            print(f"[string_type] A5:{type(obj)}")
        return f"A{i}s{'x'.join(map(str, obj.shape))}"

    import torch

    # Dim, SymInt
    if isinstance(obj, torch.export.dynamic_shapes._DerivedDim):
        if verbose:
            print(f"[string_type] Y1:{type(obj)}")
        return "DerivedDim"
    if isinstance(obj, torch.export.dynamic_shapes._Dim):
        if verbose:
            print(f"[string_type] Y2:{type(obj)}")
        return f"Dim({obj.__name__})"
    if isinstance(obj, torch.SymInt):
        if verbose:
            print(f"[string_type] Y3:{type(obj)}")
        return "SymInt"
    if isinstance(obj, torch.SymFloat):
        if verbose:
            print(f"[string_type] Y4:{type(obj)}")
        return "SymFloat"

    if isinstance(obj, torch.export.dynamic_shapes._DimHint):
        cl = (
            torch.export.dynamic_shapes._DimHintType
            if hasattr(torch.export.dynamic_shapes, "_DimHintType")
            else torch.export.Dim
        )
        if obj in (torch.export.Dim.DYNAMIC, cl.DYNAMIC):
            if verbose:
                print(f"[string_type] Y8:{type(obj)}")
            return "DYNAMIC"
        if obj in (torch.export.Dim.AUTO, cl.AUTO):
            if verbose:
                print(f"[string_type] Y9:{type(obj)}")
            return "AUTO"
        if verbose:
            print(f"[string_type] Y7:{type(obj)}")
        return str(obj)

    if isinstance(obj, bool):
        if with_min_max:
            if verbose:
                print(f"[string_type] W1:{type(obj)}")
            return f"bool={obj}"
        if verbose:
            print(f"[string_type] W2:{type(obj)}")
        return "bool"
    if isinstance(obj, int):
        if with_min_max:
            if verbose:
                print(f"[string_type] W3:{type(obj)}")
            return f"int={obj}"
        if verbose:
            print(f"[string_type] W4:{type(obj)}")
        return "int"
    if isinstance(obj, float):
        if with_min_max:
            if verbose:
                print(f"[string_type] W6:{type(obj)}")
            return f"float={obj}"
        if verbose:
            print(f"[string_type] W8:{type(obj)}")
        return "float"
    if isinstance(obj, str):
        if verbose:
            print(f"[string_type] W9:{type(obj)}")
        return "str"
    if isinstance(obj, slice):
        if verbose:
            print(f"[string_type] W10:{type(obj)}")
        return "slice"

    if is_dataclass(obj):
        # That includes torch.export.Dim.AUTO, torch.export.Dim.DYNAMIC so they need to be
        # handled before that.
        values = {f.name: getattr(obj, f.name, None) for f in fields(obj)}
        values = {k: v for k, v in values.items() if v is not None}
        s = string_type(
            values,
            with_shape=with_shape,
            with_min_max=with_min_max,
            with_device=with_device,
            ignore=ignore,
            limit=limit,
            verbose=verbose,
        )
        if verbose:
            print(f"[string_type] B:{type(obj)}")
        return f"{obj.__class__.__name__}{s[4:]}"

    # Tensors
    if isinstance(obj, torch._subclasses.fake_tensor.FakeTensor):
        from .onnx_helper import torch_dtype_to_onnx_dtype

        i = torch_dtype_to_onnx_dtype(obj.dtype)
        prefix = ("G" if obj.get_device() >= 0 else "C") if with_device else ""
        if not with_shape:
            if verbose:
                print(f"[string_type] F1:{type(obj)}")
            return f"{prefix}F{i}r{len(obj.shape)}"
        if verbose:
            print(f"[string_type] F2:{type(obj)}")
        return f"{prefix}F{i}s{'x'.join(map(str, obj.shape))}"
    if isinstance(obj, torch.Tensor):
        from .onnx_helper import torch_dtype_to_onnx_dtype

        if with_min_max:
            s = string_type(obj, with_shape=with_shape, with_device=with_device)
            if len(obj.shape) == 0:
                if verbose:
                    print(f"[string_type] T1:{type(obj)}")
                return f"{s}={obj}"
            if obj.numel() == 0:
                if verbose:
                    print(f"[string_type] T2:{type(obj)}")
                return f"{s}[empty]"
            n_nan = obj.reshape((-1,)).isnan().to(int).sum()
            if n_nan > 0:
                nob = obj.reshape((-1,))
                nob = nob[~nob.isnan()]
                if obj.dtype in {torch.complex64, torch.complex128}:
                    if verbose:
                        print(f"[string_type] T3:{type(obj)}")
                    return (
                        f"{s}[{nob.abs().min()},{nob.abs().max():A{nob.mean()}N{n_nan}nans}]"
                    )
                if verbose:
                    print(f"[string_type] T5:{type(obj)}")
                return f"{s}[{obj.min()},{obj.max()}:A{obj.to(float).mean()}N{n_nan}nans]"
            if obj.dtype in {torch.complex64, torch.complex128}:
                if verbose:
                    print(f"[string_type] T6:{type(obj)}")
                return f"{s}[{obj.abs().min()},{obj.abs().max()}:A{obj.abs().mean()}]"
            if verbose:
                print(f"[string_type] T7:{type(obj)}")
            return f"{s}[{obj.min()},{obj.max()}:A{obj.to(float).mean()}]"
        i = torch_dtype_to_onnx_dtype(obj.dtype)
        prefix = ("G" if obj.get_device() >= 0 else "C") if with_device else ""
        if not with_shape:
            if verbose:
                print(f"[string_type] T8:{type(obj)}")
            return f"{prefix}T{i}r{len(obj.shape)}"
        if verbose:
            print(f"[string_type] T9:{type(obj)}")
        return f"{prefix}T{i}s{'x'.join(map(str, obj.shape))}"

    if obj.__class__.__name__ == "OrtValue":
        if not obj.has_value():
            if verbose:
                print(f"[string_type] V1:{type(obj)}")
            return "OV(<novalue>)"
        if not obj.is_tensor():
            if verbose:
                print(f"[string_type] V2:{type(obj)}")
            return "OV(NOTENSOR)"
        if with_min_max:
            try:
                t = obj.numpy()
            except Exception:
                # pass unable to convert into numpy (bfloat16, ...)
                if verbose:
                    print(f"[string_type] V3:{type(obj)}")
                return "OV(NO-NUMPY:FIXIT)"
            if verbose:
                print(f"[string_type] V4:{type(obj)}")
            return f"OV({string_type(t, with_shape=with_shape, with_min_max=with_min_max)})"
        dt = obj.element_type()
        shape = obj.shape()
        if with_shape:
            if verbose:
                print(f"[string_type] V5:{type(obj)}")
            return f"OV{dt}s{'x'.join(map(str, shape))}"
        if verbose:
            print(f"[string_type] V6:{type(obj)}")
        return f"OV{dt}r{len(shape)}"

    # others classes

    if obj.__class__.__name__ == "MambaCache":
        c = string_type(
            obj.conv_states,
            with_shape=with_shape,
            with_min_max=with_min_max,
            with_device=with_device,
            limit=limit,
            verbose=verbose,
        )
        d = string_type(
            obj.ssm_states,
            with_shape=with_shape,
            with_min_max=with_min_max,
            with_device=with_device,
            limit=limit,
            verbose=verbose,
        )
        if verbose:
            print(f"[string_type] CACHE1:{type(obj)}")
        return f"MambaCache(conv_states={c}, ssm_states={d})"

    if obj.__class__.__name__ in ("DynamicCache", "SlidingWindowCache"):
        kc = string_type(
            obj.key_cache,
            with_shape=with_shape,
            with_min_max=with_min_max,
            with_device=with_device,
            limit=limit,
            verbose=verbose,
        )
        vc = string_type(
            obj.value_cache,
            with_shape=with_shape,
            with_min_max=with_min_max,
            with_device=with_device,
            limit=limit,
            verbose=verbose,
        )
        if verbose:
            print(f"[string_type] CACHE2:{type(obj)}")
        return f"{obj.__class__.__name__}(key_cache={kc}, value_cache={vc})"

    if obj.__class__.__name__ == "EncoderDecoderCache":
        att = string_type(
            obj.self_attention_cache,
            with_shape=with_shape,
            with_min_max=with_min_max,
            with_device=with_device,
            limit=limit,
            verbose=verbose,
        )
        cross = string_type(
            obj.cross_attention_cache,
            with_shape=with_shape,
            with_min_max=with_min_max,
            with_device=with_device,
            limit=limit,
            verbose=verbose,
        )
        if verbose:
            print(f"[string_type] CACHE3:{type(obj)}")
        return (
            f"{obj.__class__.__name__}(self_attention_cache={att}, "
            f"cross_attention_cache={cross})"
        )

    if obj.__class__ in torch.utils._pytree.SUPPORTED_NODES:
        from .cache_helper import flatten_unflatten_for_dynamic_shapes

        args = flatten_unflatten_for_dynamic_shapes(obj)
        att = string_type(
            args,
            with_shape=with_shape,
            with_min_max=with_min_max,
            with_device=with_device,
            limit=limit,
            verbose=verbose,
        )
        if verbose:
            print(f"[string_type] DS:{type(obj)}")
        return f"{obj.__class__.__name__}[serialized]({att})"

    if type(obj).__name__ == "Node" and hasattr(obj, "meta"):
        # torch.fx.node.Node
        if verbose:
            print(f"[string_type] TT1:{type(obj)}")
        return f"%{obj.target}"
    if type(obj).__name__ == "ValueInfoProto":
        if verbose:
            print(f"[string_type] OO1:{type(obj)}")
        return f"OT{obj.type.tensor_type.elem_type}"

    if obj.__class__.__name__ == "BatchFeature":
        s = string_type(
            obj.data,
            with_shape=with_shape,
            with_min_max=with_min_max,
            with_device=with_device,
            limit=limit,
            verbose=verbose,
        )
        if verbose:
            print(f"[string_type] TT2:{type(obj)}")
        return f"BatchFeature(data={s})"

    if obj.__class__.__name__ == "BatchEncoding":
        s = string_type(
            obj.data,
            with_shape=with_shape,
            with_min_max=with_min_max,
            with_device=with_device,
            limit=limit,
            verbose=verbose,
        )
        if verbose:
            print(f"[string_type] TT3:{type(obj)}")
        return f"BatchEncoding(data={s})"

    if obj.__class__.__name__ == "VirtualTensor":
        if verbose:
            print(f"[string_type] TT4:{type(obj)}")
        return (
            f"{obj.__class__.__name__}(name={obj.name!r}, "
            f"dtype={obj.dtype}, shape={obj.shape})"
        )

    if isinstance(obj, torch.nn.Module):
        if verbose:
            print(f"[string_type] MM:{type(obj)}")
        return f"{obj.__class__.__name__}(...)"

    if isinstance(obj, (torch.device, torch.dtype, torch.memory_format, torch.layout)):
        if verbose:
            print(f"[string_type] TT7:{type(obj)}")
        return f"{obj.__class__.__name__}({obj})"

    if isinstance(  # TreeSpec, MappingKey, SequenceKey
        obj,
        (
            torch.utils._pytree.TreeSpec,
            torch.utils._pytree.MappingKey,
            torch.utils._pytree.SequenceKey,
        ),
    ):
        if verbose:
            print(f"[string_type] TT8:{type(obj)}")
        return repr(obj).replace(" ", "").replace("\n", " ")

    if ignore:
        if verbose:
            print(f"[string_type] CACHE4:{type(obj)}")
        return f"{obj.__class__.__name__}(...)"

    if obj.__class__.__name__.endswith("Config"):
        import transformers.configuration_utils as tcu

        if isinstance(obj, tcu.PretrainedConfig):
            if verbose:
                print(f"[string_type] CONFIG:{type(obj)}")
            s = str(obj.to_diff_dict()).replace("\n", "").replace(" ", "")
            return f"{obj.__class__.__name__}(**{s})"

    if verbose:
        print(f"[string_type] END:{type(obj)}")
    raise AssertionError(f"Unsupported type {type(obj).__name__!r} - {type(obj)}")


def string_signature(sig: Any) -> str:
    """Displays the signature of a functions."""

    def _k(p, kind):
        for name in dir(p):
            if getattr(p, name) == kind:
                return name
        return repr(kind)

    text = [" __call__ ("]
    for p in sig.parameters:
        pp = sig.parameters[p]
        kind = repr(pp.kind)
        t = f"{p}: {pp.annotation}" if pp.annotation is not inspect._empty else p
        if pp.default is not inspect._empty:
            t = f"{t} = {pp.default!r}"
        if kind == pp.VAR_POSITIONAL:
            t = f"*{t}"
        le = (30 - len(t)) * " "
        text.append(f"    {t}{le}|{_k(pp,kind)}")
    text.append(
        f") -> {sig.return_annotation}" if sig.return_annotation is not inspect._empty else ")"
    )
    return "\n".join(text)


def string_sig(f: Callable, kwargs: Optional[Dict[str, Any]] = None) -> str:
    """
    Displays the signature of a function if the default
    if the given value is different from
    """
    if hasattr(f, "__init__") and kwargs is None:
        fct = f.__init__
        kwargs = f.__dict__
        name = f.__class__.__name__
    else:
        fct = f
        name = f.__name__

    if kwargs is None:
        kwargs = {}
    rows = []
    sig = inspect.signature(fct)
    for p in sig.parameters:
        pp = sig.parameters[p]
        d = pp.default
        if d is inspect._empty:
            if p in kwargs:
                v = kwargs[p]
                rows.append(
                    f"{p}={v!r}" if not isinstance(v, enum.IntEnum) else f"{p}={v.name}"
                )
            continue
        v = kwargs.get(p, d)
        if d != v:
            rows.append(f"{p}={v!r}" if not isinstance(v, enum.IntEnum) else f"{p}={v.name}")
            continue
    atts = ", ".join(rows)
    return f"{name}({atts})"


def make_hash(obj: Any) -> str:
    """
    Returns a simple hash of ``id(obj)`` in four letter.
    """
    aa = id(obj) % (26**3)
    return f"{chr(65 + aa // 26 ** 2)}{chr(65 + (aa // 26) % 26)}{chr(65 + aa % 26)}"


def rename_dynamic_dimensions(
    constraints: Dict[str, Set[str]], original: Set[str], ban_prefix: str = "DYN"
) -> Dict[str, str]:
    """
    Renames dynamic shapes as requested by the user. :func:`torch.export.export` uses
    many names for dynamic dimensions. When building the onnx model,
    some of them are redundant and can be replaced by the name provided by the user.

    :param constraints: exhaustive list of used names and all the values equal to it
    :param original: the names to use if possible
    :param ban_prefix: avoid any rewriting by a constant starting with this prefix
    :return: replacement dictionary
    """
    replacements = {s: s for s in original}
    all_values = set(constraints) | original

    not_done = set(constraints)
    max_iter = len(replacements)
    while not_done and max_iter > 0:
        max_iter -= 1
        for k, v in constraints.items():
            common = v & original
            if not common:
                continue
            sorted_common = sorted(common)
            by = sorted_common[0]
            if ban_prefix and by.startswith(ban_prefix):
                continue
            replacements[k] = by
            for vv in v:
                if vv not in replacements:
                    replacements[vv] = by
        not_done = all_values - set(replacements)
    return replacements


def rename_dynamic_expression(expression: str, replacements: Dict[str, str]):
    """
    Renames variables of an expression.

    :param expression: something like ``s15 + seq_length``
    :param replacements: replacements to make
    :return: new string
    """

    class RenameVariable(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id in replacements:
                node.id = replacements[node.id]
            return node

    tree = ast.parse(expression)
    transformer = RenameVariable()
    new_tree = transformer.visit(tree)
    return ast.unparse(new_tree)


def flatten_object(x: Any, drop_keys: bool = False) -> Any:
    """
    Flattens the object.
    It accepts some common classes used in deep learning.

    :param x: any object
    :param drop_keys: drop the keys if a dictionary is flattened.
        Keeps the order defined by the dictionary if False, sort them if True.
    :return: flattened object
    """
    if x is None:
        return x
    if isinstance(x, (list, tuple)):
        res = []
        for i in x:
            if i is None or hasattr(i, "shape") or isinstance(i, (int, float, str)):
                res.append(i)
            else:
                res.extend(flatten_object(i, drop_keys=drop_keys))
        return tuple(res) if isinstance(x, tuple) else res
    if isinstance(x, dict):
        # We flatten the keys.
        if drop_keys:
            return flatten_object(list(x.values()), drop_keys=drop_keys)
        return flatten_object(list(x.items()), drop_keys=drop_keys)

    if x.__class__.__name__ == "DynamicCache":
        res = flatten_object(x.key_cache) + flatten_object(x.value_cache)
        return tuple(res)
    if x.__class__.__name__ == "EncoderDecoderCache":
        res = flatten_object(x.self_attention_cache) + flatten_object(x.cross_attention_cache)
        return tuple(res)
    if x.__class__.__name__ == "MambaCache":
        if isinstance(x.conv_states, list):
            res = flatten_object(x.conv_states) + flatten_object(x.ssm_states)
            return tuple(res)
        return (x.conv_states, x.ssm_states)
    if hasattr(x, "to_tuple"):
        return flatten_object(x.to_tuple(), drop_keys=drop_keys)
    if hasattr(x, "shape"):
        # A tensor. Nothing to do.
        return x
    raise TypeError(
        f"Unexpected type {type(x)} for x, drop_keys={drop_keys}, "
        f"content is {string_type(x, with_shape=True)}"
    )


def max_diff(
    expected: Any,
    got: Any,
    verbose: int = 0,
    level: int = 0,
    flatten: bool = False,
    debug_info: Optional[List[str]] = None,
    begin: int = 0,
    end: int = -1,
    _index: int = 0,
    allow_unique_tensor_with_list_of_one_element: bool = True,
) -> Dict[str, float]:
    """
    Returns the maximum discrepancy.

    :param expected: expected values
    :param got: values
    :param verbose: verbosity level
    :param level: for embedded outputs, used for debug purpposes
    :param flatten: flatten outputs
    :param debug_info: debug information
    :param begin: first output to considered
    :param end: last output to considered (-1 for the last one)
    :param _index: used with begin and end
    :param allow_unique_tensor_with_list_of_one_element:
        allow a comparison between a single tensor and a list of one tensor
    :return: dictionary with many values

    * abs: max absolute error
    * rel: max relative error
    * sum: sum of the errors
    * n: number of outputs values, if there is one
        output, this number will be the number of elements
        of this output
    * dnan: difference in the number of nan

    You may use :func:`string_diff` to display the discrepancies in one string.
    """
    if expected is None and got is None:
        return dict(abs=0, rel=0, sum=0, n=0, dnan=0)
    if allow_unique_tensor_with_list_of_one_element:
        if hasattr(expected, "shape") and isinstance(got, (list, tuple)) and len(got) == 1:
            return max_diff(
                expected,
                got[0],
                verbose=verbose,
                level=level,
                flatten=False,
                debug_info=debug_info,
                allow_unique_tensor_with_list_of_one_element=False,
            )
        return max_diff(
            expected,
            got,
            verbose=verbose,
            level=level,
            flatten=flatten,
            debug_info=debug_info,
            begin=begin,
            end=end,
            _index=_index,
            allow_unique_tensor_with_list_of_one_element=False,
        )
    if hasattr(expected, "to_tuple"):
        if verbose >= 6:
            print(f"[max_diff] to_tuple1: {string_type(expected)} ? {string_type(got)}")
        return max_diff(
            expected.to_tuple(),
            got,
            verbose=verbose,
            level=level + 1,
            debug_info=(
                [*(debug_info if debug_info else []), f"{' ' * level}to_tupleA"]
                if verbose > 5
                else None
            ),
            begin=begin,
            end=end,
            _index=_index,
            flatten=flatten,
        )

    if hasattr(got, "to_tuple"):
        if verbose >= 6:
            print(f"[max_diff] to_tuple2: {string_type(expected)} ? {string_type(got)}")
        return max_diff(
            expected,
            got.to_tuple(),
            verbose=verbose,
            level=level + 1,
            debug_info=(
                [*(debug_info if debug_info else []), f"{' ' * level}to_tupleB"]
                if verbose > 5
                else None
            ),
            begin=begin,
            end=end,
            _index=_index,
            flatten=flatten,
        )

        if isinstance(got, (list, tuple)):
            if len(got) != 1:
                if verbose >= 6:
                    print(
                        f"[max_diff] list,tuple,2: {string_type(expected)} "
                        f"? {string_type(got)}"
                    )
                if verbose > 2:
                    import torch

                    print(
                        f"[max_diff] (a) inf because len(expected)={len(expected)}!=1, "
                        f"len(got)={len(got)}, level={level}, _index={_index}"
                    )
                    for i, (a, b) in enumerate(zip(expected, got)):
                        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
                            print(
                                f"    i={i} expected {a.dtype}:{a.shape}, "
                                f"has {b.dtype}:{b.shape}, _index={_index}"
                            )
                        else:
                            print(
                                f"    i={i} a is {type(a)}, "
                                f"b is {type(b)}, _index={_index}"
                            )
                return dict(abs=np.inf, rel=np.inf, sum=np.inf, n=np.inf, dnan=np.inf)
            if verbose >= 6:
                print(f"[max_diff] list,tuple,1: {string_type(expected)} ? {string_type(got)}")
            return max_diff(
                expected,
                got[0],
                verbose=verbose,
                level=level + 1,
                begin=begin,
                end=end,
                _index=_index,
                debug_info=debug_info,
                flatten=flatten,
            )

    if isinstance(expected, (tuple, list)):
        if verbose >= 6:
            print(f"[max_diff] list,tuple,0: {string_type(expected)} ? {string_type(got)}")
        if len(expected) == 1 and not isinstance(got, type(expected)):
            if verbose >= 6:
                print(f"[max_diff] list,tuple,3: {string_type(expected)} ? {string_type(got)}")
            return max_diff(
                expected[0],
                got,
                verbose=verbose,
                level=level + 1,
                begin=begin,
                end=end,
                _index=_index,
                debug_info=debug_info,
                flatten=flatten,
            )
        if not isinstance(got, (tuple, list)):
            if verbose >= 6:
                print(f"[max_diff] list,tuple,4: {string_type(expected)} ? {string_type(got)}")
            if verbose > 2:
                print(
                    f"[max_diff] inf because type(expected)={type(expected)}, "
                    f"type(got)={type(got)}, level={level}, _index={_index}"
                )
            return dict(abs=np.inf, rel=np.inf, sum=np.inf, n=np.inf, dnan=np.inf)

        if len(got) != len(expected):
            if flatten:
                if verbose >= 6:
                    print(
                        f"[max_diff] list,tuple,5: {string_type(expected)} "
                        f"? {string_type(got)}"
                    )
                # Let's flatten.
                if verbose > 2:
                    print(
                        f"[max_diff] flattening because of length mismatch, "
                        f"expected is\n  {string_type(expected)}\n  -- and got is\n  "
                        f"{string_type(got)}"
                    )
                flat_a = flatten_object(expected, drop_keys=True)
                flat_b = flatten_object(got, drop_keys=True)
                if verbose > 2:
                    print(
                        f"[max_diff] after flattening, "
                        f"expected is\n  {string_type(flat_a)}\n  -- and got is\n  "
                        f"{string_type(flat_b)}"
                    )
                return max_diff(
                    flat_a,
                    flat_b,
                    verbose=verbose,
                    level=level,
                    begin=begin,
                    end=end,
                    _index=_index,
                    debug_info=(
                        [
                            *(debug_info if debug_info else []),
                            (
                                f"{' ' * level}flatten["
                                f"{string_type(expected)},{string_type(got)}]"
                            ),
                        ]
                        if verbose > 5
                        else None
                    ),
                    flatten=False,
                )

            if verbose > 2:
                import torch

                print(
                    f"[max_diff] (b) inf because len(expected)={len(expected)}, "
                    f"len(got)={len(got)}, level={level}, _index={_index}"
                )
                for i, (a, b) in enumerate(zip(expected, got)):
                    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
                        print(
                            f"    i={i} expected {a.dtype}:{a.shape}, "
                            f"has {b.dtype}:{b.shape}, _index={_index}"
                        )
                    else:
                        print(f"    i={i} a is {type(a)}, b is {type(b)}")
            return dict(abs=np.inf, rel=np.inf, sum=np.inf, n=np.inf, dnan=np.inf)

        if verbose >= 6:
            print(f"[max_diff] list,tuple,6: {string_type(expected)} ? {string_type(got)}")
        am, rm, sm, n, dn = 0, 0, 0.0, 0.0, 0
        for ip, (e, g) in enumerate(zip(expected, got)):
            d = max_diff(
                e,
                g,
                verbose=verbose,
                level=level + 1,
                debug_info=(
                    [
                        *(debug_info if debug_info else []),
                        f"{' ' * level}[{ip}] so far abs {am} - rel {rm}",
                    ]
                    if verbose > 5
                    else None
                ),
                begin=begin,
                end=end,
                _index=_index + ip,
                flatten=flatten,
            )
            am = max(am, d["abs"])
            dn = max(dn, d["dnan"])
            rm = max(rm, d["rel"])
            sm += d["sum"]
            n += d["n"]
        return dict(abs=am, rel=rm, sum=sm, n=n, dnan=dn)

    if isinstance(expected, dict):
        if verbose >= 6:
            print(f"[max_diff] dict: {string_type(expected)} ? {string_type(got)}")
        assert (
            begin == 0 and end == -1
        ), f"begin={begin}, end={end} not compatible with dictionaries"
        if isinstance(got, dict):
            if len(expected) != len(got):
                return dict(abs=np.inf, rel=np.inf, sum=np.inf, n=np.inf, dnan=np.inf)
            if set(expected) != set(got):
                return dict(abs=np.inf, rel=np.inf, sum=np.inf, n=np.inf, dnan=np.inf)
            keys = sorted(expected)
            return max_diff(
                [expected[k] for k in keys],
                [got[k] for k in keys],
                level=level,
                flatten=flatten,
                debug_info=debug_info,
                begin=begin,
                end=end,
                _index=_index,
                verbose=verbose,
            )

        if not isinstance(got, (tuple, list)):
            return dict(abs=np.inf, rel=np.inf, sum=np.inf, n=np.inf, dnan=np.inf)
        if len(expected) != len(got):
            return dict(abs=np.inf, rel=np.inf, sum=np.inf, n=np.inf, dnan=np.inf)
        return max_diff(
            list(expected.values()),
            got,
            level=level,
            flatten=flatten,
            debug_info=debug_info,
            begin=begin,
            end=end,
            _index=_index,
            verbose=verbose,
        )

    import torch

    if isinstance(expected, np.ndarray) or isinstance(got, np.ndarray):
        if isinstance(expected, torch.Tensor):
            expected = expected.detach().cpu().numpy()
        if isinstance(got, torch.Tensor):
            got = got.detach().cpu().numpy()

        if verbose >= 6:
            print(f"[max_diff] tensor: {string_type(expected)} ? {string_type(got)}")

        if _index < begin or (end != -1 and _index >= end):
            # out of boundary
            return dict(abs=0.0, rel=0.0, sum=0.0, n=0.0, dnan=0)
        if isinstance(expected, (int, float)):
            if isinstance(got, np.ndarray) and len(got.shape) == 0:
                got = float(got)
            if isinstance(got, (int, float)):
                if expected == got:
                    return dict(abs=0.0, rel=0.0, sum=0.0, n=0.0, dnan=0)
                return dict(
                    abs=abs(expected - got),
                    rel=abs(expected - got) / (abs(expected) + 1e-5),
                    sum=abs(expected - got),
                    n=1,
                    dnan=0,
                )
            return dict(abs=np.inf, rel=np.inf, sum=np.inf, n=np.inf, dnan=np.inf)
        if expected.dtype in (np.complex64, np.complex128):
            if got.dtype == expected.dtype:
                got = np.real(got)
            elif got.dtype not in (np.float32, np.float64):
                if verbose >= 10:
                    # To understand the value it comes from.
                    if debug_info:
                        print("\n".join(debug_info))
                    print(
                        f"[max_diff-c] expected.dtype={expected.dtype}, "
                        f"got.dtype={got.dtype}"
                    )
                return dict(abs=np.inf, rel=np.inf, sum=np.inf, n=np.inf, dnan=np.inf)
            expected = np.real(expected)

        if expected.shape != got.shape:
            if verbose >= 10:
                # To understand the value it comes from.
                if debug_info:
                    print("\n".join(debug_info))
                print(f"[max_diff-s] expected.shape={expected.shape}, got.shape={got.shape}")
            return dict(abs=np.inf, rel=np.inf, sum=np.inf, n=np.inf, dnan=np.inf)
        # nan are replace by 1e10, any discrepancies in that order of magnitude
        # is likely caused by nans
        exp_cpu = np.nan_to_num(expected.astype(np.float64), nan=1e10)
        got_cpu = np.nan_to_num(got.astype(np.float64), nan=1e10)
        diff = np.abs(got_cpu - exp_cpu)
        ndiff = np.abs(np.isnan(expected).astype(int) - np.isnan(got).astype(int))
        rdiff = diff / (np.abs(exp_cpu) + 1e-3)
        if diff.size == 0:
            abs_diff, rel_diff, sum_diff, n_diff, nan_diff = (
                (0, 0, 0, 0, 0)
                if exp_cpu.size == got_cpu.size
                else (np.inf, np.inf, np.inf, 0, np.inf)
            )
        else:
            abs_diff, rel_diff, sum_diff, n_diff, nan_diff = (
                float(diff.max()),
                float(rdiff.max()),
                float(diff.sum()),
                float(diff.size),
                float(ndiff.sum()),
            )
        if verbose >= 10 and (abs_diff >= 10 or rel_diff >= 10):
            # To understand the value it comes from.
            if debug_info:
                print("\n".join(debug_info))
            print(
                f"[max_diff-1] abs_diff={abs_diff}, rel_diff={rel_diff}, "
                f"nan_diff={nan_diff}, dtype={expected.dtype}, "
                f"shape={expected.shape}, level={level}, _index={_index}"
            )
            if abs_diff >= 10:
                idiff = np.argmax(diff.reshape((-1,)))
                x = expected.reshape((-1,))[idiff]
                y = got.reshape((-1,))[idiff]
                print(
                    f"   [max_diff-2] abs diff={abs_diff}, "
                    f"x={x}, y={y}, level={level}, "
                    f"_index={_index}"
                )
                print(y)

            if rel_diff >= 10:
                idiff = np.argmax(rdiff.reshape((-1,)))
                x = expected.reshape((-1,))[idiff]
                y = got.reshape((-1,))[idiff]
                print(
                    f"   [max_diff-3] rel diff={rel_diff}, "
                    f"x={x}, y={y}, level={level}, "
                    f"_index={_index}"
                )

        return dict(abs=abs_diff, rel=rel_diff, sum=sum_diff, n=n_diff, dnan=nan_diff)

    if isinstance(expected, torch.Tensor) and isinstance(got, torch.Tensor):
        if verbose >= 6:
            print(f"[max_diff] tensor: {string_type(expected)} ? {string_type(got)}")
        if _index < begin or (end != -1 and _index >= end):
            # out of boundary
            return dict(abs=0.0, rel=0.0, sum=0.0, n=0.0, dnan=0)
        if expected.dtype in (torch.complex64, torch.complex128):
            if got.dtype == expected.dtype:
                got = torch.view_as_real(got)
            elif got.dtype not in (torch.float32, torch.float64):
                if verbose >= 10:
                    # To understand the value it comes from.
                    if debug_info:
                        print("\n".join(debug_info))
                    print(
                        f"[max_diff-c] expected.dtype={expected.dtype}, "
                        f"got.dtype={got.dtype}"
                    )
                return dict(abs=np.inf, rel=np.inf, sum=np.inf, n=np.inf, dnan=np.inf)
            expected = torch.view_as_real(expected)

        if expected.shape != got.shape:
            if verbose >= 10:
                # To understand the value it comes from.
                if debug_info:
                    print("\n".join(debug_info))
                print(f"[max_diff-s] expected.shape={expected.shape}, got.shape={got.shape}")
            return dict(abs=np.inf, rel=np.inf, sum=np.inf, n=np.inf, dnan=np.inf)
        # nan are replace by 1e10, any discrepancies in that order of magnitude
        # is likely caused by nans
        exp_cpu = expected.to(torch.float64).cpu().nan_to_num(1e10)
        got_cpu = got.to(torch.float64).cpu().nan_to_num(1e10)
        diff = (got_cpu - exp_cpu).abs()
        ndiff = (expected.isnan().cpu().to(int) - got.isnan().cpu().to(int)).abs()
        rdiff = diff / (exp_cpu.abs() + 1e-3)
        if diff.numel() > 0:
            abs_diff, rel_diff, sum_diff, n_diff, nan_diff = (
                float(diff.max()),
                float(rdiff.max()),
                float(diff.sum()),
                float(diff.numel()),
                float(ndiff.sum()),
            )
        elif got_cpu.numel() == exp_cpu.numel():
            abs_diff, rel_diff, sum_diff, n_diff, nan_diff = (0.0, 0.0, 0.0, 0.0, 0.0)
        else:
            abs_diff, rel_diff, sum_diff, n_diff, nan_diff = (
                np.inf,
                np.inf,
                np.inf,
                np.inf,
                np.inf,
            )

        if verbose >= 10 and (abs_diff >= 10 or rel_diff >= 10):
            # To understand the value it comes from.
            if debug_info:
                print("\n".join(debug_info))
            print(
                f"[max_diff-1] abs_diff={abs_diff}, rel_diff={rel_diff}, "
                f"nan_diff={nan_diff}, dtype={expected.dtype}, "
                f"shape={expected.shape}, level={level}, _index={_index}"
            )
            if abs_diff >= 10:
                idiff = torch.argmax(diff.reshape((-1,)))
                x = expected.reshape((-1,))[idiff]
                y = got.reshape((-1,))[idiff]
                print(
                    f"   [max_diff-2] abs diff={abs_diff}, "
                    f"x={x}, y={y}, level={level}, "
                    f"_index={_index}"
                )
                print(y)

            if rel_diff >= 10:
                idiff = torch.argmax(rdiff.reshape((-1,)))
                x = expected.reshape((-1,))[idiff]
                y = got.reshape((-1,))[idiff]
                print(
                    f"   [max_diff-3] rel diff={rel_diff}, "
                    f"x={x}, y={y}, level={level}, "
                    f"_index={_index}"
                )

        return dict(abs=abs_diff, rel=rel_diff, sum=sum_diff, n=n_diff, dnan=nan_diff)

    if "SquashedNormal" in expected.__class__.__name__:
        if verbose >= 6:
            print(f"[max_diff] SquashedNormal: {string_type(expected)} ? {string_type(got)}")
        values = (
            expected.mean.detach().to("cpu"),
            expected.scale.detach().to("cpu"),
        )
        return max_diff(
            values,
            got,
            verbose=verbose,
            level=level + 1,
            begin=begin,
            end=end,
            _index=_index,
            flatten=flatten,
        )

    if expected.__class__ in torch.utils._pytree.SUPPORTED_NODES:
        if got.__class__ not in torch.utils._pytree.SUPPORTED_NODES:
            return dict(abs=np.inf, rel=np.inf, sum=np.inf, n=np.inf, dnan=np.inf)
        if verbose >= 6:
            print(
                f"[max_diff] {expected.__class__.__name__}: "
                f"{string_type(expected)} ? {string_type(got)}"
            )
        expected_args, _spec = torch.utils._pytree.tree_flatten(expected)
        got_args, _spec = torch.utils._pytree.tree_flatten(got)
        return max_diff(
            expected_args,
            got_args,
            level=level,
            flatten=flatten,
            debug_info=debug_info,
            begin=begin,
            end=end,
            _index=_index,
            verbose=verbose,
        )

    # backup function in case pytorch does not know how to serialize.
    if expected.__class__.__name__ == "DynamicCache":
        if got.__class__.__name__ == "DynamicCache":
            if verbose >= 6:
                print(f"[max_diff] DynamicCache: {string_type(expected)} ? {string_type(got)}")
            return max_diff(
                [expected.key_cache, expected.value_cache],
                [got.key_cache, got.value_cache],
                verbose=verbose,
            )
        if isinstance(got, tuple) and len(got) == 2:
            return max_diff(
                [expected.key_cache, expected.value_cache],
                [got[0], got[1]],
                verbose=verbose,
            )
        raise AssertionError(
            f"DynamicCache not fully implemented with classes "
            f"{expected.__class__.__name__!r} and {got.__class__.__name__!r}, "
            f"and expected={string_type(expected)}, got={string_type(got)},\n"
            f"level={level}"
        )

    if expected.__class__.__name__ == "SlidingWindowCache":
        if got.__class__.__name__ == "SlidingWindowCache":
            if verbose >= 6:
                print(f"[max_diff] DynamicCache: {string_type(expected)} ? {string_type(got)}")
            return max_diff(
                [expected.key_cache, expected.value_cache],
                [got.key_cache, got.value_cache],
                verbose=verbose,
            )
        if isinstance(got, tuple) and len(got) == 2:
            return max_diff(
                [expected.key_cache, expected.value_cache],
                [got[0], got[1]],
                verbose=verbose,
            )
        raise AssertionError(
            f"SlidingWindowCache not fully implemented with classes "
            f"{expected.__class__.__name__!r} and {got.__class__.__name__!r}, "
            f"and expected={string_type(expected)}, got={string_type(got)},\n"
            f"level={level}"
        )

    if expected.__class__.__name__ == "EncoderDecoderCache":
        if got.__class__.__name__ == "EncoderDecoderCache":
            if verbose >= 6:
                print(
                    f"[max_diff] EncoderDecoderCache: "
                    f"{string_type(expected)} ? {string_type(got)}"
                )
            return max_diff(
                [expected.self_attention_cache, expected.cross_attention_cache],
                [got.self_attention_cache, got.cross_attention_cache],
                verbose=verbose,
            )
        if isinstance(got, tuple) and len(got) == 2:
            return max_diff(
                [expected.self_attention_cache, expected.cross_attention_cache],
                [got[0], got[1]],
                verbose=verbose,
            )
        raise AssertionError(
            f"EncoderDecoderCache not fully implemented with classes "
            f"{expected.__class__.__name__!r} and {got.__class__.__name__!r}, "
            f"and expected={string_type(expected)}, got={string_type(got)},\n"
            f"level={level}"
        )

    if expected.__class__.__name__ in ("transformers.cache_utils.MambaCache", "MambaCache"):
        if verbose >= 6:
            print(f"[max_diff] MambaCache: {string_type(expected)} ? {string_type(got)}")
        if got.__class__.__name__ != expected.__class__.__name__:
            # This case happens with onnx where the outputs are flattened.
            return dict(abs=np.inf, rel=np.inf, sum=np.inf, n=np.inf, dnan=np.inf)
        atts = []
        for k in ["conv_states", "ssm_states"]:
            if hasattr(expected, k) and not hasattr(got, k):
                return dict(abs=np.inf, rel=np.inf, sum=np.inf, n=np.inf, dnan=np.inf)
            atts.append(k)

        return max_diff(
            [getattr(expected, k) for k in atts],
            [getattr(got, k) for k in atts],
            level=level,
            flatten=flatten,
            debug_info=debug_info,
            begin=begin,
            end=end,
            _index=_index,
            verbose=verbose,
        )

    raise AssertionError(
        f"Not implemented with implemented with expected="
        f"{string_type(expected)}, got={string_type(got)},\n"
        f"level={level}"
    )


def string_diff(diff: Dict[str, Any]) -> str:
    """Renders discrepancies return by :func:`max_diff` into one string."""
    # dict(abs=, rel=, sum=, n=n_diff, dnan=)
    if diff.get("dnan", None):
        if diff["abs"] == 0 or diff["rel"] == 0:
            return f"abs={diff['abs']}, rel={diff['rel']}, dnan={diff['dnan']}"
        return f"abs={diff['abs']}, rel={diff['rel']}, n={diff['n']}, dnan={diff['dnan']}"
    if diff["abs"] == 0 or diff["rel"] == 0:
        return f"abs={diff['abs']}, rel={diff['rel']}"
    return f"abs={diff['abs']}, rel={diff['rel']}, n={diff['n']}"
