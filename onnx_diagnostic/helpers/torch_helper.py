import contextlib
import ctypes
import inspect
import math
import os
import sys
import warnings
from collections.abc import Iterable
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import onnx
from onnx.external_data_helper import load_external_data_for_tensor, uses_external_data
import torch
from .helper import string_type, size_type
from .cache_helper import (
    make_dynamic_cache,
    make_encoder_decoder_cache,
    make_static_cache,
    CacheKeyValue,
)
from .mini_onnx_builder import create_onnx_model_from_input_tensors
from .onnx_helper import (
    to_array_extended,
    tensor_dtype_to_np_dtype,
    _STORAGE_TYPE,
    onnx_dtype_name,
)


def proto_from_tensor(
    arr: torch.Tensor, name: Optional[str] = None, verbose: int = 0
) -> onnx.TensorProto:
    """
    Converts a torch Tensor into a TensorProto.

    :param arr: tensor
    :param verbose: display the type and shape
    :return: a TensorProto
    """
    import torch

    if not isinstance(arr, torch.Tensor):
        raise TypeError(f"Unexpected type {type(arr)}.")
    if arr.is_sparse:
        raise NotImplementedError(
            f"Sparse tensor is not supported yet but initializer {name!r} is."
        )

    # arr.contiguous() is slow after a transpose, maybe there is a way to optimize this.
    if arr.is_contiguous():
        arr_cpu = arr.cpu()
    else:
        arr_cpu = arr.contiguous().cpu()

    numel = torch.numel(arr_cpu)
    element_size = arr_cpu.element_size()

    if arr_cpu.dtype in {torch.bfloat16}:
        np_arr = arr_cpu
    elif arr_cpu.data_ptr() == arr.data_ptr():
        copy = arr_cpu.clone().detach().requires_grad_(False)
        assert (
            arr_cpu.data_ptr() == 0 or arr_cpu.data_ptr() != copy.data_ptr()
        ), f"Pointers are not null and different {arr_cpu.data_ptr()} != {copy.data_ptr()}"
        np_arr = np.from_dlpack(copy)
    else:
        np_arr = np.from_dlpack(arr_cpu.detach())

    tensor = onnx.TensorProto()
    tensor.dims.extend(arr_cpu.shape)
    if name:
        tensor.name = name
    itype = torch_dtype_to_onnx_dtype(arr_cpu.dtype)
    assert not hasattr(onnx.TensorProto, "INT4") or itype not in {
        onnx.TensorProto.INT4,
        onnx.TensorProto.UINT4,
    }, f"Type {arr.dtype} is not supported yet for name={name!r}"
    tensor.data_type = itype

    if verbose > 1 and numel > 100:
        print(f"[proto_from_array] {tensor.data_type}[{arr_cpu.shape}]")

    if isinstance(np_arr, torch.Tensor):
        byte_data = (ctypes.c_ubyte * numel * element_size).from_address(np_arr.data_ptr())
        tensor.raw_data = bytes(byte_data)
        if sys.byteorder == "big":
            np_dtype = _STORAGE_TYPE[tensor.data_type]  # type: ignore
            np.byteswap(np.frombuffer(tensor.raw_data, dtype=np_dtype), inplace=True)  # type: ignore
    else:
        tensor.raw_data = np_arr.tobytes()
        if sys.byteorder == "big":
            np_dtype = tensor_dtype_to_np_dtype(tensor.data_type)
            np.byteswap(np.frombuffer(tensor.raw_data, dtype=np_dtype), inplace=True)
    return tensor


def onnx_dtype_to_torch_dtype(itype: int) -> torch.dtype:
    """
    Converts an onnx type into a torch dtype.

    :param to: onnx dtype
    :return: torch dtype
    """
    if itype == onnx.TensorProto.FLOAT:
        return torch.float32
    if itype == onnx.TensorProto.FLOAT16:
        return torch.float16
    if itype == onnx.TensorProto.BFLOAT16:
        return torch.bfloat16
    if itype == onnx.TensorProto.DOUBLE:
        return torch.float64
    if itype == onnx.TensorProto.INT32:
        return torch.int32
    if itype == onnx.TensorProto.INT64:
        return torch.int64
    if itype == onnx.TensorProto.UINT32:
        return torch.uint32
    if itype == onnx.TensorProto.UINT64:
        return torch.uint64
    if itype == onnx.TensorProto.BOOL:
        return torch.bool
    if itype == onnx.TensorProto.INT16:
        return torch.int16
    if itype == onnx.TensorProto.UINT16:
        return torch.uint16
    if itype == onnx.TensorProto.INT8:
        return torch.int8
    if itype == onnx.TensorProto.UINT8:
        return torch.uint8
    if itype == onnx.TensorProto.COMPLEX64:
        return torch.complex64
    if itype == onnx.TensorProto.COMPLEX128:
        return torch.complex128
    raise NotImplementedError(
        f"Unable to convert onnx type {onnx_dtype_name(itype)} to torch.type."
    )


_TYPENAME = dict(
    FLOAT=onnx.TensorProto.FLOAT,
    INT64=onnx.TensorProto.INT64,
    INT32=onnx.TensorProto.INT32,
    FLOAT16=onnx.TensorProto.FLOAT16,
    BFLOAT16=onnx.TensorProto.BFLOAT16,
)


def torch_dtype_to_onnx_dtype(to: torch.dtype) -> int:
    """
    Converts a torch dtype into a onnx element type.

    :param to: torch dtype
    :return: onnx type
    """
    import torch

    if to == torch.float32:
        return onnx.TensorProto.FLOAT
    if to == torch.float16:
        return onnx.TensorProto.FLOAT16
    if to == torch.bfloat16:
        return onnx.TensorProto.BFLOAT16
    if to == torch.float64:
        return onnx.TensorProto.DOUBLE
    if to == torch.int64:
        return onnx.TensorProto.INT64
    if to == torch.int32:
        return onnx.TensorProto.INT32
    if to == torch.uint64:
        return onnx.TensorProto.UINT64
    if to == torch.uint32:
        return onnx.TensorProto.UINT32
    if to == torch.bool:
        return onnx.TensorProto.BOOL
    if to == torch.SymInt:
        return onnx.TensorProto.INT64
    if to == torch.int16:
        return onnx.TensorProto.INT16
    if to == torch.uint16:
        return onnx.TensorProto.UINT16
    if to == torch.int8:
        return onnx.TensorProto.INT8
    if to == torch.uint8:
        return onnx.TensorProto.UINT8
    if to == torch.SymFloat:
        return onnx.TensorProto.FLOAT
    if to == torch.complex64:
        return onnx.TensorProto.COMPLEX64
    if to == torch.complex128:
        return onnx.TensorProto.COMPLEX128
    # SymbolicTensor
    sto = str(to)
    if sto in _TYPENAME:
        return _TYPENAME[sto]
    raise NotImplementedError(
        f"Unable to convert torch dtype {to!r} ({type(to)}) to onnx dtype."
    )


def _forward_(
    *args,
    _f=None,
    _fprint=string_type,
    _prefix="",
    _context=None,
    _storage=None,
    _storage_limit=2**27,
    _verbose=0,
    **kwargs,
):
    assert _f is not None, "_f cannot be None"
    assert _context is not None, "_context cannot be None"
    indent = "  " * (len(_prefix) - len(_prefix.lstrip()))
    _prefix = _prefix.lstrip()
    print(
        f"{indent}+{_prefix} -- stolen forward for class {_context['class_name']} "
        f"-- iteration {_context['iteration']}"
    )
    kws = dict(
        with_shape=_context.get("with_shape", False),
        with_min_max=_context.get("with_min_max", False),
    )
    if not hasattr(torch.compiler, "is_exporting") or not torch.compiler.is_exporting():
        # torch.compiler.is_exporting requires torch>=2.7
        print(f"{indent}  <- args={_fprint(args, **kws)} --- kwargs={_fprint(kwargs, **kws)}")
    if _storage is not None:
        it = _context["iteration"]
        key = (_prefix, it)
        _storage[(*key, "I")] = (torch_deepcopy(args), torch_deepcopy(kwargs))
    res = _f(*args, **kwargs)
    if not hasattr(torch.compiler, "is_exporting") or not torch.compiler.is_exporting():
        print(f"{indent}  -> {_fprint(res, **kws)}")
        print(f"{indent}-{_prefix}.")
    if _storage is not None:
        size = torch_tensor_size(res)
        if size < _storage_limit:
            if _verbose:
                print(
                    f"-- stores key={key}, size {size // 2**10}Kb -- "
                    f"{string_type(res, with_shape=True)}"
                )
            _storage[(*key, "O")] = torch_deepcopy(res)
        else:
            if _verbose:
                print(
                    f"-- skips key={key}, size {size // 2**10}Kb -- "
                    f"{string_type(res, with_shape=True)}"
                )
    _context["iteration"] += 1
    return res


_steal_forward_status = [False]
_additional_stolen_objects = {}


def is_stealing() -> bool:
    """Returns true if :func:`steal_forward` was yielded."""
    return _steal_forward_status[0]


def steal_append(name: str, obj: Any):
    """
    When outside a forward method, it is still possible to add
    a python object which contains tensors and dump after the execution
    of the model.

    .. code-block:: python

        steal_append("quantize", [t1, t2])

    The same code can executed multiple times, then
    the name can extended with a number.
    """
    if is_stealing():
        if name in _additional_stolen_objects:
            i = 1
            n = f"{name}_{i}"
            while n in _additional_stolen_objects:
                i += 1
                n = f"{name}_{i}"
            print(f"-- stolen {name!r} renamed in {n!r}: {string_type(obj, with_shape=True)}")
            _additional_stolen_objects[n] = obj
        else:
            print(f"-- stolen {name!r}: {string_type(obj, with_shape=True)}")
            _additional_stolen_objects[name] = obj


@contextlib.contextmanager
def steal_forward(
    model: Union[
        Union[torch.nn.Module, Tuple[str, torch.nn.Module]],
        List[Union[torch.nn.Module, Tuple[str, torch.nn.Module]]],
    ],
    fprint: Callable = string_type,
    dump_file: Optional[str] = None,
    dump_drop: Optional[Set[str]] = None,
    submodules: bool = False,
    verbose: int = 0,
    storage_limit: int = 2**27,
    save_as_external_data: bool = True,
    **kwargs,
):
    """
    The necessary modification to steem forward method and prints out inputs
    and outputs using :func:`onnx_diagnostic.helpers.string_type`.
    See example :ref:`l-plot-tiny-llm-export` or
    :ref:`l-plot-intermediate-results`.

    :param model: a model or a list of models to monitor,
        every model can also be a tuple(name, model), name is displayed well.
    :param fprint: function used to print out (or dump), by default, it is
        :func:`onnx_diagnostic.helpers.string_type`
    :param kwargs: additional parameters sent to :func:`onnx_diagnostic.helpers.string_type`
        or any other function defined by ``fprint``
    :param dump_file: dumps stolen inputs and outputs in an onnx model,
        they can be restored with :func:`create_input_tensors_from_onnx_model
        <onnx_diagnostic.helpers.mini_onnx_builder.create_input_tensors_from_onnx_model>`
    :param dump_drop: to drop some inputs too big (only if dump_file is specified)
    :param save_as_external_data: True by default, but maybe better to have everything
        in a single file if possible
    :param submodules: if True and model is a module, the list extended with all the submodules
        the module contains
    :param verbose: verbosity
    :param storage_limit: do not stored object bigger than this

    The following examples shows how to steal and dump all the inputs / outputs
    for a module and its submodules, then restores them.

    .. runpython::
        :showcode:

        import torch
        from onnx_diagnostic.helpers.torch_helper import steal_forward
        from onnx_diagnostic.helpers.mini_onnx_builder import (
            create_input_tensors_from_onnx_model,
        )

        class SubModel(torch.nn.Module):
            def forward(self, x):
                return x * x

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.s1 = SubModel()
                self.s2 = SubModel()

            def forward(self, x, y):
                return self.s1(x) + self.s2(y)

        inputs = torch.rand(2, 1), torch.rand(2, 1)
        model = Model()
        dump_file = "dump_steal_forward_submodules.onnx"
        with steal_forward(model, submodules=True, dump_file=dump_file):
            model(*inputs)

        # Let's restore the stolen data.
        restored = create_input_tensors_from_onnx_model(dump_file)
        for k, v in sorted(restored.items()):
            if isinstance(v, tuple):
                args, kwargs = v
                print("input", k, args, kwargs)
            else:
                print("output", k, v)

    Function :func:`steal_append` can be used to dump more tensors.
    When inside the context, func:`is_stealing` returns True, False otherwise.
    """
    assert not is_stealing(), "steal_forward was already called."
    # We clear the cache.
    _steal_forward_status[0] = True
    _additional_stolen_objects.clear()
    assert not submodules or isinstance(
        model, torch.nn.Module
    ), f"submodules can only be True if model is a module but is is {type(model)}."
    context = dict(iteration=0, **kwargs)
    if "with_shape" not in context and fprint == string_type:
        context["with_shape"] = True
    if not isinstance(model, list):
        assert isinstance(model, torch.nn.Module), f"Unexpected type {type(model)} for model"
        if submodules:
            models = []
            for idx, m in model.named_modules():
                level = str(idx).split(".")
                ll = len(level)
                try:
                    _, start_line = inspect.getsourcelines(m.forward)
                except OSError:
                    # The code is not available.
                    start_line = 0
                name = f"{idx}-{m.__class__.__name__}-{start_line}"
                models.append((f"{'  ' * ll}{name}", m))
            model = models
        else:
            model = [model]
    keep_model_forward = {}
    storage: Optional[Dict[Any, Any]] = {} if dump_file else None
    for mt in model:
        name, m = mt if isinstance(mt, tuple) else ("", mt)
        keep_model_forward[id(m)] = (m, m.forward)
        c = context.copy()
        c["class_name"] = m.__class__.__name__
        m.forward = lambda *args, _f=m.forward, _fp=fprint, _c=c, _p=name, _s=storage, _v=verbose, _sl=storage_limit, **kws: _forward_(  # noqa: E501
            *args,
            _f=_f,
            _fprint=_fp,
            _context=_c,
            _prefix=_p,
            _storage=_s,
            _verbose=_v,
            _storage_limit=_sl,
            **kws,
        )
    try:
        yield
    finally:
        _steal_forward_status[0] = False
        for f in keep_model_forward.values():
            f[0].forward = f[1]
        if dump_file:
            # Let's add the cached tensor
            assert storage is not None, "storage cannot be None but mypy is confused here."
            storage.update(_additional_stolen_objects)
            # We clear the cache.
            _additional_stolen_objects.clear()
            if verbose:
                size = torch_tensor_size(storage)
                print(f"-- gather stored {len(storage)} objects, size={size // 2 ** 20} Mb")
            if dump_drop:
                for k, v in storage.items():
                    if k[-1] == "I":
                        _args, kwargs = v
                        ii = set(kwargs) & dump_drop
                        if ii:
                            for i in ii:
                                print("---", i)
                                del kwargs[i]
            proto = create_onnx_model_from_input_tensors(storage)
            if verbose:
                print("-- dumps stored objects")
            location = f"{os.path.split(dump_file)[-1]}.data"
            if os.path.exists(location):
                os.remove(location)
            onnx.save(
                proto,
                dump_file,
                save_as_external_data=save_as_external_data,
                all_tensors_to_one_file=True,
                location=location,
            )
            if verbose:
                print("-- done dump stored objects")


@contextlib.contextmanager
def fake_torchdynamo_exporting():
    """
    Sets ``torch.compiler._is_exporting_flag`` to True to trigger
    pieces of code only enabled during export.
    """
    memorize = torch.compiler._is_exporting_flag
    torch.compiler._is_exporting_flag = True
    assert torch.compiler.is_exporting(), (
        f"Changes not detected "
        f"torch.compiler._is_exporting_flag={torch.compiler._is_exporting_flag} "
        f"and torch.compiler.is_exporting()={torch.compiler.is_exporting()}"
    )
    try:
        yield
    finally:
        torch.compiler._is_exporting_flag = memorize


def is_torchdynamo_exporting() -> bool:
    """
    Tells if :epkg:`torch` is exporting a model.
    Relies on ``torch.compiler.is_exporting()``.
    """
    import torch

    if not hasattr(torch.compiler, "is_exporting"):
        # torch.compiler.is_exporting requires torch>=2.7
        return False

    try:
        return torch.compiler.is_exporting()
    except Exception:
        try:
            import torch._dynamo as dynamo

            return dynamo.is_exporting()  # type: ignore
        except Exception:
            return False


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Converts a :class:`torch.Tensor` to :class:`numpy.ndarray`."""
    try:
        return tensor.detach().cpu().numpy()
    except TypeError:
        # We try with ml_dtypes
        pass

    import ml_dtypes

    conv = {torch.bfloat16: ml_dtypes.bfloat16}
    assert tensor.dtype in conv, f"Unsupported type {tensor.dtype}, not in {conv}"
    return tensor.detach().to(torch.float32).cpu().numpy().astype(conv[tensor.dtype])


def from_numpy(tensor: np.ndarray) -> torch.Tensor:
    """Converts a :class:`numpy.ndarray` to :class:`torch.Tensor`."""
    try:
        return torch.from_numpy(tensor)
    except TypeError:
        # We try with ml_dtypes
        pass

    import ml_dtypes

    conv = {ml_dtypes.bfloat16: torch.bfloat16}
    assert tensor.dtype in conv, f"Unsupported type {tensor.dtype}, not in {conv}"
    return torch.from_numpy(tensor.astype(torch.float32)).to(conv[tensor.dtype])


def replace_string_by_dynamic(dynamic_shapes: Any) -> Any:
    """Replaces strings by ``torch.export.Dim.DYNAMIC``."""
    import torch

    if isinstance(dynamic_shapes, torch.export.dynamic_shapes._Dim):
        return dynamic_shapes
    if isinstance(dynamic_shapes, str):
        return torch.export.Dim.DYNAMIC
    if not dynamic_shapes:
        return dynamic_shapes
    if isinstance(dynamic_shapes, (tuple, list)):
        return type(dynamic_shapes)(replace_string_by_dynamic(i) for i in dynamic_shapes)
    if isinstance(dynamic_shapes, dict):
        return {k: replace_string_by_dynamic(v) for k, v in dynamic_shapes.items()}
    raise AssertionError(f"Unexpected type {type(dynamic_shapes)} for dynamic_shapes")


def dummy_llm(
    cls_name: Optional[str] = None,
    dynamic_shapes: bool = False,
) -> Union[
    Tuple[torch.nn.Module, Tuple[torch.Tensor, ...]],
    Tuple[torch.nn.Module, Tuple[torch.Tensor, ...], Any],
]:
    """
    Creates a dummy LLM for test purposes.

    :param cls_name: None for whole model or a piece of it
    :param dynamic_shapes: returns dynamic shapes as well

    .. runpython::
        :showcode:

        from onnx_diagnostic.helpers.torch_helper import dummy_llm
        print(dummy_llm())
    """

    class Embedding(torch.nn.Module):
        def __init__(self, vocab_size: int = 1024, embedding_dim: int = 16):
            super().__init__()
            self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
            self.pe = torch.nn.Embedding(vocab_size, embedding_dim)

        def forward(self, x):
            word_emb = self.embedding(x)
            word_pe = self.pe(x)
            return word_emb + word_pe

    class AttentionBlock(torch.nn.Module):

        def __init__(self, embedding_dim: int = 16, context_size: int = 256):
            super().__init__()
            self.query = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)
            self.key = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)
            self.value = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)
            # torch.nn.Buffer are not fully handled by symbolic tracing
            # Buffer(...)[:Prowy()] is not working
            self.mask = torch.nn.Parameter(
                torch.tril(
                    input=torch.ones(size=[context_size, context_size], dtype=torch.float)
                )
            )

        def forward(self, x):
            _B, T, C = x.shape

            query = self.query(x)
            key = self.key(x)
            value = self.value(x)

            qk = query @ key.transpose(-2, -1) * C**-0.5
            attention = qk.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
            attention = torch.nn.functional.softmax(input=attention, dim=-1)

            out = attention @ value
            return out

    class MultiAttentionBlock(torch.nn.Module):

        def __init__(
            self, embedding_dim: int = 16, num_heads: int = 2, context_size: int = 256
        ):
            super().__init__()
            self.attention = torch.nn.ModuleList(
                modules=[AttentionBlock(embedding_dim, context_size) for _ in range(num_heads)]
            )
            self.linear = torch.nn.Linear(
                in_features=embedding_dim * num_heads, out_features=embedding_dim
            )

        def forward(self, x):
            out = torch.cat(tensors=[attention(x) for attention in self.attention], dim=-1)
            x = self.linear(out)
            return x

    class FeedForward(torch.nn.Module):

        def __init__(self, embedding_dim: int = 16, ff_dim: int = 128):
            super().__init__()
            self.linear_1 = torch.nn.Linear(embedding_dim, ff_dim)
            self.relu = torch.nn.ReLU()
            self.linear_2 = torch.nn.Linear(ff_dim, embedding_dim)

        def forward(self, x):
            x = self.linear_1(x)
            x = self.relu(x)
            x = self.linear_2(x)
            return x

    class DecoderLayer(torch.nn.Module):

        def __init__(
            self,
            embedding_dim: int = 16,
            num_heads: int = 2,
            context_size: int = 256,
            ff_dim: int = 128,
        ):
            super().__init__()
            self.attention = MultiAttentionBlock(embedding_dim, num_heads, context_size)
            self.feed_forward = FeedForward(embedding_dim, ff_dim)
            self.norm_1 = torch.nn.LayerNorm(normalized_shape=embedding_dim)
            self.norm_2 = torch.nn.LayerNorm(normalized_shape=embedding_dim)

        def forward(self, x):
            x_norm = self.norm_1(x)
            attention = self.attention(x_norm)
            attention = attention + x

            attention_norm = self.norm_2(attention)
            ff = self.feed_forward(attention_norm)
            ff = ff + attention

            return ff

    class LLM(torch.nn.Module):

        def __init__(
            self,
            vocab_size: int = 1024,
            embedding_dim: int = 16,
            num_heads: int = 2,
            context_size: int = 256,
            ff_dim: int = 128,
        ):
            super().__init__()
            self.embedding = Embedding(vocab_size, embedding_dim)
            self.decoder = DecoderLayer(embedding_dim, num_heads, context_size, ff_dim)

        def forward(self, input_ids):
            x = self.embedding(input_ids)
            y = self.decoder(x)
            return y

    if cls_name in (None, "LLM"):
        dec: torch.nn.Module = LLM()
        x = torch.randint(0, 1024, (2 if dynamic_shapes else 1, 30)).to(torch.int64)
        dec(x)
        if dynamic_shapes:
            dyn = {
                "input_ids": {
                    0: torch.export.Dim("batch", min=1, max=1024),
                    1: torch.export.Dim("length", min=1, max=255),
                }
            }
            return dec, (x,), dyn
        return dec, (x,)

    if cls_name == "DecoderLayer":
        LLM()(torch.randint(0, 1024, (2 if dynamic_shapes else 1, 30)).to(torch.int64))

        dec = DecoderLayer()
        x = Embedding()(
            torch.randint(0, 1024, (2 if dynamic_shapes else 1, 30)).to(torch.int64)
        )
        dec(x)
        if dynamic_shapes:
            dyn = {
                "x": {
                    0: torch.export.Dim("batch", min=1, max=1024),
                    1: torch.export.Dim("length", min=1, max=255),
                }
            }
            return dec, (x,), dyn
        return dec, (x,)

    if cls_name == "MultiAttentionBlock":
        dec = MultiAttentionBlock()
        x = torch.rand(2 if dynamic_shapes else 1, 30, 16).to(torch.float32)
        dec(x)
        if dynamic_shapes:
            dyn = {
                "x": {
                    0: torch.export.Dim("batch", min=1, max=1024),
                    1: torch.export.Dim("length", min=1, max=255),
                }
            }
            return dec, (x,), dyn
        return dec, (x,)

    if cls_name == "AttentionBlock":
        dec = AttentionBlock()
        x = torch.rand(2 if dynamic_shapes else 1, 30, 16).to(torch.float32)
        dec(x)
        if dynamic_shapes:
            dyn = {
                "x": {
                    0: torch.export.Dim("batch", min=1, max=1024),
                    1: torch.export.Dim("length", min=1, max=255),
                }
            }
            return dec, (x,), dyn
        return dec, (x,)

    raise NotImplementedError(f"cls_name={cls_name}")


def to_any(value: Any, to_value: Union[torch.dtype, torch.device, str]) -> Any:
    """Applies torch.to if applicable. Goes recursively."""
    if isinstance(value, (torch.nn.Module, torch.Tensor)) and value.__class__.__name__ not in {
        "DynamicCache",
        "EncoderDecoderCache",
    }:
        if (
            (
                isinstance(to_value, torch.dtype)
                or to_value in {"float16", "bfloat16", "float32", "float64"}
            )
            and hasattr(value, "dtype")
            and value.dtype in {torch.int32, torch.int64, torch.int8, torch.int16}
        ):
            # int vector should not be changed.
            return value
        return value.to(to_value)
    if isinstance(value, list):
        return [to_any(t, to_value) for t in value]
    if isinstance(value, tuple):
        return tuple(to_any(t, to_value) for t in value)
    if isinstance(value, set):
        return {to_any(t, to_value) for t in value}
    if type(value) is dict:
        return {k: to_any(t, to_value) for k, t in value.items()}
    if value.__class__.__name__ == "DynamicCache":
        cc = CacheKeyValue(value)
        return make_dynamic_cache(
            list(
                zip(
                    [t.to(to_value) if t is not None else t for t in cc.key_cache],
                    [t.to(to_value) if t is not None else t for t in cc.value_cache],
                )
            ),
            cls_layers=cc.cls_layers,
        )
    if value.__class__.__name__ in "HybridCache":
        from .cache_helper import make_hybrid_cache

        cc = CacheKeyValue(value)
        return make_hybrid_cache(
            list(
                zip(
                    [t.to(to_value) if t is not None else t for t in cc.key_cache],
                    [t.to(to_value) if t is not None else t for t in cc.value_cache],
                )
            )
        )
    if value.__class__.__name__ == "StaticCache":
        cc = CacheKeyValue(value)
        return make_static_cache(
            list(
                zip(
                    [t.to(to_value) if t is not None else t for t in cc.key_cache],
                    [t.to(to_value) if t is not None else t for t in cc.value_cache],
                )
            ),
            max_cache_len=value.max_cache_len,
        )
    if value.__class__.__name__ == "EncoderDecoderCache":
        return make_encoder_decoder_cache(
            to_any(value.self_attention_cache, to_value),
            to_any(value.cross_attention_cache, to_value),
        )
    if value.__class__ in torch.utils._pytree.SUPPORTED_NODES:
        args, spec = torch.utils._pytree.tree_flatten(value)
        new_args = to_any(args, to_value)
        return torch.utils._pytree.tree_unflatten(new_args, spec)

    if hasattr(value, "to"):
        return value.to(to_value)

    assert "Cache" not in value.__class__.__name__, (
        f"Class {value.__class__.__name__!r} should be registered "
        f"to be able to change the type in every tensor it contains."
    )
    assert not isinstance(value, Iterable), f"Unsupported type {type(value)}"
    return value


def torch_deepcopy(value: Any) -> Any:
    """
    Makes a deep copy.

    :param value: any value
    :return: a deep copy
    """
    if value is None:
        return None
    if isinstance(value, (int, float, str)):
        return value
    if isinstance(value, tuple):
        return tuple(torch_deepcopy(v) for v in value)
    if isinstance(value, list):
        if type(value) is list:
            return [torch_deepcopy(v) for v in value]
    if isinstance(value, set):
        return {torch_deepcopy(v) for v in value}
    if isinstance(value, dict):
        if type(value) is dict:
            return {k: torch_deepcopy(v) for k, v in value.items()}
        # for BaseModelOutput
        return value.__class__(**{k: torch_deepcopy(v) for k, v in value.items()})
    if isinstance(value, np.ndarray):
        return value.copy()
    if hasattr(value, "clone"):
        return value.clone()
    if value.__class__.__name__ == "DynamicCache":
        from .cache_helper import CacheKeyValue

        ca = CacheKeyValue(value)
        return make_dynamic_cache(
            torch_deepcopy(list(zip(ca.key_cache, ca.value_cache))), cls_layers=ca.cls_layers
        )
    if value.__class__.__name__ == "StaticCache":
        from .cache_helper import CacheKeyValue

        ca = CacheKeyValue(value)
        if len(ca.key_cache) == 0:
            # Use of deepcopy.
            import copy

            return copy.deepcopy(value)
        return make_static_cache(
            torch_deepcopy(list(zip(ca.key_cache, ca.value_cache))),
            max_cache_len=max([value.max_cache_len, *[t.shape[2] for t in ca.key_cache]]),
        )
    if value.__class__.__name__ == "HybridCache":
        from .cache_helper import CacheKeyValue, make_hybrid_cache

        ca = CacheKeyValue(value)
        return make_hybrid_cache(torch_deepcopy(list(zip(ca.key_cache, ca.value_cache))))
    if value.__class__.__name__ == "SlidingWindowCache":
        from .cache_helper import CacheKeyValue, make_sliding_window_cache

        ca = CacheKeyValue(value)
        return make_sliding_window_cache(
            torch_deepcopy(list(zip(ca.key_cache, ca.value_cache)))
        )
    if value.__class__.__name__ == "EncoderDecoderCache":
        return make_encoder_decoder_cache(
            torch_deepcopy(value.self_attention_cache),
            torch_deepcopy(value.cross_attention_cache),
        )
    if value.__class__.__name__ == "MambaCache":
        from .cache_helper import make_mamba_cache

        return make_mamba_cache(list(zip(value.conv_states, value.ssm_states)))

    if value.__class__ in torch.utils._pytree.SUPPORTED_NODES:
        args, spec = torch.utils._pytree.tree_flatten(value)
        new_args = torch_deepcopy(args)
        return torch.utils._pytree.tree_unflatten(new_args, spec)

    if value.__class__.__name__ == "Results":
        import copy
        import ultralytics

        assert isinstance(
            value, ultralytics.engine.results.Results
        ), f"Unexpected type={type(value)}"
        return copy.deepcopy(value)

    if hasattr(value, "__nocopy__"):
        return value

    # We should have a code using serialization, deserialization assuming a model
    # cannot be exported without them.
    raise NotImplementedError(
        f"torch_deepcopy not implemented for type {type(value)}, "
        f"add attribute '__nocopy__' to return it as is."
    )


def torch_tensor_size(value: Any) -> Any:
    """Returns the number of bytes stored in tensors."""
    if value is None:
        return 0
    if isinstance(value, (int, float, str)):
        return 0
    if isinstance(value, (tuple, list, set)):
        return sum(torch_tensor_size(v) for v in value)
    if isinstance(value, dict):
        return sum(torch_tensor_size(v) for v in value.values())
    if isinstance(value, np.ndarray):
        return value.copy()
    if hasattr(value, "clone"):
        return value.numel() * size_type(value.dtype)
    if value.__class__.__name__ in {
        "DynamicCache",
        "SlidingWindowCache",
        "HybridCache",
        "StaticCache",
    }:
        cc = CacheKeyValue(value)
        return torch_tensor_size(cc.key_cache) + torch_tensor_size(cc.value_cache)
    if value.__class__.__name__ == "EncoderDecoderCache":
        return torch_tensor_size(value.self_attention_cache) + torch_tensor_size(
            value.cross_attention_cache
        )
    if value.__class__.__name__ == "MambaCache":
        return torch_tensor_size(value.conv_states) + torch_tensor_size(value.ssm_states)
    if value.__class__ in torch.utils._pytree.SUPPORTED_NODES:
        args, _spec = torch.utils._pytree.tree_flatten(value)
        return sum(torch_tensor_size(a) for a in args)

    # We should have a code using serialization, deserialization assuming a model
    # cannot be exported without them.
    raise NotImplementedError(f"torch_tensor_size not implemented for type {type(value)}")


def model_statistics(model: torch.nn.Module):
    """Returns statistics on a model in a dictionary."""
    n_subs = len(list(model.modules()))
    sizes = {}
    param_size = 0
    for param in model.parameters():
        size = param.nelement() * param.element_size()
        param_size += size
        name = str(param.dtype).replace("torch.", "")
        if name not in sizes:
            sizes[name] = 0
        sizes[name] += size

    buffer_size = 0
    for buffer in model.buffers():
        size = buffer.nelement() * buffer.element_size()
        buffer_size += size
        name = str(buffer.dtype).replace("torch.", "")
        if name not in sizes:
            sizes[name] = 0
        sizes[name] += size

    res = dict(
        type=model.__class__.__name__,
        n_modules=n_subs,
        param_size=param_size,
        buffer_size=buffer_size,
        size_mb=(param_size + buffer_size) // 2**20,
    )
    res.update(sizes)
    return res


def to_tensor(tensor: onnx.TensorProto, base_dir: str = "") -> torch.Tensor:
    """
    Converts a TensorProto to a numpy array.

    :param tensor: a TensorProto object.
    :param base_dir: if external tensor exists, base_dir can help to find the path to it
    :return: the converted tensor
    """
    assert not tensor.HasField("segment"), "Currently not supporting loading segments."
    assert (
        tensor.data_type != onnx.TensorProto.UNDEFINED
    ), "The element type in the input tensor is not defined."
    assert tensor.data_type != onnx.TensorProto.STRING, "to_tensor not implemented for strings"

    tensor_dtype = tensor.data_type
    torch_dtype = onnx_dtype_to_torch_dtype(tensor_dtype)
    dims = tuple(tensor.dims)
    if uses_external_data(tensor):
        # Load raw data from external tensor if it exists
        load_external_data_for_tensor(tensor, base_dir)

    if tensor.HasField("raw_data"):
        raw_data = tensor.raw_data
        if len(raw_data) == 0:
            return torch.tensor([], dtype=torch_dtype).reshape(dims)
        if sys.byteorder == "big":
            # Convert endian from little to big
            raw_data = torch.frombuffer(raw_data, dtype=torch_dtype).byteswap().tobytes()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return torch.frombuffer(raw_data, dtype=torch_dtype).reshape(dims)

    # Other cases, it should be small tensor. We use numpy.
    np_tensor = to_array_extended(tensor)
    return torch.from_numpy(np_tensor)


def get_weight_type(model: torch.nn.Module) -> torch.dtype:
    """Returns the most probable dtype in a model."""
    counts = {}
    for _name, param in model.named_parameters():
        dt = param.dtype
        if dt not in counts:
            counts[dt] = 1
        else:
            counts[dt] += 1
    final = max(list(counts.items()))
    return final[0]


def closest_factor_pair(n: int):
    """Tries to find ``a, b`` such as ``n == a * b``."""
    assert n > 0, f"n={n} must be a positive integer"
    start = math.isqrt(n)
    for a in range(start, 0, -1):
        if n % a == 0:
            b = n // a
            return a, b
    return 1, n


def study_discrepancies(
    t1: torch.Tensor,
    t2: torch.Tensor,
    bins: int = 50,
    figsize: Optional[Tuple[int, int]] = (15, 15),
    title: Optional[str] = None,
    name: Optional[str] = None,
) -> "matplotlib.axes.Axes":  # noqa: F821
    """
    Computes different metrics for the discrepancies.
    Returns graphs.

    .. plot::
        :include-source:

        import torch
        from onnx_diagnostic.helpers.torch_helper import study_discrepancies

        t1 = torch.randn((512, 1024)) * 10
        t2 = t1 + torch.randn((512, 1024))
        study_discrepancies(t1, t2, title="Random noise")
    """
    assert t1.dtype == t2.dtype, f"Type mismatch {t1.dtype} != {t2.dtype}"
    assert t1.shape == t2.shape, f"Shape mismatch {t1.shape} != {t2.shape}"
    d1, d2 = (
        (t1, t2) if t1.dtype == torch.float64 else (t1.to(torch.float32), t2.to(torch.float32))
    )

    d1 = d1.squeeze()
    d2 = d2.squeeze()
    if len(d1.shape) == 1:
        new_shape = closest_factor_pair(d1.shape[0])
        d1, d2 = d1.reshape(new_shape), d2.reshape(new_shape)
    elif len(d1.shape) > 2:
        new_shape = (-1, max(d1.shape))
        d1, d2 = d1.reshape(new_shape), d2.reshape(new_shape)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(3, 2, figsize=figsize)
    vmin, vmax = d1.min().item(), d1.max().item()
    ax[0, 0].imshow(d1.detach().cpu().numpy(), cmap="Greys", vmin=vmin, vmax=vmax)
    ax[0, 0].set_title(
        f"Color plot of the first tensor in\n[{vmin}, {vmax}]\n{t1.shape} -> {d1.shape}"
    )

    diff = d2 - d1
    vmin, vmax = diff.min().item(), diff.max().item()
    ax[0, 1].imshow(diff.detach().cpu().numpy(), cmap="seismic", vmin=vmin, vmax=vmax)
    ax[0, 1].set_title(f"Color plot of the differences in \n[{vmin}, {vmax}]")

    ax[1, 0].hist(d1.detach().cpu().numpy().ravel(), bins=bins)
    ax[1, 0].set_title("Distribution of the first tensor")

    ax[1, 1].hist(diff.detach().cpu().numpy().ravel(), bins=bins)
    ax[1, 1].set_title("Distribution of the differences")

    tf1 = d1.ravel()
    td1 = diff.ravel()
    ax[2, 1].plot(tf1.detach().cpu().numpy(), td1.detach().cpu().numpy(), ".")
    ax[2, 1].set_title("Graph XY")
    ax[2, 1].set_xlabel("First tensor values")
    ax[2, 1].set_ylabel("Difference values")

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if name:
        fig.savefig(name)
    return ax


def int_device_to_torch_device(device_id: int) -> torch.device:
    """
    Converts a device defined as an integer (coming from :meth:`torch.Tensor.get_device`)
    into a ``torch.device``.
    """
    if device_id < 0:
        return torch.device("cpu")
    return torch.device("cuda", device_id)
