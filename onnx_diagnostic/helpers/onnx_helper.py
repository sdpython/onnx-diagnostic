import ctypes
import functools
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import numpy as np
import numpy.typing as npt
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh
from onnx import (
    AttributeProto,
    FunctionProto,
    GraphProto,
    ModelProto,
    NodeProto,
    TensorProto,
    ValueInfoProto,
    load as onnx_load,
)


def _make_stat(init: TensorProto) -> Dict[str, float]:
    """
    Produces statistics.

    :param init: tensor
    :return statistics
    """
    ar = onh.to_array(init)
    return dict(
        mean=float(ar.mean()),
        std=float(ar.std()),
        shape=ar.shape,
        itype=np_dtype_to_tensor_dtype(ar.dtype),
        min=float(ar.min()),
        max=float(ar.max()),
    )


def onnx_lighten(
    onx: Union[str, ModelProto],
    verbose: int = 0,
) -> Tuple[ModelProto, Dict[str, Dict[str, float]]]:
    """
    Creates a model without big initializers but stores statistics
    into dictionaries. The function can be reversed with
    :func:`onnx_diagnostic.helpers.onnx_helper.onnx_unlighten`.
    The model is modified inplace.

    :param onx: model
    :param verbose: verbosity
    :return: new model, statistics
    """
    if isinstance(onx, str):
        if verbose:
            print(f"[onnx_lighten] load {onx!r}")
        model = onnx.load(onx)
    else:
        assert isinstance(onx, ModelProto), f"Unexpected type {type(onx)}"
        model = onx

    keep = []
    stats = []
    for init in model.graph.initializer:
        shape = init.dims
        size = np.prod(shape)
        if size > 2**12:
            stat = _make_stat(init)
            stats.append((init.name, stat))
            if verbose:
                print(f"[onnx_lighten] remove initializer {init.name!r} stat={stat}")
        else:
            keep.append(init)

    del model.graph.initializer[:]
    model.graph.initializer.extend(keep)
    return model, dict(stats)


def _get_tensor(min=None, max=None, mean=None, std=None, shape=None, itype=None):
    assert itype is not None, "itype must be specified."
    assert shape is not None, "shape must be specified."
    dtype = tensor_dtype_to_np_dtype(itype)
    if (mean is None or std is None) or (
        min is not None and max is not None and abs(max - min - 1) < 0.01
    ):
        if min is None:
            min = 0
        if max is None:
            max = 0
        return (np.random.random(shape) * (max - min) + min).astype(dtype)
    assert std is not None and mean is not None, f"mean={mean} or std={std} is None"
    t = np.random.randn(*shape).astype(dtype)
    return t


def onnx_unlighten(
    onx: Union[str, ModelProto],
    stats: Optional[Dict[str, Dict[str, float]]] = None,
    verbose: int = 0,
) -> ModelProto:
    """
    Function fixing the model produced by function
    :func:`onnx_diagnostic.helpers.onnx_helper.onnx_lighten`.
    The model is modified inplace.

    :param onx: model
    :param stats: statistics, can be None if onx is a file,
        then it loads the file ``<filename>.stats``,
        it assumes it is json format
    :param verbose: verbosity
    :return: new model, statistics
    """
    if isinstance(onx, str):
        if stats is None:
            fstats = f"{onx}.stats"
            assert os.path.exists(fstats), f"File {fstats!r} is missing."
            if verbose:
                print(f"[onnx_unlighten] load {fstats!r}")
            with open(fstats, "r") as f:
                stats = json.load(f)
        if verbose:
            print(f"[onnx_unlighten] load {onx!r}")
        model = onnx.load(onx)
    else:
        assert isinstance(onx, ModelProto), f"Unexpected type {type(onx)}"
        model = onx
        assert stats is not None, "stats is missing"

    keep = []
    for name, stat in stats.items():
        t = _get_tensor(**stat)
        init = from_array_extended(t, name=name)
        keep.append(init)

    model.graph.initializer.extend(keep)
    return model


def _validate_graph(
    g: GraphProto,
    existing: Set[str],
    verbose: int = 0,
    watch: Optional[Set[str]] = None,
    path: Optional[Sequence[str]] = None,
):
    found = []
    path = path or ["root"]
    set_init = set(i.name for i in g.initializer)
    set_input = set(i.name for i in g.input)
    existing |= set_init | set_input
    if watch and set_init & watch:
        if verbose:
            print(f"-- found init {set_init & watch} in {path}")
        found.extend([i for i in g.initializer if i.name in set_init & watch])
    if watch and set_input & watch:
        if verbose:
            print(f"-- found input {set_input & watch} in {path}")
        found.extend([i for i in g.input if i.name in set_input & watch])
    try:
        import tqdm

        loop = tqdm.tqdm(g.node) if verbose else g.node
    except ImportError:
        loop = g.node

    for node in loop:
        ins = set(node.input) & existing
        if ins != set(node.input):
            raise AssertionError(
                f"One input is missing from node.input={node.input}, "
                f"existing={ins}, path={'/'.join(path)}, "
                f"node: {node.op_type}[{node.name}]"
            )
        if watch and ins & watch:
            if verbose:
                print(
                    f"-- found input {ins & watch} in "
                    f"{'/'.join(path)}/{node.op_type}[{node.name}]"
                )
            found.append(node)
        for att in node.attribute:
            if att.type == AttributeProto.GRAPH:
                found.extend(
                    _validate_graph(
                        att.g,
                        existing.copy(),
                        watch=watch,
                        path=[*path, f"{node.op_type}[{node.name}]"],
                        verbose=verbose,
                    )
                )
        existing |= set(node.output)
        if watch and set(node.output) & watch:
            if verbose:
                print(
                    f"-- found output {set(node.output) & watch} "
                    f"in {'/'.join(path)}/{node.op_type}[{node.name}]"
                )
            found.append(node)
    out = set(o.name for o in g.output)
    ins = out & existing
    if ins != out:
        raise AssertionError(
            f"One output is missing, out={node.input}, existing={ins}, path={path}"
        )
    return found


def _validate_function(g: FunctionProto, verbose: int = 0, watch: Optional[Set[str]] = None):
    existing = set(g.input)
    found = []
    for node in g.node:
        ins = set(node.input) & existing
        if ins != set(node.input):
            raise AssertionError(
                f"One input is missing from node.input={node.input}, existing={ins}"
            )
        if watch and ins & watch:
            if verbose:
                print(f"-- found input {ins & watch} in {node.op_type}[{node.name}]")
            found.append(node)
        for att in node.attribute:
            if att.type == AttributeProto.GRAPH:
                found.extend(
                    _validate_graph(g, existing.copy(), path=[g.name], verbose=verbose)
                )
        existing |= set(node.output)
        if watch and set(node.output) & watch:
            if verbose:
                print(
                    f"-- found output {set(node.output) & watch} "
                    f"in {node.op_type}[{node.name}]"
                )
    out = set(g.output)
    ins = out & existing
    if ins != out:
        raise AssertionError(
            f"One output is missing, out={node.input}, existing={ins}, path={g.name}"
        )
    return found


def onnx_find(
    onx: Union[str, ModelProto], verbose: int = 0, watch: Optional[Set[str]] = None
) -> List[Union[NodeProto, TensorProto]]:
    """
    Looks for node producing or consuming some results.

    :param onx: model
    :param verbose: verbosity
    :param watch: names to search for
    :return: list of nodes
    """

    if isinstance(onx, str):
        onx = onnx.load(onx, load_external_data=False)
    found = []
    found.extend(_validate_graph(onx.graph, set(), verbose=verbose, watch=watch))
    for f in onx.functions:
        found.extend(_validate_function(f, watch=watch, verbose=verbose))
    if verbose and found:
        print(f"-- found {len(found)} nodes")
    return found


def check_model_ort(
    onx: ModelProto,
    providers: Optional[Union[str, List[Any]]] = None,
    dump_file: Optional[str] = None,
) -> "onnxruntime.InferenceSession":  # noqa: F821
    """
    Loads a model with onnxruntime.

    :param onx: ModelProto
    :param providers: list of providers, None fur CPU, cpu for CPU, cuda for CUDA
    :param dump_file: if not empty, dumps the model into this file if
        an error happened
    :return: InferenceSession
    """
    from onnxruntime import InferenceSession

    if providers is None or providers == "cpu":
        providers = ["CPUExecutionProvider"]
    elif not isinstance(providers, list) and providers.startswith("cuda"):
        device_id = 0 if ":" not in providers else int(providers.split(":")[1])
        providers = [
            ("CUDAExecutionProvider", {"device_id": device_id}),
            ("CPUExecutionProvider", {}),
        ]

    if isinstance(onx, str):
        try:
            return InferenceSession(onx, providers=providers)
        except Exception as e:
            import onnx

            if dump_file:
                onnx.save(onx, dump_file)

            raise AssertionError(  # noqa: B904
                f"onnxruntime cannot load the model "
                f"due to {e}\n{pretty_onnx(onnx.load(onx))}"
            )
        return
    try:
        return InferenceSession(onx.SerializeToString(), providers=providers)
    except Exception as e:
        if dump_file:
            onnx.save(onx, dump_file)
        raise AssertionError(  # noqa: B904
            f"onnxruntime cannot load the modeldue to {e}\n{pretty_onnx(onx)}"
        )


@functools.cache
def onnx_dtype_name(itype: int) -> str:
    """
    Returns the ONNX name for a specific element type.

    .. runpython::
        :showcode:

        import onnx
        from onnx_diagnostic.helpers.onnx_helper import onnx_dtype_name

        itype = onnx.TensorProto.BFLOAT16
        print(onnx_dtype_name(itype))
        print(onnx_dtype_name(7))
    """
    for k in dir(TensorProto):
        v = getattr(TensorProto, k)
        if v == itype:
            return k
    raise ValueError(f"Unexpected value itype: {itype}")


def pretty_onnx(
    onx: Union[FunctionProto, GraphProto, ModelProto, ValueInfoProto, str],
    with_attributes: bool = False,
    highlight: Optional[Set[str]] = None,
    shape_inference: bool = False,
) -> str:
    """
    Displays an onnx prot in a better way.

    :param with_attributes: displays attributes as well, if only a node is printed
    :param highlight: to highlight some names
    :param shape_inference: run shape inference before printing the model
    :return: text
    """
    assert onx is not None, "onx cannot be None"
    if isinstance(onx, str):
        onx = onnx_load(onx, load_external_data=False)
    assert onx is not None, "onx cannot be None"

    if shape_inference:
        onx = onnx.shape_inference.infer_shapes(onx)

    if isinstance(onx, ValueInfoProto):
        name = onx.name
        itype = onx.type.tensor_type.elem_type
        shape = tuple((d.dim_param or d.dim_value) for d in onx.type.tensor_type.shape.dim)
        shape_str = ",".join(map(str, shape))
        return f"{onnx_dtype_name(itype)}[{shape_str}] {name}"

    if isinstance(onx, AttributeProto):
        att = onx
        if att.type == AttributeProto.INT:
            return f"{att.name}={att.i}"
        if att.type == AttributeProto.INTS:
            return f"{att.name}={att.ints}"
        if att.type == AttributeProto.FLOAT:
            return f"{att.name}={att.f}"
        if att.type == AttributeProto.FLOATS:
            return f"{att.name}={att.floats}"
        if att.type == AttributeProto.STRING:
            return f"{att.name}={att.s!r}"
        if att.type == AttributeProto.TENSOR:
            v = to_array_extended(att.t)
            assert hasattr(v, "reshape"), f"not a tensor {type(v)}"
            assert hasattr(v, "shape"), f"not a tensor {type(v)}"
            vf = v.reshape((-1,))
            if vf.size < 10:
                tt = f"[{', '.join(map(str, vf))}]"
            else:
                tt = f"[{', '.join(map(str, vf[:10]))}, ...]"
            if len(v.shape) != 1:
                return f"{att.name}=tensor({tt}, dtype={v.dtype}).reshape({v.shape})"
            return f"{att.name}=tensor({tt}, dtype={v.dtype})"
        raise NotImplementedError(
            f"pretty_onnx not implemented yet for AttributeProto={att!r}"
        )

    if isinstance(onx, NodeProto):

        def _high(n):
            if highlight and n in highlight:
                return f"**{n}**"
            return n

        text = (
            f"{onx.op_type}({', '.join(map(_high, onx.input))})"
            f" -> {', '.join(map(_high, onx.output))}"
        )
        if onx.domain:
            text = f"{onx.domain}.{text}"
        if not with_attributes or not onx.attribute:
            return text
        rows = []
        for att in onx.attribute:
            rows.append(pretty_onnx(att))
        if len(rows) > 1:
            suffix = "\n".join(f"    {s}" for s in rows)
            return f"{text}\n{suffix}"
        return f"{text}  ---  {rows[0]}"

    if isinstance(onx, TensorProto):
        shape = "x".join(map(str, onx.dims))
        return f"TensorProto:{onx.data_type}:{shape}:{onx.name}"

    try:
        from onnx_array_api.plotting.text_plot import onnx_simple_text_plot

        if isinstance(onx, FunctionProto):
            return (
                f"function: {onx.name}[{onx.domain}]\n"
                f"{onnx_simple_text_plot(onx, recursive=True)}"
            )
        return onnx_simple_text_plot(onx, recursive=True)
    except ImportError:
        from onnx.printer import to_text

        return to_text(onx)


def get_onnx_signature(model: ModelProto) -> Tuple[Tuple[str, Any], ...]:
    """
    Produces a tuple of tuples corresponding to the signatures.

    :param model: model
    :return: signature
    """
    sig: List[Any] = []
    for i in model.graph.input:
        dt = i.type
        if dt.HasField("sequence_type"):
            dst = dt.sequence_type.elem_type
            tdt = dst.tensor_type
            el = tdt.elem_type
            shape = tuple(d.dim_param or d.dim_value for d in tdt.shape.dim)
            sig.append((i.name, [(i.name, el, shape)]))
        elif dt.HasField("tensor_type"):
            el = dt.tensor_type.elem_type
            shape = tuple(d.dim_param or d.dim_value for d in dt.tensor_type.shape.dim)
            sig.append((i.name, el, shape))
        else:
            raise AssertionError(f"Unable to interpret dt={dt!r} in {i!r}")
    return tuple(sig)


def convert_endian(tensor: TensorProto) -> None:
    """Call to convert endianness of raw data in tensor.

    Args:
        tensor: TensorProto to be converted.
    """
    tensor_dtype = tensor.data_type
    np_dtype = tensor_dtype_to_np_dtype(tensor_dtype)
    tensor.raw_data = np.frombuffer(tensor.raw_data, dtype=np_dtype).byteswap().tobytes()


def from_array_ml_dtypes(arr: npt.ArrayLike, name: Optional[str] = None) -> TensorProto:
    """
    Converts a numpy array to a tensor def assuming the dtype
    is defined in ml_dtypes.

    Args:
        arr: a numpy array.
        name: (optional) the name of the tensor.

    Returns:
        TensorProto: the converted tensor def.
    """
    import ml_dtypes

    assert isinstance(arr, np.ndarray), f"arr must be of type numpy.ndarray, got {type(arr)}"

    tensor = TensorProto()
    tensor.dims.extend(arr.shape)
    if name:
        tensor.name = name

    if arr.dtype == ml_dtypes.bfloat16:
        dtype = TensorProto.BFLOAT16
    elif arr.dtype == ml_dtypes.float8_e4m3fn:
        dtype = TensorProto.FLOAT8E4M3FN
    elif arr.dtype == ml_dtypes.float8_e4m3fnuz:
        dtype = TensorProto.FLOAT8E4M3FNUZ
    elif arr.dtype == ml_dtypes.float8_e5m2:
        dtype = TensorProto.FLOAT8E5M2
    elif arr.dtype == ml_dtypes.float8_e5m2fnuz:
        dtype = TensorProto.FLOAT8E5M2FNUZ
    else:
        raise NotImplementedError(f"No conversion from {arr.dtype}")
    tensor.data_type = dtype
    tensor.raw_data = arr.tobytes()  # note: tobytes() is only after 1.9.
    if sys.byteorder == "big":
        convert_endian(tensor)
    return tensor


_STORAGE_TYPE = {
    TensorProto.FLOAT16: np.int16,
    TensorProto.BFLOAT16: np.int16,
}


def proto_from_tensor(
    arr: "torch.Tensor",  # noqa: F821
    name: Optional[str] = None,
    verbose: int = 0,
) -> TensorProto:
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

    tensor = TensorProto()
    tensor.dims.extend(arr_cpu.shape)
    if name:
        tensor.name = name
    itype = torch_dtype_to_onnx_dtype(arr_cpu.dtype)
    assert not hasattr(TensorProto, "INT4") or itype not in {
        TensorProto.INT4,
        TensorProto.UINT4,
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


def from_array_extended(tensor: npt.ArrayLike, name: Optional[str] = None) -> TensorProto:
    """
    Converts an array into a :class:`onnx.TensorProto`.

    :param tensor: numpy array or torch tensor
    :param name: name
    :return: TensorProto
    """
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and isinstance(tensor, torch.Tensor):
        return proto_from_tensor(tensor, name=name)

    from onnx.reference.ops.op_cast import (
        bfloat16,
        float8e4m3fn,
        float8e4m3fnuz,
        float8e5m2,
        float8e5m2fnuz,
    )

    dt = tensor.dtype
    if dt == float8e4m3fn and dt.descr[0][0] == "e4m3fn":
        to = TensorProto.FLOAT8E4M3FN
        dt_to = np.uint8
    elif dt == float8e4m3fnuz and dt.descr[0][0] == "e4m3fnuz":
        to = TensorProto.FLOAT8E4M3FNUZ
        dt_to = np.uint8
    elif dt == float8e5m2 and dt.descr[0][0] == "e5m2":
        to = TensorProto.FLOAT8E5M2
        dt_to = np.uint8
    elif dt == float8e5m2fnuz and dt.descr[0][0] == "e5m2fnuz":
        to = TensorProto.FLOAT8E5M2FNUZ
        dt_to = np.uint8
    elif dt == bfloat16 and dt.descr[0][0] == "bfloat16":
        to = TensorProto.BFLOAT16
        dt_to = np.uint16
    else:
        try:
            import ml_dtypes
        except ImportError:
            ml_dtypes = None
        if ml_dtypes is not None and (
            tensor.dtype == ml_dtypes.bfloat16
            or tensor.dtype == ml_dtypes.float8_e4m3fn
            or tensor.dtype == ml_dtypes.float8_e4m3fnuz
            or tensor.dtype == ml_dtypes.float8_e5m2
            or tensor.dtype == ml_dtypes.float8_e5m2fnuz
        ):
            return from_array_ml_dtypes(tensor, name)
        return onh.from_array(tensor, name)

    t = onh.from_array(tensor.astype(dt_to), name)
    t.data_type = to
    return t


def to_array_extended(proto: TensorProto) -> npt.ArrayLike:
    """Converts :class:`onnx.TensorProto` into a numpy array."""
    arr = onh.to_array(proto)
    if proto.data_type >= onnx.TensorProto.BFLOAT16:
        # Types not supported by numpy
        ml_dtypes = onnx_dtype_to_np_dtype(proto.data_type)
        return arr.view(ml_dtypes)
    return arr


def onnx_dtype_to_torch_dtype(itype: int) -> "torch.dtype":  # noqa: F821
    """
    Converts an onnx type into a torch dtype.

    :param to: onnx dtype
    :return: torch dtype
    """
    import torch

    if itype == TensorProto.FLOAT:
        return torch.float32
    if itype == TensorProto.FLOAT16:
        return torch.float16
    if itype == TensorProto.BFLOAT16:
        return torch.bfloat16
    if itype == TensorProto.DOUBLE:
        return torch.float64
    if itype == TensorProto.INT32:
        return torch.int32
    if itype == TensorProto.INT64:
        return torch.int64
    if itype == TensorProto.UINT32:
        return torch.uint32
    if itype == TensorProto.UINT64:
        return torch.uint64
    if itype == TensorProto.BOOL:
        return torch.bool
    if itype == TensorProto.INT16:
        return torch.int16
    if itype == TensorProto.UINT16:
        return torch.uint16
    if itype == TensorProto.INT8:
        return torch.int16
    if itype == TensorProto.UINT8:
        return torch.uint16
    if itype == TensorProto.COMPLEX64:
        return torch.complex64
    if itype == TensorProto.COMPLEX128:
        return torch.complex128
    raise NotImplementedError(
        f"Unable to convert onnx type {onnx_dtype_name(itype)} to torch.type."
    )


def onnx_dtype_to_np_dtype(itype: int) -> Any:
    """
    Converts an onnx type into a to numpy dtype.
    That includes :epkg:`ml_dtypes` dtypes.

    :param to: onnx dtype
    :return: numpy dtype
    """
    if itype == TensorProto.FLOAT:
        return np.float32
    if itype == TensorProto.FLOAT16:
        return np.float16
    if itype == TensorProto.BFLOAT16:
        import ml_dtypes

        return ml_dtypes.bfloat16
    if itype == TensorProto.DOUBLE:
        return np.float64
    if itype == TensorProto.INT32:
        return np.int32
    if itype == TensorProto.INT64:
        return np.int64
    if itype == TensorProto.UINT32:
        return np.uint32
    if itype == TensorProto.UINT64:
        return np.uint64
    if itype == TensorProto.BOOL:
        return np.bool
    if itype == TensorProto.INT16:
        return np.int16
    if itype == TensorProto.UINT16:
        return np.uint16
    if itype == TensorProto.INT8:
        return np.int16
    if itype == TensorProto.UINT8:
        return np.uint16
    if itype == TensorProto.COMPLEX64:
        return np.complex64
    if itype == TensorProto.COMPLEX128:
        return np.complex128
    raise NotImplementedError(
        f"Unable to convert onnx type {onnx_dtype_name(itype)} to torch.type."
    )


def torch_dtype_to_onnx_dtype(to: "torch.dtype") -> int:  # noqa: F821
    """
    Converts a torch dtype into a onnx element type.

    :param to: torch dtype
    :return: onnx type
    """
    import torch

    if to == torch.float32:
        return TensorProto.FLOAT
    if to == torch.float16:
        return TensorProto.FLOAT16
    if to == torch.bfloat16:
        return TensorProto.BFLOAT16
    if to == torch.float64:
        return TensorProto.DOUBLE
    if to == torch.int64:
        return TensorProto.INT64
    if to == torch.int32:
        return TensorProto.INT32
    if to == torch.uint64:
        return TensorProto.UINT64
    if to == torch.uint32:
        return TensorProto.UINT32
    if to == torch.bool:
        return TensorProto.BOOL
    if to == torch.SymInt:
        return TensorProto.INT64
    if to == torch.int16:
        return TensorProto.INT16
    if to == torch.uint16:
        return TensorProto.UINT16
    if to == torch.int8:
        return TensorProto.INT8
    if to == torch.uint8:
        return TensorProto.UINT8
    if to == torch.SymFloat:
        return TensorProto.FLOAT
    if to == torch.complex64:
        return TensorProto.COMPLEX64
    if to == torch.complex128:
        return TensorProto.COMPLEX128
    raise NotImplementedError(f"Unable to convert torch dtype {to!r} to onnx dtype.")


def dtype_to_tensor_dtype(dt: Union[np.dtype, "torch.dtype"]) -> int:  # noqa: F821
    """
    Converts a torch dtype or numpy dtype into a onnx element type.

    :param to: dtype
    :return: onnx type
    """
    try:
        return np_dtype_to_tensor_dtype(dt)
    except (KeyError, TypeError, ValueError):
        pass
    return torch_dtype_to_onnx_dtype(dt)


def np_dtype_to_tensor_dtype(dt: np.dtype) -> int:  # noqa: F821
    """
    Converts a numpy dtype into a onnx element type.

    :param to: dtype
    :return: onnx type
    """
    try:
        return oh.np_dtype_to_tensor_dtype(dt)
    except ValueError:
        try:
            import ml_dtypes
        except ImportError:
            ml_dtypes = None  # type: ignore
        if ml_dtypes is not None:
            if dt == ml_dtypes.bfloat16:
                return TensorProto.BFLOAT16
            if dt == ml_dtypes.float8_e4m3fn:
                return TensorProto.FLOAT8E4M3FN
            if dt == ml_dtypes.float8_e4m3fnuz:
                return TensorProto.FLOAT8E4M3FNUZ
            if dt == ml_dtypes.float8_e5m2:
                return TensorProto.FLOAT8E5M2
            if dt == ml_dtypes.float8_e5m2fnuz:
                return TensorProto.FLOAT8E5M2FNUZ
    if dt == np.float32:
        return TensorProto.FLOAT
    if dt == np.float16:
        return TensorProto.FLOAT16
    if dt == np.float64:
        return TensorProto.DOUBLE
    if dt == np.int64:
        return TensorProto.INT64
    if dt == np.uint64:
        return TensorProto.UINT64
    if dt == np.int16:
        return TensorProto.INT16
    if dt == np.uint16:
        return TensorProto.UINT16
    if dt == np.int32:
        return TensorProto.INT32
    if dt == np.int8:
        return TensorProto.INT8
    if dt == np.uint8:
        return TensorProto.UINT8
    if dt == np.uint32:
        return TensorProto.UINT32
    if dt == np.bool:
        return TensorProto.BOOL
    if dt == np.complex64:
        return TensorProto.COMPLEX64
    if dt == np.complex128:
        return TensorProto.COMPLEX128
    raise ValueError(f"Unable to convert type {dt}")


def type_info(itype: int, att: str):
    """
    Returns the minimum or maximum value for a type.

    :param itype: onnx type
    :param att: 'min' or 'max'
    :return: value
    """
    if itype in {TensorProto.FLOAT, TensorProto.FLOAT16, TensorProto.DOUBLE}:
        dtype = tensor_dtype_to_np_dtype(itype)
        fi = np.finfo(dtype)
    elif itype == TensorProto.BFLOAT16:
        import ml_dtypes

        dtype = tensor_dtype_to_np_dtype(itype)
        fi = ml_dtypes.finfo(dtype)  # type: ignore
    else:
        dtype = tensor_dtype_to_np_dtype(itype)
        fi = np.iinfo(dtype)  # type: ignore
    if att == "min":
        return fi.min
    if att == "max":
        return fi.max
    raise ValueError(f"Unexpected value {att!r}")


def tensor_dtype_to_np_dtype(tensor_dtype: int) -> np.dtype:
    """
    Converts a TensorProto's data_type to corresponding numpy dtype.
    It can be used while making tensor.

    :param tensor_dtype: TensorProto's data_type
    :return: numpy's data_type
    """
    if tensor_dtype >= 16:
        try:
            import ml_dtypes  # noqa: F401
        except ImportError as e:
            raise ValueError(
                f"Unsupported value for tensor_dtype, "
                f"numpy does not support onnx type {tensor_dtype}. "
                f"ml_dtypes can be used."
            ) from e

        mapping: Dict[int, np.dtype] = {
            TensorProto.BFLOAT16: ml_dtypes.bfloat16,
            TensorProto.FLOAT8E4M3FN: ml_dtypes.float8_e4m3fn,
            TensorProto.FLOAT8E4M3FNUZ: ml_dtypes.float8_e4m3fnuz,
            TensorProto.FLOAT8E5M2: ml_dtypes.float8_e5m2,
            TensorProto.FLOAT8E5M2FNUZ: ml_dtypes.float8_e5m2fnuz,
        }
        assert (
            tensor_dtype in mapping
        ), f"Unable to find tensor_dtype={tensor_dtype!r} in mapping={mapping}"
        return mapping[tensor_dtype]

    return oh.tensor_dtype_to_np_dtype(tensor_dtype)
