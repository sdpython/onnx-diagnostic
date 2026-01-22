import functools
import json
import os
import sys
import warnings
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)
import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh
from onnx import (
    AttributeProto,
    FunctionProto,
    GraphProto,
    ModelProto,
    NodeProto,
    OperatorSetIdProto,
    TensorProto,
    TypeProto,
    ValueInfoProto,
    load as onnx_load,
)
from ..typing import InferenceSessionLike, TensorLike


def _make_stat(init: TensorProto) -> Dict[str, Any]:
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
) -> List[Union[NodeProto, TensorProto, ValueInfoProto]]:
    found: List[Union[NodeProto, TensorProto, ValueInfoProto]] = []
    path = path or ["root"]
    set_init = {i.name for i in g.initializer}
    set_input = {i.name for i in g.input}
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
    out = {o.name for o in g.output}
    ins = out & existing
    assert ins == out, f"One output is missing, out={node.input}, existing={ins}, path={path}"
    return found


def _validate_function(g: FunctionProto, verbose: int = 0, watch: Optional[Set[str]] = None):
    existing: Set[str] = set(g.input)
    found: List[Union[NodeProto, TensorProto, ValueInfoProto]] = []
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
                    _validate_graph(att.g, existing.copy(), path=[g.name], verbose=verbose)
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
) -> InferenceSessionLike:
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
            # pyrefly: ignore[bad-return]
            return InferenceSession(onx, providers=providers)
        except Exception as e:
            if dump_file:
                onnx.save(onx, dump_file)

            raise AssertionError(  # noqa: B904
                f"onnxruntime cannot load the model "
                f"due to {e}\n{pretty_onnx(onnx.load(onx))}"
            )
    try:
        # pyrefly: ignore[bad-return]
        return InferenceSession(onx.SerializeToString(), providers=providers)
    except Exception as e:
        if dump_file:
            onnx.save(onx, dump_file)
        raise AssertionError(  # noqa: B904
            f"onnxruntime cannot load the modeldue to {e}\n{pretty_onnx(onx)}"
        )


@functools.cache
def onnx_dtype_name(itype: int, exc: bool = True) -> str:
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
        if k.upper() == k and k not in {"DESCRIPTOR", "EXTERNAL", "DEFAULT"}:
            v = getattr(TensorProto, k)
            if v == itype:
                return k
    if exc:
        raise ValueError(f"Unexpected value itype: {itype}")
    if itype == 0:
        return "UNDEFINED"
    return "UNEXPECTED"


def pretty_onnx(
    onx: Union[
        AttributeProto,
        FunctionProto,
        GraphProto,
        ModelProto,
        NodeProto,
        onnx.SparseTensorProto,
        TensorProto,
        ValueInfoProto,
        str,
    ],
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
        assert isinstance(
            onx, ModelProto
        ), f"shape inference only works for ModelProto, not {type(onx)}"
        onx = onnx.shape_inference.infer_shapes(onx)

    if isinstance(onx, ValueInfoProto):
        name = onx.name
        itype = onx.type.tensor_type.elem_type
        shape = tuple((d.dim_param or d.dim_value) for d in onx.type.tensor_type.shape.dim)
        shape_str = ",".join(map(str, shape))
        return f"{onnx_dtype_name(itype, exc=False)}[{shape_str}] {name}"

    if isinstance(onx, TypeProto):
        itype = onx.tensor_type.elem_type
        shape = tuple((d.dim_param or d.dim_value) for d in onx.tensor_type.shape.dim)
        shape_str = ",".join(map(str, shape))
        return f"{onnx_dtype_name(itype, exc=False)}[{shape_str}]"

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

    assert not isinstance(onx, onnx.SparseTensorProto), "SparseTensorProto is not handled yet."

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


def from_array_ml_dtypes(arr: TensorLike, name: Optional[str] = None) -> TensorProto:
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


def from_array_extended(tensor: TensorLike, name: Optional[str] = None) -> TensorProto:
    """
    Converts an array into a :class:`onnx.TensorProto`.

    :param tensor: numpy array or torch tensor
    :param name: name
    :return: TensorProto
    """
    if not isinstance(tensor, np.ndarray):
        import torch
        from .torch_helper import proto_from_tensor

        assert isinstance(
            tensor, torch.Tensor
        ), f"Unable to convert type {type(tensor)} into TensorProto."
        return proto_from_tensor(tensor, name=name)

    assert isinstance(tensor, np.ndarray)  # type checking
    # pyrefly: ignore[bad-argument-type]
    return onh.from_array(tensor, name)


def to_array_extended(proto: TensorProto) -> TensorLike:
    """Converts :class:`onnx.TensorProto` into a numpy array."""
    arr = onh.to_array(proto)
    if proto.data_type >= onnx.TensorProto.BFLOAT16:
        # Types not supported by numpy
        ml_dtypes = onnx_dtype_to_np_dtype(proto.data_type)
        return arr.view(ml_dtypes)
    return arr


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


# pyrefly: ignore[unknown-name]
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
    from .torch_helper import torch_dtype_to_onnx_dtype

    # pyrefly: ignore[bad-argument-type]
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
        import ml_dtypes

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

        # pyrefly: ignore[bad-assignment]
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


def iterator_initializer_constant(
    model: Union[FunctionProto, GraphProto, ModelProto],
    use_numpy: bool = True,
    prefix: str = "",
) -> Iterator[Tuple[str, TensorLike]]:  # noqa: F821
    """
    Iterates on iniatialiers and constant in an onnx model.

    :param model: model
    :param use_numpy: use numpy or pytorch
    :param prefix: for subgraph
    :return: iterator
    """
    if not isinstance(model, FunctionProto):
        graph = model if isinstance(model, GraphProto) else model.graph
        if not use_numpy:
            from .torch_helper import to_tensor
        if prefix:
            prefix += "."
        for init in graph.initializer:
            s = f"{prefix}{init.name}"
            if use_numpy:
                yield s, to_array_extended(init)
            else:
                # pyrefly: ignore[unbound-name]
                yield s, to_tensor(init)
        nodes = graph.node
        name = graph.name
        if isinstance(model, ModelProto):
            for f in model.functions:
                yield from iterator_initializer_constant(
                    f, use_numpy=use_numpy, prefix=f"{prefix}{f.name}"
                )
    else:
        nodes = model.node
        name = model.name
    for node in nodes:
        if node.op_type == "Constant" and node.domain == "":
            from ..reference import ExtendedReferenceEvaluator as Inference

            sess = Inference(node)
            value = sess.run(None, {})[0]

            if not use_numpy:
                import torch

                yield f"{prefix}{node.output[0]}", (torch.from_numpy(value))
            else:
                yield f"{prefix}{node.output[0]}", (value)

        if node.op_type in {"Loop", "Body", "Scan"}:
            for att in node.attribute:
                assert (
                    att.type != onnx.AttributeProto.GRAPHS
                ), "Not implemented for type AttributeProto.GRAPHS."
                if att.type == onnx.AttributeProto.GRAPH:
                    yield from iterator_initializer_constant(
                        att.g, use_numpy=use_numpy, prefix=f"{prefix}{name}"
                    )


def tensor_statistics(tensor: Union[np.ndarray, TensorProto]) -> Dict[str, Union[float, str]]:
    """
    Produces statistics on a tensor.

    :param tensor: tensor
    :return: statistics

    .. runpython::
        :showcode:

        import pprint
        import numpy as np
        from onnx_diagnostic.helpers.onnx_helper import tensor_statistics

        t = np.random.rand(40, 50).astype(np.float16)
        pprint.pprint(tensor_statistics(t))
    """
    from .helper import size_type

    if isinstance(tensor, TensorProto):
        # pyrefly: ignore[bad-assignment]
        tensor = to_array_extended(tensor)
    assert isinstance(tensor, np.ndarray)  # type checking
    itype = np_dtype_to_tensor_dtype(tensor.dtype)
    stat = dict(
        mean=float(tensor.mean()),
        std=float(tensor.std()),
        shape="x".join(map(str, tensor.shape)),
        numel=tensor.size,
        size=tensor.size * size_type(tensor.dtype),
        itype=itype,
        stype=onnx_dtype_name(itype),
        min=float(tensor.min()),
        max=float(tensor.max()),
        nnan=float(np.isnan(tensor).sum()),
    )

    if tensor.size < 8:
        return stat

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            hist = np.array(
                [
                    0,
                    1e-10,
                    1e-8,
                    1e-7,
                    1e-6,
                    1e-5,
                    0.0001,
                    0.001,
                    0.01,
                    0.1,
                    0.5,
                    1,
                    1.96,
                    10,
                    1e2,
                    1e3,
                    1e4,
                    1e5,
                    1e6,
                    1e7,
                    1e8,
                    1e10,
                    1e50,
                ],
                dtype=tensor.dtype,
            )
        except OverflowError as e:
            from .helper import string_type

            raise ValueError(
                f"Unable to convert one value into {tensor.dtype}, "
                f"tensor={string_type(tensor, with_shape=True)}"
            ) from e
    hist = np.array(sorted(set(hist[~np.isinf(hist)])), dtype=tensor.dtype)
    ind = np.digitize(np.abs(tensor).reshape((-1,)), hist, right=True)
    cou = np.bincount(ind, minlength=ind.shape[0] + 1)
    stat.update(
        dict(zip([f">{x}" for x in hist], [int(i) for i in (cou.sum() - np.cumsum(cou))]))
    )
    ii = (np.arange(9) + 1) / 10
    qu = np.quantile(tensor, ii)
    stat.update({f"q{i}": float(q) for i, q in zip(ii, qu)})
    return stat


class NodeCoordinates:
    """
    A way to localize a node,
    path is a tuple of three information, node index, node type, node name.
    """

    __slots__ = ("node", "path")

    def __init__(
        self,
        node: Union[TensorProto, NodeProto, onnx.SparseTensorProto, ValueInfoProto, str],
        path: Tuple[Tuple[int, str, str], ...],
    ):
        assert isinstance(path, tuple), f"Unexpected type {type(path)} for path"
        assert all(isinstance(t, tuple) for t in path), f"Unexpected type in path={path}"
        self.node = node
        self.path = path

    def __str__(self) -> str:
        "usual"
        if isinstance(self.node, str):
            return f"{self.path_to_str()} :: {self.node!r}"
        return f"{self.path_to_str()} :: {pretty_onnx(self.node)}"

    def path_to_str(self) -> str:
        "Strings representing coordinates."
        return "x".join(f"({':'.join(map(str, t))})" for t in self.path)


class ResultFound:
    """Class returned by :func:`enumerate_results`."""

    __slots__ = ("consumer", "name", "producer")

    def __init__(
        self,
        name: str,
        producer: Optional[NodeCoordinates],
        consumer: Optional[NodeCoordinates],
    ):
        assert isinstance(name, str), f"unexpected type {type(name)} for name"
        self.name = name
        self.producer = producer
        self.consumer = consumer

    def __str__(self) -> str:
        "usuals"
        return (
            f"<< {self.name} - {self.consumer}"
            if self.producer is None
            else f">> {self.name} - {self.producer}"
        )


def enumerate_results(
    proto: Union[FunctionProto, GraphProto, ModelProto, Sequence[NodeProto]],
    name: Union[Set[str], str],
    verbose: int = 0,
    coordinates: Optional[List[Tuple[int, str, str]]] = None,
) -> Iterator[ResultFound]:
    """
    Iterates on all nodes, attributes to find where a name is used.

    :param proto: a proto
    :param name: name or names to find
    :param verbose: verbosity
    :param coordinates: coordinates of a node
    :return: iterator on :class:`ResultFound`
    """
    if not isinstance(name, set):
        name = {name}
    coordinates = coordinates or []
    assert all(
        isinstance(c, tuple) for c in coordinates
    ), f"Unexpected type in coordinates={coordinates}"
    indent = "  " * len(coordinates)
    if isinstance(proto, ModelProto):
        if verbose:
            print(f"[enumerate_results] {indent}searching for {name!r} into ModelProto...")
        yield from enumerate_results(proto.graph, name, verbose=verbose)
    elif isinstance(proto, FunctionProto):
        if verbose:
            print(f"[enumerate_results] {indent}searching for {name!r} into FunctionProto...")
        for i in proto.input:
            if i in name:
                r = ResultFound(
                    i,
                    NodeCoordinates(i, tuple([*coordinates, (-1, "INPUT", "")])),  # noqa: C409
                    None,
                )
                if verbose > 1:
                    print(f"[enumerate_results] {indent}-- {r}")
                yield r
        yield from enumerate_results(proto.node, name, verbose=verbose)
        for i in proto.output:
            if i in name:
                r = ResultFound(
                    i,
                    None,
                    NodeCoordinates(
                        i, tuple([*coordinates, (len(proto.node), "OUTPUT", "")])  # noqa: C409
                    ),
                )
                if verbose > 1:
                    print(f"[enumerate_results] {indent}-- {r}")
                yield r
    elif isinstance(proto, GraphProto):
        if verbose:
            print(f"[enumerate_results] {indent}searching for {name!r} into GraphProto...")
        for i in proto.initializer:
            if i.name in name:
                r = ResultFound(
                    i.name,
                    NodeCoordinates(i, tuple([*coordinates, (-1, "INIT", "")])),  # noqa: C409
                    None,
                )
                if verbose > 1:
                    print(f"[enumerate_results] {indent}-- {r}")
                yield r
        for i in proto.sparse_initializer:
            if i.values.name in name:
                r = ResultFound(
                    i.values.name,
                    NodeCoordinates(i, tuple([*coordinates, (-1, "INIT", "")])),  # noqa: C409
                    None,
                )
                if verbose > 1:
                    print(f"[enumerate_results] {indent}-- {r}")
                yield r
        for i in proto.input:
            if i.name in name:
                r = ResultFound(
                    i.name,
                    NodeCoordinates(i, tuple([*coordinates, (-1, "INPUT", "")])),  # noqa: C409
                    None,
                )
                if verbose > 1:
                    print(f"[enumerate_results] {indent}-- {r}")
                yield r
        yield from enumerate_results(
            proto.node, name, verbose=verbose, coordinates=coordinates
        )
        for i in proto.output:
            if i.name in name:
                r = ResultFound(
                    i.name,
                    None,
                    NodeCoordinates(
                        i, tuple([*coordinates, (len(proto.node), "OUTPUT", "")])  # noqa: C409
                    ),
                )
                if verbose > 1:
                    print(f"[enumerate_results] {indent}-- {r}")
                yield r
    else:
        if verbose:
            print(
                f"[enumerate_results] {indent}searching for {name!r} into List[NodeProto]..."
            )
        for node_i, node in enumerate(proto):
            if set(node.input) & name:
                for n in node.input:
                    if n in name:
                        r = ResultFound(
                            n,
                            NodeCoordinates(
                                node,
                                tuple(  # noqa: C409
                                    [*coordinates, (node_i, node.op_type, node.name)]
                                ),
                            ),
                            None,
                        )
                        if verbose > 1:
                            print(f"[enumerate_results] {indent}-- {r}")
                        yield r
            if node.op_type in {"If", "Scan", "Loop", "SequenceMap"}:
                for att in node.attribute:
                    if att.type == onnx.AttributeProto.GRAPH:
                        yield from enumerate_results(
                            att.g,
                            name,
                            verbose=verbose,
                            coordinates=[*coordinates, (node_i, node.op_type, node.name)],
                        )
            if set(node.output) & name:
                for n in node.output:
                    if n in name:
                        r = ResultFound(
                            n,
                            None,
                            NodeCoordinates(
                                node,
                                tuple(  # noqa: C409
                                    [*coordinates, (node_i, node.op_type, node.name)]
                                ),
                            ),
                        )
                        if verbose > 1:
                            print(f"[enumerate_results] {indent}-- {r}")
                        yield r
    if verbose:
        print(f"[enumerate_results] {indent}done")


def shadowing_names(
    proto: Union[FunctionProto, GraphProto, ModelProto, Sequence[NodeProto]],
    verbose: int = 0,
    existing: Optional[Set[str]] = None,
    shadow_context: Optional[Set[str]] = None,
    post_shadow_context: Optional[Set[str]] = None,
) -> Tuple[Set[str], Set[str], Set[str]]:
    """
    Returns the shadowing names, the names created in the main graph
    after they were created in a subgraphs and the names created by the nodes.
    """
    if isinstance(proto, ModelProto):
        return shadowing_names(proto.graph)
    if isinstance(proto, GraphProto):
        assert (
            existing is None and shadow_context is None
        ), "existing must be None if nodes is None"
        return shadowing_names(
            proto.node,
            verbose=verbose,
            existing={i.name for i in proto.initializer}
            | {i.values.name for i in proto.sparse_initializer}
            | {i.name for i in proto.input if i.name},
            shadow_context=set(),
            post_shadow_context=set(),
        )
    if isinstance(proto, FunctionProto):
        assert (
            existing is None and shadow_context is None
        ), "existing must be None if nodes is None"
        return shadowing_names(
            proto.node,
            verbose=verbose,
            existing=set(i for i in proto.input if i),
            shadow_context=set(),
            post_shadow_context=set(),
        )

    assert (
        existing is not None and shadow_context is not None
    ), "existing must not be None if nodes is not None"
    shadow = set()
    shadow_context = shadow_context.copy()
    existing = existing.copy()
    created = set()
    post_shadow = set()
    for node in proto:
        not_empty = set(n for n in node.input if n)
        intersection = not_empty & existing
        assert len(intersection) == len(not_empty), (
            f"One input in {not_empty}, node={pretty_onnx(node)} "
            f"was not found in {existing}"
        )
        for att in node.attribute:
            if att.type == AttributeProto.GRAPH:
                g = att.g
                shadow |= {i.name for i in g.input} & shadow_context
                shadow |= {i.name for i in g.initializer} & shadow_context
                shadow |= {i.values.name for i in g.sparse_initializer} & shadow_context
                s, _ps, c = shadowing_names(
                    g.node, verbose=verbose, existing=existing, shadow_context=existing
                )
                shadow |= s
                created |= c

        not_empty = set(n for n in node.output if n)
        post_shadow |= not_empty & created
        shadow |= not_empty & shadow_context
        existing |= not_empty
        created |= not_empty
    return shadow, post_shadow, created


def get_hidden_inputs(graph: onnx.GraphProto) -> Set[str]:
    """
    Returns the hidden inputs (inputs coming from an upper context)
    used by a subgraph. It excludes empty names.
    """
    hidden = set()
    memo = (
        {i.name for i in graph.initializer}
        | {i.values.name for i in graph.sparse_initializer}
        | {i.name for i in graph.input}
    )
    for node in graph.node:
        for i in node.input:
            if i and i not in memo:
                hidden.add(i)
        for att in node.attribute:
            if att.type == onnx.AttributeProto.GRAPH and att.g:
                hid = get_hidden_inputs(att.g)
                less = set(h for h in hid if h not in memo)
                hidden |= less
        memo |= set(node.output)
    return hidden


def get_all_node_inputs(node: onnx.NodeProto) -> Set[str]:
    """
    Returns input and hidden inputs of a node.
    See :func:`get_hidden_inputs`. It excludes empty names.
    """
    start = {i for i in node.input if i}
    if node.op_type in {"Scan", "Loop", "If"}:
        for att in node.attribute:
            if att.type == onnx.AttributeProto.GRAPH:
                start |= get_hidden_inputs(att.g)
    return start


def extract_subset_of_nodes(
    model: ModelProto,
    name: str,
    node_index: Optional[int] = None,
    cut_points: Optional[Set[str]] = None,
) -> List[NodeProto]:
    """
    Extracts the minimal subgraphs which can produce the output ``name``
    knowing ``cut_points``.

    :param model: original model
    :param name: result name
    :param node_index: if the node index is known, otherwise searches for it
    :param cut_points: the known results or input name otherwise
    :return: minimal list of nodes
    """
    if node_index is None:
        for i, node in enumerate(model.graph.node):
            if name in node.output:
                node_index = i
                break
    assert node_index is not None and node_index < len(model.graph.node), (
        f"node_index={node_index} (n_nodes={len(model.graph.node)}) "
        f"is still empty or wrong for result {name!r}"
    )
    assert name in model.graph.node[node_index].output, (
        f"Unable to find {name!r} in {model.graph.node[node_index].output}, "
        f"node={pretty_onnx(model.graph.node[node_index])}"
    )
    if cut_points is None:
        cut_points = {n.name for n in model.graph.input} | {
            n.name for n in model.graph.initializer
        }
    elif model.graph.initializer:
        cut_points = cut_points | {n.name for n in model.graph.initializer}

    node = model.graph.node[node_index]
    selected = {node_index}
    current_node_index = node_index
    current_input_index = 0
    intermediate = {name}
    cut_points -= {name}
    cached: Dict[int, List[str]] = {}
    inputs = set(k for k in node.input if k)
    while not (inputs <= cut_points) and current_node_index >= 0:
        node = model.graph.node[current_node_index]
        # node inputs including hidden ones
        if current_node_index in cached:
            node_inputs = cached[current_node_index]
        else:
            set_inputs = set(i for i in node.input if i)
            if node.op_type in {"Scan", "If", "Loop"}:
                # there are hidden inputs
                for att in node.attribute:
                    if att.type == onnx.AttributeProto.GRAPH:
                        set_inputs |= get_hidden_inputs(att.g)
            node_inputs = list(set_inputs)
            cached[current_node_index] = node_inputs
        # processing
        if current_input_index == 0 or not node_inputs:
            needs = [o for o in node.output if o in intermediate and o not in cut_points]
            if needs:
                selected.add(current_node_index)
                if not node_inputs:
                    current_node_index -= 1
                    current_input_index = 0
                    continue
            else:
                current_node_index -= 1
                current_input_index = 0
                continue
        # more intermediate results
        assert current_input_index < len(node_inputs), (
            f"current_input_index={current_input_index} but node_inputs={node_inputs}, "
            f"node={pretty_onnx(node)}"
        )
        res = node_inputs[current_input_index]
        if res not in cut_points:
            intermediate.add(res)
        current_input_index += 1
        if current_input_index >= len(node_inputs):
            current_node_index -= 1
            current_input_index = 0

    return [model.graph.node[i] for i in sorted(selected)]


def make_submodel(
    nodes: List[NodeProto],
    ir_version: int,
    opset_imports: List[OperatorSetIdProto],
    output_names: List[str],
    type_rank_fn: Callable[[str], Tuple[int, int]],
) -> ModelProto:
    """
    Creates a model with the given list of nodes.
    It computes the minimum list of inputs needed for this model.
    The function assumes the nodes are sorted.
    It does not handle yet subgraphs.

    :param nodes: list of nodes
    :param ir_version: ir version
    :param opset_imports: opset import
    :param output_names: desired outputs
    :param function: function returning the type and the rank of a result
    :return: model proto
    """

    def _mkv_(name, itype, irank):
        return oh.make_tensor_value_info(name, itype, [f"{name}_d{i}" for i in range(irank)])

    not_known: Set[str] = set()
    for node in nodes[::-1]:
        not_known -= {o for o in node.output if o}
        not_known |= {i for i in node.input if i}
        if node.op_type in {"Scan", "If", "Loop"}:
            # there are hidden inputs
            for att in node.attribute:
                if att.type == onnx.AttributeProto.GRAPH:
                    not_known |= get_hidden_inputs(att.g)

    model = oh.make_model(
        oh.make_graph(
            nodes,
            "submodel",
            [_mkv_(n, *type_rank_fn(n)) for n in sorted(not_known) if n],
            [_mkv_(n, *type_rank_fn(n)) for n in sorted(output_names) if n],
        ),
        ir_version=ir_version,
        opset_imports=opset_imports,
    )
    return model


def get_tensor_shape(
    obj: Union[onnx.ValueInfoProto, onnx.TypeProto, onnx.TensorProto],
) -> Optional[List[Optional[Union[int, str]]]]:
    """Returns the shape if that makes sense for this object."""
    if isinstance(obj, ValueInfoProto):
        return get_tensor_shape(obj.type)
    elif not isinstance(obj, onnx.TypeProto):
        raise TypeError(f"Unexpected type {type(obj)!r}.")
    if not obj.tensor_type.HasField("shape"):
        return None
    shape = []
    for d in obj.tensor_type.shape.dim:
        v = d.dim_value if d.dim_value > 0 else d.dim_param
        shape.append(v)
    if not shape:
        return shape
    return [None if s in (0, "") else s for s in shape]


def _enumerate_model_node_outputs(
    model: ModelProto, add_node: bool = False, order: bool = False
) -> Iterable[Union[str, Tuple[str, NodeProto]]]:
    """
    Enumerates all the nodes of a model.

    :param model: :epkg:`ONNX` graph
    :param add_node: if False, the function enumerates
        all output names from every node, otherwise, it
        enumerates tuple (output name, node)
    :param order: goes through outputs following the graph order
    :return: enumerator
    """
    assert hasattr(model, "graph"), "Parameter model is not an ONNX model but {type(model)}"
    if order:
        edges = []
        d_order = {}
        node_names = {}
        for inp in model.graph.input:
            d_order[0, inp.name] = 0
        for node in model.graph.node:
            d_order[1, node.name] = 0
            for i in node.input:
                edges.append(("in", i, node.name))
            for o in node.output:
                edges.append(("out", o, node.name))
                node_names[o] = node
                d_order[0, o] = 0

        modif = 1
        n_iter = 0
        while modif > 0 and n_iter <= len(model.graph.node):
            modif = 0
            n_iter += 1
            for kind, data_name, node_name in edges:
                if kind == "in":
                    if (0, data_name) not in d_order:
                        continue
                    if d_order[0, data_name] + 1 > d_order[1, node_name]:
                        modif += 1
                        d_order[1, node_name] = d_order[0, data_name] + 1
                else:
                    if d_order[1, node_name] + 1 > d_order[0, data_name]:
                        modif += 1
                        d_order[0, data_name] = d_order[1, node_name] + 1

        orders = [(v, k) for k, v in d_order.items()]
        orders.sort()

        for _, k in orders:
            if k[0] == 1:
                continue
            out = k[1]
            if out not in node_names:
                continue
            yield (out, node_names[out]) if add_node else out
    else:
        for node in model.graph.node:
            for out in node.output:
                yield (out, node) if add_node else out


def onnx_remove_node_unused(
    graph: Union[onnx.GraphProto, onnx.FunctionProto], recursive=True
) -> Union[onnx.GraphProto, onnx.FunctionProto]:
    """
    Removes unused nodes of the graph. An unused node
    is not involved in the output computation.

    :param onnx_model: onnx model
    :param recursive: looks into subgraphs
    :return: new Graph
    """
    is_function = isinstance(graph, FunctionProto)

    # mark outputs
    marked: Dict[str, Set[str]] = (
        {o: set() for o in graph.output}
        if is_function
        else {o.name: set() for o in graph.output}
    )
    nodes = list(graph.node)

    # mark node output
    for node in reversed(nodes):
        used = False
        for o in node.output:
            if o and o in marked:
                for i in get_all_node_inputs(node):
                    marked[o].add(i)
                    used = True
        if used:
            for i in get_all_node_inputs(node):
                marked[i] = set()

    # removed nodes
    removed = set()
    marked_set = set(marked)
    for ind, node in enumerate(nodes):
        if not ({o for o in node.output if o} & marked_set):
            removed.add(ind)

    new_nodes = [node for i, node in enumerate(nodes) if i not in removed]

    # Finally create the new graph.
    if is_function:
        return oh.make_function(
            graph.domain,
            graph.name,
            graph.input,
            graph.output,
            new_nodes,
            opset_imports=graph.opset_import,
            attributes=graph.attribute,
            doc_string=graph.doc_string,
        )

    initializers = [i for i in graph.initializer if i.name in marked]
    sparse_initializers = [i for i in graph.sparse_initializer if i.values.name in marked]
    new_graph = oh.make_graph(
        new_nodes,
        graph.name,
        graph.input,
        graph.output,
        initializers,
        sparse_initializer=sparse_initializers,
    )
    new_graph.value_info.extend(graph.value_info)
    return new_graph


def select_model_inputs_outputs(
    model: ModelProto,
    outputs: Optional[List[str]] = None,
    inputs: Optional[List[str]] = None,
    infer_shapes: bool = True,
    overwrite: Optional[Dict[str, Any]] = None,
    remove_unused: bool = True,
    verbose: int = 0,
) -> ModelProto:
    """
    Takes a model and changes its outputs.

    :param model: :epkg:`ONNX` model
    :param inputs: new inputs, same ones if None
    :param outputs: new outputs, same ones if None
    :param infer_shapes: infer inputs and outputs shapes
    :param overwrite: overwrite type and shapes for
        inputs or outputs, *overwrite* is a
        dictionary `{'name': (numpy dtype, shape)}`
    :param remove_unused: remove unused nodes from the graph
    :param verbose: display information while converting
    :return: modified model

    The function removes unneeded nodes.

    The following example shows how to change the inputs of model
    to bypass the first nodes. Shape inferences fails to determine
    the new inputs type. They need to be overwritten.
    `verbose=1` shows the number of deleted nodes.

    ::

        import onnx
        from onnx_extended.tools.onnx_nodes import select_model_inputs_outputs

        onx = onnx.load(path)
        onx2 = select_model_inputs_outputs(
            onx, inputs=["a", "b"],
            infer_shapes=True, verbose=1,
            overwrite={'a': (numpy.int32, None), 'b': (numpy.int64, None)})
        onnx.save(onx2, path2)
    """
    if not isinstance(model, ModelProto):
        raise TypeError(f"Unexpected type {type(model)} for model.")
    if inputs is not None and not isinstance(inputs, list):
        inputs = [inputs]
    if outputs is not None and not isinstance(outputs, list):
        outputs = [outputs]
    if inputs is None:
        inputs = [i.name for i in model.graph.input]
    if outputs is None:
        outputs = [o.name for o in model.graph.output]

    mark_var = {}
    for out in _enumerate_model_node_outputs(model):
        mark_var[out] = 0
    for inp in inputs:
        mark_var[inp] = 0
    for out in outputs:
        assert out in mark_var, f"Output {out!r} not found in model."
        mark_var[out] = 1

    nodes = list(model.graph.node[::-1])
    mark_op = {}
    for node in list(nodes):
        mark_op[id(node)] = 0

    # We mark all the nodes we need to keep.
    nb = 1
    while nb > 0:
        nb = 0
        for node in nodes:
            if mark_op[id(node)] == 1:
                continue
            mod = False
            for out in node.output:
                if mark_var[out] == 1:
                    mark_op[id(node)] = 1
                    mod = True
                    break
            if not mod:
                continue

            node_inputs = get_all_node_inputs(node)

            nb += 1
            for inp in node_inputs:
                if inp in inputs:
                    continue
                if mark_var.get(inp, 0) == 1:
                    continue
                mark_var[inp] = 1
                nb += 1

    # All nodes verifies mark_op[node.name] == 1
    keep_nodes = [node for node in nodes[::-1] if mark_op[id(node)] == 1]

    known_shapes = {}
    if infer_shapes:
        shapes = onnx.shape_inference.infer_shapes(model)
        for shape in shapes.graph.value_info:
            known_shapes[shape.name] = shape.type
        for shape in shapes.graph.input:
            known_shapes[shape.name] = shape.type
        for shape in shapes.graph.output:
            known_shapes[shape.name] = shape.type
    else:
        for shape in model.graph.input:
            known_shapes[shape.name] = shape.type
        for shape in model.graph.output:
            known_shapes[shape.name] = shape.type

    var_in = []
    existing = {i.name: i for i in model.graph.input}
    for name in inputs:
        if overwrite is not None and name in overwrite:
            dtype, shape = overwrite[name]
            proto_dtype = np_dtype_to_tensor_dtype(dtype)
            value_info = oh.make_tensor_value_info(name, proto_dtype, shape)
        elif name in known_shapes:
            info = known_shapes[name].tensor_type
            proto_dtype = info.elem_type
            if proto_dtype == 0:
                value_info = ValueInfoProto()
                value_info.name = name
            else:
                shape = get_tensor_shape(known_shapes[name])
                value_info = oh.make_tensor_value_info(name, proto_dtype, shape)
        elif name in existing:
            value_info = existing[name]
        else:
            value_info = ValueInfoProto()
            value_info.name = name
        var_in.append(value_info)

    var_out = []
    existing = {i.name: i for i in model.graph.output}
    for name in outputs:
        if overwrite is not None and name in overwrite:
            dtype, shape = overwrite[name]
            proto_dtype = np_dtype_to_tensor_dtype(dtype)
            value_info = oh.make_tensor_value_info(name, proto_dtype, shape)
        elif name in known_shapes:
            info = known_shapes[name].tensor_type
            proto_dtype = info.elem_type
            if proto_dtype == 0:
                value_info = ValueInfoProto()
                value_info.name = name
            else:
                shape = get_tensor_shape(known_shapes[name])
                value_info = oh.make_tensor_value_info(name, proto_dtype, shape)
        elif name in existing:
            value_info = existing[name]
        else:
            value_info = ValueInfoProto()
            value_info.name = name
        var_out.append(value_info)

    graph = oh.make_graph(
        keep_nodes,
        model.graph.name,
        var_in,
        var_out,
        model.graph.initializer,
        sparse_initializer=model.graph.sparse_initializer,
    )
    if remove_unused:
        graph = onnx_remove_node_unused(graph, recursive=False)
        assert isinstance(graph, GraphProto)  # type checking
    onnx_model = oh.make_model(graph, functions=model.functions)
    onnx_model.ir_version = model.ir_version
    onnx_model.producer_name = model.producer_name
    onnx_model.producer_version = model.producer_version
    onnx_model.domain = model.domain
    onnx_model.model_version = model.model_version
    onnx_model.doc_string = model.doc_string
    if model.metadata_props:
        values = {p.key: p.value for p in model.metadata_props}
        oh.set_model_props(onnx_model, values)

    del onnx_model.opset_import[:]
    for oimp in model.opset_import:
        op_set = onnx_model.opset_import.add()
        op_set.domain = oimp.domain
        op_set.version = oimp.version

    return onnx_model
