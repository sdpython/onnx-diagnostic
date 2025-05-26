from typing import Any, Dict, List, Union
import numpy as np
import onnx
import torch
from .helper import string_type, flatten_object
from .onnx_helper import dtype_to_tensor_dtype
from .cache_helper import is_cache_dynamic_registered


def name_type_to_onnx_dtype(name: str) -> int:
    if name == "tensor(int64)":
        return onnx.TensorProto.INT64
    if name == "tensor(float)":
        return onnx.TensorProto.FLOAT
    if name == "tensor(float16)":
        return onnx.TensorProto.FLOAT16
    raise AssertionError(f"Unexpected value {name!r}")


def make_feeds(
    proto: Union[onnx.ModelProto, List[str]],
    inputs: Any,
    use_numpy: bool = False,
    copy: bool = False,
    check_flatten: bool = True,
) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
    """
    Serializes the inputs to produce feeds expected
    by :class:`onnxruntime.InferenceSession`.

    :param proto: onnx model or list of names
    :param inputs: any kind of inputs
    :param use_numpy: if True, converts torch tensors into numpy arrays
    :param copy: a copy is made, this should be the case if the inputs is ingested
        by ``OrtValue``
    :param check_flatten: if True, checks the ``torch.utils._pytree.tree_flatten``
        returns the same number of outputs
    :return: feeds dictionary
    """
    # position_ids is a special case because ModelBuilder does not usually use it.
    # We use types to detect the best inputs.
    flat = flatten_object(inputs, drop_keys=True)
    assert (
        not check_flatten
        or not all(isinstance(obj, torch.Tensor) for obj in flat)
        or not is_cache_dynamic_registered(fast=True)
        or len(flat) == len(torch.utils._pytree.tree_flatten(inputs)[0])
    ), (
        f"Unexpected number of flattened objects, "
        f"{string_type(flat, with_shape=True)} != "
        f"{string_type(torch.utils._pytree.tree_flatten(inputs)[0], with_shape=True)}"
    )
    if use_numpy:
        flat = [t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else t for t in flat]
    names = (
        [i.name for i in proto.graph.input]
        if isinstance(proto, onnx.ModelProto)
        else ([i.name for i in proto.get_inputs()] if hasattr(proto, "get_inputs") else proto)
    )
    assert (
        isinstance(names, list)
        and len(names) <= len(flat)
        and (
            len(names) == len(flat)
            or isinstance(proto, onnx.ModelProto)
            or hasattr(proto, "get_inputs")
        )
    ), (
        f"Not the same number of given inputs {len(flat)} "
        f"and the number of model inputs {len(names)}, "
        f"type(names)={type(names)}, type(proto)={type(proto)}"
        f"\n-- inputs={string_type(inputs, with_shape=True)}"
        f"\n-- names={names}"
    )
    if len(names) < len(flat) and (
        isinstance(proto, onnx.ModelProto) or hasattr(proto, "get_inputs")
    ):

        typed_names = (
            [(i.name, i.type.tensor_type.elem_type) for i in proto.graph.input]
            if isinstance(proto, onnx.ModelProto)
            else [(i.name, name_type_to_onnx_dtype(i.type)) for i in proto.get_inputs()]
        )

        new_flat = []
        pos = 0
        for _name, dtype in typed_names:
            assert isinstance(
                dtype, int
            ), f"Unexpected value for dtype={dtype!r}, type(proto)={type(proto)}"
            itype = dtype_to_tensor_dtype(flat[pos].dtype)
            while dtype != itype:
                pos += 1
                if pos >= len(flat):
                    break
                itype = dtype_to_tensor_dtype(flat[pos].dtype)
            if pos >= len(flat):
                break
            new_flat.append(flat[pos])
            pos += 1
        assert len(new_flat) == len(names), (
            f"Unable to align expected input {names} with the given input"
            f"\n-- inputs: {string_type(inputs, with_shape=True)}"
            f"\n-- model: {string_type(proto.graph.input)}"
        )
        flat = new_flat

    if copy:
        flat = [t.copy() if hasattr(t, "copy") else t.clone() for t in flat]
    return dict(zip(names, flat))
