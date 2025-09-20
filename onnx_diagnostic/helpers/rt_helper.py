from typing import Any, Dict, List, Union
import numpy as np
import onnx
import torch
from .helper import string_type, flatten_object
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
    is_modelbuilder: bool = False,
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
    :param is_modelbuilder: if True, the exporter is ModelBuilder, and we need to reorder
        the past_key_values inputs to match the expected order, and get rid of position_ids.
    :return: feeds dictionary
    """
    # NOTE: position_ids is a special case because ModelBuilder does not usually use it,
    # because it's fued into rotary embedding in GQA.
    if is_modelbuilder and isinstance(inputs, dict):
        inputs.pop("position_ids", None)  # Ensure 'position_ids' absent before removing.

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
        else (
            [i.name for i in proto.get_inputs()]
            if hasattr(proto, "get_inputs")
            else (proto.input_names if hasattr(proto, "input_names") else proto)
        )
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

    if copy:
        flat = [t.copy() if hasattr(t, "copy") else t.clone() for t in flat]
    # bool, int, float, onnxruntime does not support float, bool, int
    new_flat = []
    for i in flat:
        if isinstance(i, bool):
            i = np.array(i, dtype=np.bool_)
        elif isinstance(i, int):
            i = np.array(i, dtype=np.int64)
        elif isinstance(i, float):
            i = np.array(i, dtype=np.float32)
        new_flat.append(i)

    # NOTE: model builder has a different order for past_key_values
    #       we need to reorder them to match the expected order
    if is_modelbuilder:
        # We assume that if "past_key_values" is in the names when it's
        # modelbuilder
        non_past_kv_input_names = [n for n in names if "past_key_values" not in n]
        past_kv_names = [n for n in names if "past_key_values" in n]
        reorder_past_kv_names = reorder_modelbuilder_cache_to_torch(past_kv_names)
        names = non_past_kv_input_names + reorder_past_kv_names
    return dict(zip(names, new_flat))


def reorder_modelbuilder_cache_to_torch(past_kv: List[Any]) -> List[Any]:
    """
    Reorders the past_kvs for ModelBuilder to match the expected order
    by PyTorch exported models.

    .. note::
        This function can take either the names or the actual tensors
        as long as they are in a list.

    Conceptually,

    From::

        [past_key_values.0.key, past_key_values.0.value,
        past_key_values.1.key, past_key_values.1.value, ...]

    To::

        [past_key_values.0.key, past_key_values.1.key,
        ..., past_key_values.0.value, past_key_values.1.value, ...]

    :param past_kv: list of flattened inputs
    :return: reordered list of flattened inputs
    """
    total_len = len(past_kv)
    if total_len % 2 != 0:
        raise ValueError("The length of past_key_values should be even.")
    keys = []
    values = []
    for i in range(0, total_len, 2):
        keys.append(past_kv[i])
        values.append(past_kv[i + 1])
    return keys + values
