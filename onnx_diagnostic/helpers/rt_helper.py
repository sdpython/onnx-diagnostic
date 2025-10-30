from typing import Any, Dict, List, Tuple, Union
import numpy as np
import onnx
import torch
from .helper import string_type, flatten_object
from .ort_session import InferenceSessionForTorch


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
        # or not is_cache_dynamic_registered(fast=True)
        or len(flat) == len(torch.utils._pytree.tree_flatten(inputs)[0])
    ), (
        f"Unexpected number of flattened objects, "
        f"{string_type(flat, with_shape=True)} != "
        f"{string_type(torch.utils._pytree.tree_flatten(inputs)[0], with_shape=True)}"
    )
    if use_numpy:
        from .torch_helper import to_numpy

        flat = [to_numpy(t) if isinstance(t, torch.Tensor) else t for t in flat]
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
    return dict(zip(names, new_flat))


def _get_dim(i: int, s: Union[str, int], batch: int = 1) -> int:
    if isinstance(s, int):
        return s
    if s == "batch":
        return batch
    # Everything else is cache length or sequence length.
    return 0


_DTYPES = {
    "tensor(float)": torch.float32,
    "tensor(float16)": torch.float16,
    "tensor(bfloat16)": torch.bfloat16,
    "tensor(int64)": torch.int64,
    "tensor(int32)": torch.int32,
}


def rt_type_to_torch_dtype(typename: str) -> torch.dtype:
    """Converts a string such as ``tensor(float)`` into a dtype (torch.float32)."""
    return _DTYPES[typename]


def make_empty_cache(
    batch: int,
    onnx_input_names: List[str],
    onnx_input_shapes: List[Tuple[Union[int, str], ...]],
    onnx_input_types: List[str],
) -> Dict[str, torch.Tensor]:
    """
    Creates an empty cache. Example:

    .. code-block:: python

        make_empty_cache(
            1,
            sess.input_names[2:],
            [i.shape for i in sess.get_inputs()[2:]],
            [i.type for i in sess.get_inputs()[2:]],
        )
    """
    feeds = {}
    for name, shape, dtype in zip(onnx_input_names, onnx_input_shapes, onnx_input_types):
        new_shape = tuple(_get_dim(i, s, batch=batch) for i, s in enumerate(shape))
        feeds[name] = torch.empty(new_shape, dtype=rt_type_to_torch_dtype(dtype))
    return feeds


def onnx_generate(
    model_or_path: Union[onnx.ModelProto, str, InferenceSessionForTorch],
    input_ids: torch.Tensor,
    eos_token_id: int,
    max_new_tokens=100,
    return_session: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, InferenceSessionForTorch]]:
    """
    Implements a simple method ``generate`` for an ONNX model.
    The function does not expect any ``position_ids`` as input.

    :param model_or_path: model or loaded model
    :param input_ids: input tokens
    :param eos_token_ids: token representing the end of an answer
    :param max_new_tokens: stops after this number of generated tokens
    :param return_session: returns the instance of class
        :class:`InferenceSessionForTorch
        <onnx_diagnostic.helpers.ort_session.InferenceSessionForTorch>`
        created if necessary
    :return: input tokens concatenated with new tokens
    """
    if not isinstance(model_or_path, InferenceSessionForTorch):
        providers = ["CUDAExecutionProvider"] if input_ids.is_cuda else []
        providers.append("CPUExecutionProvider")
        session = InferenceSessionForTorch(model_or_path, providers=providers)
    else:
        session = model_or_path

    input_shapes = session.input_shapes
    input_names = session.input_names
    input_types = session.input_types

    assert (
        len(input_names) > 2
        and input_names[:2] == ["input_ids", "attention_mask"]
        and input_names[2].startswith("past_key_values")
    ), f"Only text generation is supported but input_names == {input_names}"

    # First call: prefill
    feeds = dict(
        input_ids=input_ids,
        attention_mask=torch.ones(
            input_ids.shape, dtype=input_ids.dtype, device=input_ids.device
        ),
        **make_empty_cache(
            input_ids.shape[0], input_names[2:], input_shapes[2:], input_types[2:]
        ),
    )

    outputs = session.run(None, feeds)

    # Next calls: decode
    for _ in range(max_new_tokens):
        next_token_logits = outputs[0][:, -1, :]

        # The most probable next token is chosen.
        next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        # But we could select it using a multinomial law
        # <<< probs = torch.softmax(next_token_logits / temperature, dim=-1)
        # <<< top_probs, top_indices = torch.topk(probs, top_k)
        # <<< next_token_id = top_indices[torch.multinomial(top_probs, 1)]

        if next_token_id.item() == eos_token_id:
            break
        input_ids = torch.cat([input_ids, next_token_id.to(input_ids.device)], dim=-1)
        feeds = dict(
            input_ids=next_token_id,
            attention_mask=torch.ones(
                input_ids.shape, dtype=input_ids.dtype, device=input_ids.device
            ),
        )
        feeds.update(dict(zip(input_names[2:], outputs[1:])))
        outputs = session.run(None, feeds)

    if return_session:
        return input_ids, session
    return input_ids
