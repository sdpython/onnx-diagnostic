import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import onnx
import torch
from .helper import string_type, flatten_object, max_diff
from .torch_helper import torch_deepcopy
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


def generate_and_validate(
    model,
    input_ids: torch.Tensor,
    eos_token_id: int,
    max_new_tokens: int = 100,
    session: Optional[Union[InferenceSessionForTorch, onnx.ModelProto, str]] = None,
    atol: float = 0.1,
) -> Union[torch.Tensor, Tuple[torch.Tensor, List[Dict]]]:
    """
    Implements a simple method ``generate`` for a torch model.
    The function does not expect any ``position_ids`` as input.
    The function also checks the outputs coming from an onnx model
    are close to the output the torch model produces.

    :param model_or_path: model or loaded model
    :param input_ids: input tokens
    :param eos_token_ids: token representing the end of an answer
    :param max_new_tokens: stops after this number of generated tokens
    :param session: the onnx model
    :return: input tokens concatenated with new tokens,
        if session is not null, it also returns the maximum differences
        at every iterations

    See example given with function :func:`onnx_generate
    <onnx_diagnostic.helpers.rt_helper.onnx_generate>`.
    """
    if session is not None:
        if not isinstance(session, InferenceSessionForTorch):
            providers = ["CUDAExecutionProvider"] if input_ids.is_cuda else []
            providers.append("CPUExecutionProvider")
            session = InferenceSessionForTorch(session, providers=providers)

    # First call: prefill
    attention_mask = torch.ones(
        input_ids.shape, dtype=input_ids.dtype, device=input_ids.device
    )
    if session:
        feeds = {
            **dict(zip(session.input_names[:2], [input_ids, attention_mask])),
            **make_empty_cache(
                input_ids.shape[0],
                session.input_names[2:],
                session.input_shapes[2:],
                session.input_types[2:],
            ),
        }
        onnx_results = session.run(None, feeds)

    outputs = model(input_ids, use_cache=True, attention_mask=attention_mask)

    if session:
        diff = max_diff(outputs, onnx_results)
        assert isinstance(diff["abs"], float) and diff["abs"] <= atol, (
            f"Unexpected issue with {type(model)}\ndiff={diff}"
            f"\ninput_ids.shape={input_ids.shape}"
            f"\nexpected={string_type(outputs, with_shape=True, with_min_max=True)}"
            f"\n     got=\n"
            f"{string_type(onnx_results, with_shape=True, with_min_max=True)}\n"
            f"feeds={string_type(feeds, with_shape=True, with_min_max=True)}"
        )
        diffs = [diff]

    # Next calls: decode
    for iteration in range(max_new_tokens):
        next_token_logits = outputs.logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        if next_token_id.item() == eos_token_id:
            break
        input_ids = torch.cat([input_ids, next_token_id], dim=-1)
        attention_mask = torch.ones(
            input_ids.shape, dtype=input_ids.dtype, device=input_ids.device
        )
        if session:
            feeds = dict(
                zip(
                    session.input_names,
                    [
                        t.detach()
                        for t in torch_deepcopy(
                            flatten_object(
                                [next_token_id, attention_mask, outputs.past_key_values]
                            )
                        )
                    ],
                )
            )
            onnx_results = session.run(None, feeds)
        outputs = model(
            next_token_id,
            use_cache=True,
            past_key_values=outputs.past_key_values,
            attention_mask=attention_mask,
        )
        if session:
            diff = max_diff(outputs, onnx_results)
            assert isinstance(diff["abs"], float) and diff["abs"] <= atol, (
                f"Unexpected issue with {type(model)}, iteration={iteration}"
                f"\ndiff={diff}\ninput_ids.shape={input_ids.shape}"
                f"\nexpected={string_type(outputs, with_shape=True, with_min_max=True)}"
                f"\n     got=\n"
                f"{string_type(onnx_results, with_shape=True, with_min_max=True)}\n"
                f"feeds={string_type(feeds, with_shape=True, with_min_max=True)}"
            )
            diffs.append(diff)
    if session:
        return input_ids, diffs
    return input_ids


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

    .. runpython::
        :showcode:

        import os
        from onnx_diagnostic.helpers import string_type, string_diff
        from onnx_diagnostic.helpers.rt_helper import (
            onnx_generate,
            generate_and_validate,
            onnx_generate_with_genai,
        )
        from onnx_diagnostic.torch_models.hghub import get_untrained_model_with_inputs
        from onnx_diagnostic.torch_export_patches import torch_export_patches
        from onnx_diagnostic.export.api import to_onnx

        mid = "arnir0/Tiny-LLM"
        print(f"-- get model for {mid!r}")
        data = get_untrained_model_with_inputs(mid)
        model, inputs, ds = data["model"], data["inputs"], data["dynamic_shapes"]
        del inputs["position_ids"]
        del ds["position_ids"]
        input_ids = inputs["input_ids"]

        print(f"-- input_ids={input_ids.shape}")
        print(f"-- inputs: {string_type(inputs, with_shape=True)}")
        print(f"-- dynamic_shapes: {string_type(ds)}")
        folder = "dump_test"
        os.makedirs(folder, exist_ok=True)
        model_name = os.path.join(folder, "model.onnx")
        print("-- test_onnx_generate: export model")
        with torch_export_patches(patch_transformers=True, patch_torch=False):
            to_onnx(
                model,
                (),
                kwargs=inputs,
                dynamic_shapes=ds,
                filename=model_name,
                exporter="custom",  # custom, dynamo or onnx-dynamo, modelbuilder
            )

        print("-- generate with onnx")
        onnx_outputs = onnx_generate(model_name, input_ids[:1], 2, max_new_tokens=10)
        print("-- onnx output", onnx_outputs)

        # The example continues with other functions doing the same.
        print("-- generate with pytorch")
        torch_outputs, diffs = generate_and_validate(
            model, input_ids[:1], 2, max_new_tokens=10, session=model_name
        )
        print("-- torch output", torch_outputs)
        print("-- differences at each step:")
        for i, d in enumerate(diffs):
            print(f"iteration {i}: {string_diff(d)}")

        print("-- generate with genai")
        genai_outputs, session = onnx_generate_with_genai(
            model_name,
            input_ids[:1],
            max_new_tokens=10,
            return_session=True,
            transformers_config=data["configuration"],
        )
        print("-- genai output", genai_outputs)
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


def onnx_generate_with_genai(
    model_or_path: Union[onnx.ModelProto, str, InferenceSessionForTorch],
    input_ids: torch.Tensor,
    max_new_tokens=100,
    return_session: bool = False,
    transformers_config: Optional[Any] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, InferenceSessionForTorch]]:
    """
    Uses :epkg:`onnxruntime-genai` to implement a simple method ``generate``
    for an ONNX model. The function does not expect any ``position_ids`` as input.

    :param model_or_path: model or loaded model
    :param input_ids: input tokens
    :param eos_token_ids: token representing the end of an answer
    :param max_new_tokens: stops after this number of generated tokens
    :param return_session: returns the instance of class
        :class:`InferenceSessionForTorch
        <onnx_diagnostic.helpers.ort_session.InferenceSessionForTorch>`
        created if necessary
    :param transformers_config: write configuration
        if missing and if this configuration is provided
    :return: input tokens concatenated with new tokens

    See example given with function :func:`onnx_generate
    <onnx_diagnostic.helpers.rt_helper.onnx_generate>`.
    """
    import onnxruntime_genai as og

    if not isinstance(model_or_path, og.Model):
        from .model_builder_helper import make_genai_config

        assert isinstance(
            model_or_path, str
        ), f"Only a filename is allowed for model_or_path but type is {type(model_or_path)}"
        folder = os.path.dirname(model_or_path)
        assert os.path.exists(folder), f"Folder {folder!r} does not exists."
        assert os.path.exists(model_or_path), f"Folder {model_or_path!r} does not exists."
        config_file = os.path.join(folder, "genai_config.json")
        if not os.path.exists(config_file):
            if not transformers_config:
                raise FileNotFoundError(
                    f"Folder {model_or_path!r} does not contain 'genai_config.json'."
                )
            config = make_genai_config(transformers_config, model_or_path)
            with open(config_file, "w") as f:
                json.dump(config, f, indent=4)

        config = og.Config(os.path.dirname(config_file))
        if input_ids.is_cuda:
            config.clear_providers()
            config.append_provider("cuda")
        session = og.Model(config)
    else:
        session = model_or_path

    params = og.GeneratorParams(session)
    params.set_search_options(
        max_length=max_new_tokens + input_ids.shape[1], batch_size=input_ids.shape[0]
    )
    generator = og.Generator(session, params)

    # First call: prefill
    cats = []
    generator.append_tokens(input_ids)
    while not generator.is_done():
        generator.generate_next_token()
        new_token = generator.get_next_tokens()[0]
        cats.append(int(new_token))

    input_ids = torch.cat([input_ids, torch.tensor([cats], dtype=torch.int64)], dim=-1)
    if return_session:
        return input_ids, session
    return input_ids
