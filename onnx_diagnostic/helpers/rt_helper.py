import json
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import onnx
import torch
from .helper import string_type, flatten_object, max_diff
from .torch_helper import torch_deepcopy
from .ort_session import InferenceSessionForTorch


def name_type_to_onnx_dtype(name: str) -> int:
    assert name.startswith("tensor(") and name.endswith(")"), f"Invalid value name={name!r}"
    look = name[7:-1]
    return getattr(onnx.TensorProto, look.upper())


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
    if is_modelbuilder and isinstance(inputs, dict) and "positions_ids" in inputs:
        position_ids = input["position_ids"]
        assert (
            (position_ids == torch.tensor(list(range(position_ids.shape[-1]))).unsqueeze(0))
            .max()
            .item()
        ), f"ModelBuilder does not support position_ids={position_ids}"
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
    eos_token_id: int = 2,
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
    eos_token_id: int = 2,
    max_new_tokens=100,
    return_session: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, InferenceSessionForTorch, Dict[str, Any]]]:
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
        created if necessary, the function returns the feeds for the next iteration
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
    has_position_ids = "position_ids" in session.input_names

    assert (
        len(input_names) > 2
        and input_names[:2] == ["input_ids", "attention_mask"]
        and input_names[3 if has_position_ids else 2].startswith("past_key_values")
    ), (
        f"Only text generation is supported but input_names == {input_names}, "
        f"has_position_ids={has_position_ids}"
    )
    assert (
        not has_position_ids or input_names[2] == "position_ids"
    ), f"position_ids must the third input but input_names={input_names}"

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
    if has_position_ids:
        feeds["position_ids"] = torch.unsqueeze(
            torch.arange(input_ids.shape[1], dtype=torch.int64, device=input_ids.device), 0
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
        if has_position_ids:
            feeds["position_ids"] = torch.unsqueeze(
                torch.arange(
                    input_ids.shape[1],
                    input_ids.shape[1] + 1,
                    dtype=torch.int64,
                    device=input_ids.device,
                ),
                0,
            )
        feeds.update(dict(zip(input_names[3 if has_position_ids else 2 :], outputs[1:])))
        outputs = session.run(None, feeds)

    if return_session:
        return input_ids, session, feeds
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


_mapping_types = {
    "float": "F",
    "double": "D",
    "float16": "H",
    "uint8": "U8",
    "uint16": "U16",
    "uint32": "U32",
    "uint64": "U64",
    "int8": "I8",
    "int16": "I16",
    "int32": "I32",
    "int64": "I64",
}


def _process_shape(shape_df):
    if isinstance(shape_df, float) or len(shape_df) == 0:
        return ""
    values = []
    for val in shape_df:
        if len(val) != 1:
            raise ValueError(f"Unable to process shape {val!r} from {values!r}.")
        for _k, _v in val.items():
            k, v = _k, _v
            break
        if v:
            vs = "x".join(map(str, v))
            values.append(f"{_mapping_types.get(k,k)}[{vs}]")
        else:
            values.append(f"{_mapping_types.get(k,k)}")
    return "+".join(values)


def post_process_df_profile(
    df: "pandas.DataFrame",  # noqa: F821
    first_it_out: bool = False,
    agg: bool = False,
    agg_op_name: bool = True,
    with_shape: bool = False,
) -> "pandas.DataFrame":  # noqa: F821
    """
    Post-processed a dataframe obtained after profiling onnxruntime.
    It adds a column for a more explicit event name and adds
    a column for the iteration number

    :param agg: aggregate the result
    :param first_it_out: leave the first iteration
        out of the aggregation
    :param agg_op_name: aggregate on operator name or operator index
    :param with_shape: keep the shape to aggregate
    :return: DataFrame
    """
    events = {"kernel_time", "fence_after", "fence_before"}

    def sep_event(s):
        for e in events:
            if s.endswith(e):
                return e
        return s

    df = df.copy()
    df["event_name"] = df["name"].apply(sep_event)
    df["iteration"] = -1
    current = -1
    for i in range(df.shape[0]):
        if df.loc[i, "name"] == "SequentialExecutor::Execute":
            current += 1
        df.loc[i, "iteration"] = current

    if not agg:
        if with_shape:
            df["args_input_type_shape"] = df["args_input_type_shape"].apply(_process_shape)
            df["args_output_type_shape"] = df["args_output_type_shape"].apply(_process_shape)
        else:
            df = df.drop(["args_input_type_shape", "args_output_type_shape"], axis=1)
        if first_it_out:
            df["it==0"] = (df["iteration"] <= 0).astype(int)
        return df

    agg_cols = ["cat", "args_node_index", "args_op_name", "args_provider", "event_name"]
    if with_shape:
        agg_cols.append("args_input_type_shape")
        df["args_input_type_shape"] = df["args_input_type_shape"].apply(_process_shape)
        df["args_output_type_shape"] = df["args_output_type_shape"].apply(_process_shape)
    else:
        df = df.drop(["args_input_type_shape", "args_output_type_shape"], axis=1)

    if first_it_out:
        df["it==0"] = (df["iteration"] <= 0).astype(int)
        agg_cols.insert(0, "it==0")
    if agg_op_name:
        del agg_cols[agg_cols.index("args_node_index")]
    for c in agg_cols:
        df[c] = df[c].fillna("")
    df["dur"] = df["dur"].fillna(0)
    agg = df[[*agg_cols, "dur"]].groupby(agg_cols).sum()
    return agg


def js_profile_to_dataframe(
    filename: str,
    as_df: bool = True,
    first_it_out: bool = False,
    agg: bool = False,
    agg_op_name: bool = False,
    with_shape: bool = False,
) -> Union[List, "pandas.DataFrame"]:  # noqa: F821
    """
    Profiles the execution of an onnx graph with onnxruntime.

    :param filename: filename holding the profiling stored in json format
    :param as_df: returns the
    :param first_it_out: if aggregated, leaves the first iteration out
    :param agg: aggregate by event
    :param agg_op_name: aggregate on operator name or operator index
    :param with_shape: keep the shape before aggregating
    :return: DataFrame or dictionary
    """
    with open(filename, "r") as f:
        content = f.read()
    js = json.loads(content)

    suffixes = ["_kernel_time", "_fence_before", "_fence_after"]
    rows = []
    for row in js:
        if "args" in row and isinstance(row["args"], dict):
            for k, v in row["args"].items():
                row[f"args_{k}"] = v
            del row["args"]
        name = row["name"]
        for suf in suffixes:
            if name.endswith(suf):
                changed = name[: -len(suf)]
                row["op_name"] = changed
                break
        rows.append(row)
    if as_df:
        import pandas

        return post_process_df_profile(
            pandas.DataFrame(rows),
            first_it_out=first_it_out,
            agg=agg,
            agg_op_name=agg_op_name,
            with_shape=with_shape,
        )
    return rows


def _preprocess_graph1(df):
    df = df.copy()
    df["args_provider"] = df["args_provider"].apply(
        lambda s: s.replace("ExecutionProvider", "") if isinstance(s, str) else s
    )
    agg_cols = ["dur", "args_op_name", "args_provider"]
    for c in ["it==0", "args_input_type_shape"]:
        if c in df.columns:
            agg_cols.append(c)
    if "it==0" in df.columns:
        vs = ["t>=1", "t=0"]
        df["it==0"] = df["it==0"].apply(lambda v: vs[v])
    gr_dur = df[agg_cols].groupby(agg_cols[1:]).sum().sort_values("dur")
    gr_n = df[agg_cols].groupby(agg_cols[1:]).count()
    gr_n = gr_n.loc[gr_dur.index, :]
    gr_n.columns = ["count"]
    gr = gr_dur.merge(gr_n, left_index=True, right_index=True, how="outer")
    gr["ratio"] = gr["dur"] / gr["dur"].sum()
    return gr_dur, gr_n, gr


def _preprocess_graph2(df):
    df = df.reset_index(drop=False).copy()
    df["args_node_index"] = df["args_node_index"].apply(
        lambda i: int(i) if i not in {None, ""} else -1
    )
    df["args_provider"] = df["args_provider"].apply(
        lambda s: s.replace("ExecutionProvider", "") if isinstance(s, str) else s
    )
    df = df[(df["cat"] == "Node") & (df["event_name"] == "kernel_time")]
    agg_cols = ["dur", "args_node_index", "args_op_name", "args_provider"]
    for c in ["it==0", "args_input_type_shape"]:
        if c in df.columns:
            agg_cols.append(c)
    if "it==0" in df.columns:
        vs = ["t>=1", "t=0"]
        df["it==0"] = df["it==0"].apply(lambda v: vs[v])
    df = df[agg_cols].groupby(agg_cols[1:]).sum()
    df = df.sort_index(ascending=False)
    df["ratio"] = df["dur"] / df["dur"].sum()
    return df


def plot_ort_profile(
    df: "pandas.DataFrame",  # noqa: F821
    ax0: Optional["matplotlib.axes.Axes"] = None,  # noqa: F821
    ax1: Optional["matplotlib.axes.Axes"] = None,  # noqa: F821
    title: Optional[str] = None,
) -> "matplotlib.axes.Axes":  # noqa: F821
    """
    Plots time spend in computation based on a dataframe
    produced by function :func:`js_profile_to_dataframe`.

    :param df: dataframe
    :param ax0: first axis to draw time
    :param ax1: second axis to draw occurrences
    :param title: graph title
    :return: the graph

    .. plot::
        :include-source:

        import numpy as np
        from onnx import TensorProto
        import onnx.helper as oh
        from onnx.checker import check_model
        from onnx.numpy_helper import from_array
        import matplotlib.pyplot as plt
        from onnxruntime import InferenceSession, SessionOptions
        from onnx_diagnostic.helpers.rt_helper import js_profile_to_dataframe, plot_ort_profile


        def get_model():
            model_def0 = oh.make_model(
                oh.make_graph(
                    [
                        oh.make_node("Add", ["X", "init1"], ["X1"]),
                        oh.make_node("Abs", ["X"], ["X2"]),
                        oh.make_node("Add", ["X", "init3"], ["inter"]),
                        oh.make_node("Mul", ["X1", "inter"], ["Xm"]),
                        oh.make_node("Sub", ["X2", "Xm"], ["final"]),
                    ],
                    "test",
                    [oh.make_tensor_value_info("X", TensorProto.FLOAT, [None])],
                    [oh.make_tensor_value_info("final", TensorProto.FLOAT, [None])],
                    [
                        from_array(np.array([1], dtype=np.float32), name="init1"),
                        from_array(np.array([3], dtype=np.float32), name="init3"),
                    ],
                ),
                opset_imports=[oh.make_opsetid("", 18)],
                ir_version=9,
            )
            check_model(model_def0)
            return model_def0


        sess_options = SessionOptions()
        sess_options.enable_profiling = True
        sess = InferenceSession(
            get_model().SerializeToString(), sess_options, providers=["CPUExecutionProvider"]
        )
        for _ in range(11):
            sess.run(None, dict(X=np.arange(10).astype(np.float32)))
        prof = sess.end_profiling()

        df = js_profile_to_dataframe(prof, first_it_out=True)
        print(df.head())

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        plot_ort_profile(df, ax[0], ax[1], "test_title")
        fig.tight_layout()

    With ``agg=True``:

    .. plot::
        :include-source:

        import numpy as np
        from onnx import TensorProto
        import onnx.helper as oh
        from onnx.checker import check_model
        from onnx.numpy_helper import from_array
        import matplotlib.pyplot as plt
        from onnxruntime import InferenceSession, SessionOptions
        from onnx_diagnostic.helpers.rt_helper import js_profile_to_dataframe, plot_ort_profile


        def get_model():
            model_def0 = oh.make_model(
                oh.make_graph(
                    [
                        oh.make_node("Add", ["X", "init1"], ["X1"]),
                        oh.make_node("Abs", ["X"], ["X2"]),
                        oh.make_node("Add", ["X", "init3"], ["inter"]),
                        oh.make_node("Mul", ["X1", "inter"], ["Xm"]),
                        oh.make_node("Sub", ["X2", "Xm"], ["final"]),
                    ],
                    "test",
                    [oh.make_tensor_value_info("X", TensorProto.FLOAT, [None])],
                    [oh.make_tensor_value_info("final", TensorProto.FLOAT, [None])],
                    [
                        from_array(np.array([1], dtype=np.float32), name="init1"),
                        from_array(np.array([3], dtype=np.float32), name="init3"),
                    ],
                ),
                opset_imports=[oh.make_opsetid("", 18)],
                ir_version=9,
            )
            check_model(model_def0)
            return model_def0


        sess_options = SessionOptions()
        sess_options.enable_profiling = True
        sess = InferenceSession(
            get_model().SerializeToString(), sess_options, providers=["CPUExecutionProvider"]
        )
        for _ in range(11):
            sess.run(None, dict(X=np.arange(10).astype(np.float32)))
        prof = sess.end_profiling()

        df = js_profile_to_dataframe(prof, first_it_out=True, agg=True)
        print(df.head())

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        plot_ort_profile(df, ax[0], ax[1], "test_title")
        fig.tight_layout()
    """
    fontsize = 10
    if ax0 is None:
        import matplotlib.pyplot as plt

        ax0 = plt.gca()

    if "args_provider" in df.columns:
        # Aggregation by operator
        gr_dur, gr_n, _ = _preprocess_graph1(df)
        gr_dur.plot.barh(ax=ax0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ax0.set_xticklabels(ax0.get_xticklabels(), fontsize=fontsize)
            ax0.get_yaxis().set_label_text("")
            ax0.set_yticklabels(
                ax0.get_yticklabels(), rotation=45, ha="right", fontsize=fontsize
            )
        if title is not None:
            ax0.set_title(title)
        if ax1 is not None:
            gr_n.plot.barh(ax=ax1)
            ax1.set_title("n occurrences")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ax1.set_xticklabels(ax1.get_xticklabels(), fontsize=fontsize)
                ax1.get_yaxis().set_label_text("")
                ax1.set_yticklabels(
                    ax1.get_yticklabels(), rotation=45, ha="right", fontsize=fontsize
                )
        return ax0

    df = _preprocess_graph2(df)
    df[["dur"]].plot.barh(ax=ax0)
    if title is not None:
        ax0.set_title(title)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ax0.set_xticklabels(ax0.get_xticklabels(), fontsize=fontsize)
        ax0.get_yaxis().set_label_text("")
        ax0.set_yticklabels(ax0.get_yticklabels(), fontsize=fontsize)
    return ax0


def plot_ort_profile_timeline(
    df: "pandas.DataFrame",  # noqa: F821
    ax: Optional["matplotlib.axes.Axes"] = None,  # noqa: F821
    iteration: int = -2,
    title: Optional[str] = None,
    quantile: float = 0.5,
    fontsize: int = 12,
) -> "matplotlib.axes.Axes":  # noqa: F821
    """
    Creates a timeline based on a dataframe
    produced by function :func:`js_profile_to_dataframe`.

    :param df: dataframe
    :param ax: first axis to draw time
    :param iteration: iteration to plot, negative value to start from the end
    :param title: graph title
    :param quantile: draw the 10% less consuming operators in a different color
    :param fontsize: font size
    :return: the graph

    .. plot::
        :include-source:

        import numpy as np
        from onnx import TensorProto
        import onnx.helper as oh
        from onnx.checker import check_model
        from onnx.numpy_helper import from_array
        import matplotlib.pyplot as plt
        from onnxruntime import InferenceSession, SessionOptions
        from onnx_diagnostic.helpers.rt_helper import (
            js_profile_to_dataframe,
            plot_ort_profile_timeline,
        )


        def get_model():
            model_def0 = oh.make_model(
                oh.make_graph(
                    [
                        oh.make_node("Add", ["X", "init1"], ["X1"]),
                        oh.make_node("Abs", ["X"], ["X2"]),
                        oh.make_node("Add", ["X", "init3"], ["inter"]),
                        oh.make_node("Mul", ["X1", "inter"], ["Xm"]),
                        oh.make_node("Sub", ["X2", "Xm"], ["final"]),
                    ],
                    "test",
                    [oh.make_tensor_value_info("X", TensorProto.FLOAT, [None])],
                    [oh.make_tensor_value_info("final", TensorProto.FLOAT, [None])],
                    [
                        from_array(np.array([1], dtype=np.float32), name="init1"),
                        from_array(np.array([3], dtype=np.float32), name="init3"),
                    ],
                ),
                opset_imports=[oh.make_opsetid("", 18)],
                ir_version=9,
            )
            check_model(model_def0)
            return model_def0


        sess_options = SessionOptions()
        sess_options.enable_profiling = True
        sess = InferenceSession(
            get_model().SerializeToString(), sess_options, providers=["CPUExecutionProvider"]
        )
        for _ in range(11):
            sess.run(None, dict(X=np.arange(10).astype(np.float32)))
        prof = sess.end_profiling()

        df = js_profile_to_dataframe(prof, first_it_out=True)
        print(df.head())

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        plot_ort_profile_timeline(df, ax, title="test_timeline", quantile=0.5)
        fig.tight_layout()
    """
    if ax is None:
        import matplotlib.pyplot as plt

        ax = plt.gca()

    df = df.copy()
    df["iteration"] = df["iteration"].astype(int)
    iterations = set(df["iteration"])
    n_iter = iteration if iteration >= 0 else max(iterations) + 1 + iteration
    dfi = df[df["iteration"] == n_iter]
    assert dfi.shape[0] > 0, f"Iteration {iteration} cannot be found in {iterations}."

    if "fence_before" in set(dfi["event_name"]):
        started = {}
        data = []
        for irow in dfi.iterrows():
            assert isinstance(irow, tuple), f"pandas has changed its api, type is {type(irow)}"
            assert len(irow) == 2, f"pandas has changed its api, row is {irow}"
            row = irow[1]
            it = row["iteration"]
            op_type = row["args_op_name"]
            op_name = row["op_name"]
            event_name = row["event_name"]
            provider = row["args_provider"]
            ts = float(row["ts"])
            dur = float(row["dur"])
            if event_name == "fence_before":
                started[op_type, op_name, it] = dict(
                    op_name=op_name, op_type=op_type, begin=ts
                )
            elif event_name == "kernel_time":
                obs = started[op_type, op_name, it]
                obs["duration"] = dur
                obs["begin_kernel"] = ts
                obs["provider"] = provider
            elif event_name == "fence_after":
                obs = started[op_type, op_name, it]
                obs["end"] = ts
                data.append(obs)
                del started[op_type, op_name, it]
            else:
                assert event_name in {
                    "SequentialExecutor::Execute",
                    "model_run",
                }, f"Unexpected event_name={event_name!r}, row={row}"
    else:
        # New format
        data = []
        for irow in dfi.iterrows():
            row = irow[1]
            if row["event_name"] != "kernel_time":
                continue
            obs = dict(
                duration=float(row["dur"]),
                op_name=row["op_name"],
                op_type=row["args_op_name"],
                provider=row["args_provider"],
                begin=float(row["ts"]),
                end=float(row["ts"]) + float(row["dur"]),
                begin_kernel=float(row["ts"]),
            )
            data.append(obs)

    # durations
    data_dur = list(sorted(d["duration"] for d in data))
    threshold = data_dur[int(quantile * len(data_dur))]
    origin = dfi["ts"].min()

    colors = ["blue", "green", "red", "orange"]

    import matplotlib.patches as mpatches

    cs = [0, 0]
    for i, obs in enumerate(data):
        dur = obs["duration"]
        cat = int(dur >= threshold)

        # color
        color = colors[cat * 2 + cs[cat] % 2]
        cs[cat] += 1

        # rectangle
        t1 = obs["begin"] - origin
        t2 = obs["end"] - origin
        shape = mpatches.Rectangle((0, t1), 1, t2 - t1, ec="none", color=color)
        ax.add_artist(shape)
        tk1 = obs["begin_kernel"] - origin
        tk2 = (obs["begin_kernel"] + obs["duration"]) - origin
        ax.plot([0, 1], [tk1, tk1], "b--")
        ax.plot([0, 1], [tk2, tk2], "b--")
        if i == 0:
            ax.plot([0, 2], [tk1, tk1], "b")
        elif i == len(data) - 1:
            ax.plot([0, 2], [tk2, tk2], "b")

        # text
        y = (tk1 + tk2) / 2
        text = obs["op_type"]
        prov = obs["provider"].replace("ExecutionProvider", "")
        name = obs["op_name"]
        if len(name) >= 10:
            name = name[:5] + "..." + name[5:]
        ax.text(1, y, f"{i}:{prov}:{text}-{name}", fontsize=fontsize, va="center")

    ax.invert_yaxis()
    return ax
