import inspect
import time
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Sequence, Tuple, Union
import onnx
import onnx.helper as oh
import numpy as np
import torch
from ..helpers import string_type, string_diff, max_diff, flatten_object
from ..helpers.onnx_helper import pretty_onnx
from ..helpers.torch_helper import (
    to_numpy,
    from_numpy,
    to_tensor,
    torch_dtype_to_onnx_dtype,
)
from ..helpers.torch_fx_graph_helper import prepare_args_kwargs, run_fx_node
from ..reference.ort_evaluator import OnnxList, OnnxruntimeEvaluator
from .sbs_dataclasses import (
    ReplayConfiguration,
    RunAlignedRecord,
    StatusRunAligned,
    make_torch_inputs,
)


def _check_tensor_(use_tensor, name, obj, flip_type=False):
    if flip_type:
        if use_tensor:
            if isinstance(obj, np.ndarray):
                obj = from_numpy(obj)
        else:
            if isinstance(obj, torch.Tensor):
                obj = to_numpy(obj)

    assert not use_tensor or isinstance(obj, (torch.Tensor, OnnxList)), (
        f"Unexpected type {type(obj)} for {name!r}. "
        f"use_tensor is True so torch.Tensor is expected."
    )
    assert use_tensor or isinstance(obj, (np.ndarray, OnnxList)), (
        f"Unexpected type {type(obj)} for {name!r}. "
        f"use_tensor is False so np.array is expected."
    )
    return obj


def _make_node_from_initializer(proto: onnx.TensorProto) -> onnx.NodeProto:
    return oh.make_node("Constant", [], [proto.name], value=proto)


def _loop_cmp(
    mapping_onnx_to_torch: Dict[str, str],
    torch_results: Dict[str, torch.Tensor],
    onnx_results: Dict[str, Any],
    onnx_name: str,
    onnx_result: torch.Tensor,
    second_onnx_result: torch.Tensor,
    verbose: int,
    atol: Optional[float],
    rtol: Optional[float],
    i_torch: int,
    i_onnx: int,
    str_kws: Dict[str, bool],
    exc: bool,
    use_tensor: bool,
) -> Optional[RunAlignedRecord]:
    onnx_results[onnx_name] = _check_tensor_(use_tensor, onnx_name, onnx_result)
    if verbose > 1:
        print(f"[run_aligned-nx] +res: {onnx_name}={string_type(onnx_result, **str_kws)}")

    to = mapping_onnx_to_torch.get(onnx_name, onnx_name)
    if to in torch_results:
        d = max_diff(torch_results[to], onnx_result, hist=[0.1, 0.01])
        if verbose > 1:
            if onnx_name == to:
                print(f"[run_aligned-==] cmp {to}: {string_diff(d)}")
            else:
                print(f"[run_aligned-~~] cmd {to}/{onnx_name}: {string_diff(d)}")
            if not (atol is None or rtol is None or (d["abs"] <= atol and d["rel"] <= rtol)):
                if exc:
                    raise ValueError(
                        f"discrepancies detected for results [{to}/{onnx_name}]: "
                        f"{string_diff(d)}"
                        f"\n-- onnx_result: {string_type(onnx_result[to], **str_kws)}"
                        f"\n-- onnx_results: {string_type(onnx_result, **str_kws)}"
                        f"\n-- torch\n{onnx_result[to]}"
                    )
                else:
                    print(
                        f"[run_align-dx] discrepancies {string_diff(d)} - [{to}/{onnx_name}]"
                    )
        r = RunAlignedRecord(
            ep_id_node=i_torch,
            onnx_id_node=i_onnx,
            ep_name=to,
            onnx_name=onnx_name,
            ep_shape_type=string_type(torch_results[to], **str_kws),
            onnx_shape_type=string_type(onnx_result, **str_kws),
        )
        r.set_diff(d)
        if second_onnx_result is not None:
            d2 = max_diff(torch_results[to], second_onnx_result, hist=[0.1, 0.01])
            r.set_diff2(d2)
        mapping_onnx_to_torch[onnx_name] = to
        return r
    return None


def _duplicated_values(d):
    rev = {}
    for k, v in d.items():
        if v in rev:
            rev[v].append(k)
        else:
            rev[v] = [k]
    res = {k: v for k, v in rev.items() if len(v) > 1}
    final = set()
    for v in res.values():
        final |= set(v)
    return final


def _validation_nn_functional(
    node: onnx.NodeProto, new_feeds: Dict[str, torch.Tensor], expected: List[torch.Tensor]
) -> Optional[str]:
    if node.op_type == "Gemm" and len(node.input) == 3:
        atts = {}
        for att in node.attribute:
            if att.name in ("alpha", "beta"):
                atts[att.name] = att.f
            elif att.name in ("transA", "transB"):
                atts[att.name] = att.i
        if atts == {"transB": 1}:
            res = torch.nn.functional.linear(*[new_feeds[i] for i in node.input])
            diff = max_diff(res, expected[0])
            return f"function.linear:{string_diff(diff)}"
    return None


def _loop_onnx_node(
    onx: onnx.ModelProto,
    ep_graph_nodes: List[torch.fx.Node],
    onnx_results: Dict[str, Any],
    mapping_onnx_to_torch: Dict[str, str],
    torch_results: Dict[str, torch.Tensor],
    ep_durations,
    use_tensor: bool,
    i_torch: int,
    i_onnx: int,
    name_to_ep_node: Dict[str, int],
    run_cls_kwargs: Dict[str, Any],
    str_kws: Dict[str, bool],
    status: StatusRunAligned,
    already_run_onnx: Set[int],
    torch_names_to_onnx_names: Dict[str, str],
    verbose: int,
    exc: bool,
    atol: float,
    rtol: float,
    reset_names: Set[str],
    replay_configuration: Optional[ReplayConfiguration],
    has_cuda: bool,
    run_cls: type,
    loop: Any,
    run_onnx_with_torch_inputs: bool,
) -> Iterator[Optional[RunAlignedRecord]]:

    if i_onnx in already_run_onnx:
        yield None
    node = onx.graph.node[i_onnx]
    if verbose > 1:
        print(
            f"[run_aligned] run onx.graph.node[{i_onnx}]: "
            f"{node.op_type}({', '.join(node.input)}) -> {', '.join(node.output)}"
        )
    elif verbose == 1:
        loop.set_description(
            f"ep {i_torch}/{len(ep_graph_nodes)} nx {i_onnx}/{len(onx.graph.node)} "
            f"{status.to_str()}"
        )
        loop.update(min(1, 1 + i_torch + i_onnx))

    ref = run_cls(node, **run_cls_kwargs)
    # We need to clone because the runtime maybe using dlpack to create OrtValue
    hidden_inputs = OnnxruntimeEvaluator._get_hidden_node_inputs(node)
    all_inputs = [*node.input, *hidden_inputs] if hidden_inputs else node.input
    feeds = (
        {k: onnx_results[k].clone() for k in all_inputs if k}
        if use_tensor
        else {k: onnx_results[k].copy() for k in all_inputs if k}
    )
    assert "" not in feeds, f"Unexpected feeds={string_type(feeds, **str_kws)}"
    if verbose > 1:
        print(f"[run_aligned] feeds={string_type(feeds, **str_kws)}")
    begin = time.perf_counter()
    try:
        res = ref.run(None, feeds)  # type: ignore[attr-defined]
    except Exception as e:
        raise RuntimeError(
            f"Unable to run node {node.op_type}, domain={node.domain} "
            f"with inputs={node.input}, feeds={string_type(feeds, **str_kws)}"
        ) from e
    duration = time.perf_counter() - begin
    if verbose > 1:
        print(f"[run_aligned] res={string_type(res, **str_kws)}")
    assert (
        not has_cuda
        or not any(t is not None and t.is_cuda for t in feeds.values())
        or any(t is not None and t.is_cuda for t in res)
        or node.op_type in {"Shape", "Size"}  # on CPU no matter what
        or node.op_type
        in {
            "Add",
            "Concat",
            "Div",
            "Gather",
            "Mul",
            "Range",
            "Squeeze",
            "Sub",
            "Unsqueeze",
        }  # not sure, could be about shapes
    ), (
        f"One input is on cuda but there is no float output on cuda, "
        f"feeds={string_type(feeds, with_device=True, with_shape=True)}, "
        f"res={string_type(res, with_device=True, with_shape=True)}, "
        f"node is {pretty_onnx(node)}"
    )

    comment = None
    cross = None
    if run_onnx_with_torch_inputs:
        # Let's run the operator with torch results if they are available
        new_feeds, removed = make_torch_inputs(
            node.input,
            {
                **{v: k for k, v in torch_names_to_onnx_names.items()},
                **mapping_onnx_to_torch,
            },
            onnx_results,
            torch_results,
            submodel=None,
        )
        if not removed:
            if verbose > 1:
                print(
                    f"[run_aligned] feeds for second run="
                    f"{string_type(new_feeds, **str_kws)}"
                )
            cross = ref.run(None, new_feeds)
            if verbose > 1:
                print(f"[run_aligned] got for second run={string_type(cross, **str_kws)}")
            # Gemm = torch.nn.function.linear, in that case, we just run it as well
            to = mapping_onnx_to_torch.get(node.output[0], node.output[0])
            if to in torch_results:
                comment = _validation_nn_functional(node, new_feeds, [torch_results[to]])
        elif verbose > 1:
            print(f"[run_aligned] second run not possible because of missing {removed}")

    if cross is None:
        cross = [None for _ in res]

    list_node_output = list(node.output)
    node_output = [o for o in list_node_output if o]
    for o, r, r2 in zip(node_output, res, cross):
        if r is None or not o:
            continue
        tmp = _loop_cmp(
            mapping_onnx_to_torch,
            torch_results,
            onnx_results,
            o,
            r,
            r2,
            verbose,
            atol,
            rtol,
            i_torch,
            i_onnx,
            str_kws,
            exc,
            use_tensor,
        )
        if tmp is not None:
            if tmp.ep_name in name_to_ep_node:
                tmp.ep_id_node = name_to_ep_node[tmp.ep_name]
                tmp.ep_target = str(ep_graph_nodes[tmp.ep_id_node].target)
                tmp.ep_time_run = ep_durations[tmp.ep_id_node]
            else:
                tmp.ep_id_node = None
                tmp.ep_target = None
                tmp.ep_name = None
            tmp.onnx_op_type = onx.graph.node[tmp.onnx_id_node].op_type
            tmp.onnx_id_output = list_node_output.index(o)
            tmp.onnx_time_run = duration
            status.yielded_nodes += 1
            if tmp.err_abs is not None:
                status.update(tmp.err_abs)
            tmp.comment = comment
            yield tmp

            # do we need to dump pieces if graph the user can replay?
            if replay_configuration:
                if replay_configuration.select(
                    name=tmp.onnx_name, op_type=tmp.onnx_op_type, err_abs=tmp.err_abs
                ):
                    replay_configuration.dump(
                        name=tmp.onnx_name,
                        onnx_id_node=tmp.onnx_id_node,
                        model=onx,
                        onnx_results=onnx_results,
                        torch_results=torch_results,
                        onnx_name_to_ep_name={
                            **{v: k for k, v in torch_names_to_onnx_names.items()},
                            **mapping_onnx_to_torch,
                        },
                        verbose=max(verbose - 1, 0),
                    )
                    status.last_replay = tmp.onnx_name

            # reset_names: replaces onnx_results by torch_results to see
            # if that fixes the discrepancies problem
            if reset_names and tmp.ep_name in reset_names:
                assert (
                    tmp.ep_name in torch_results
                ), f"name {tmp.ep_name!r} set to be reset is missing in torch_results."
                assert (
                    tmp.onnx_name in onnx_results
                ), f"name {tmp.onnx_name!r} set to be reset is missing in onnx_results."
                onnx_results[tmp.onnx_name] = torch_results[tmp.ep_name]
                tmp = _loop_cmp(
                    mapping_onnx_to_torch,
                    torch_results,
                    onnx_results,
                    o,
                    torch_results[tmp.ep_name],
                    None,
                    verbose,
                    atol,
                    rtol,
                    i_torch,
                    i_onnx,
                    str_kws,
                    exc,
                    use_tensor,
                )
                assert tmp.err_abs == 0, f"Reset did not happen, tmp={tmp}"
                if tmp is not None:
                    tmp.onnx_op_type = "reset"
                    tmp.onnx_id_output = list_node_output.index(o)
                    status.yielded_nodes += 1
                    yield tmp
    already_run_onnx.add(i_onnx)


def _preparation_with_fx_graph(
    ep_graph_nodes: List[torch.fx.Node],
    name_to_ep_node: Dict[str, int],
    torch_input_names: List[str],
    onx: onnx.ModelProto,
    torch_names_to_onnx_names: Dict[str, str],
    skip_mapping_torch_onnx,
    torch_results: Dict[str, torch.Tensor],
    placeholders: Dict[str, torch.Tensor],
    placeholders_to_state_dict: Dict[str, str],
    ep_state_dict: Dict[str, torch.Tensor],
    positions: Dict[str, Dict[str, int]],
) -> List[str]:
    torch_output_names = None
    for i, node in enumerate(ep_graph_nodes):
        if isinstance(node.name, str):
            positions[node.name] = dict(fx=i)
        else:
            for n in node.name:
                positions[n] = dict(fx=i)
        if node.op == "placeholder":
            if node.name in placeholders_to_state_dict:
                # This a weight.
                placeholders[node.name] = ep_state_dict[placeholders_to_state_dict[node.name]]
                torch_results[node.name] = placeholders[node.name]
                assert isinstance(torch_results[node.name], torch.Tensor), (
                    f"torch_results[{node.name}] not a tensor but "
                    f"{type(torch_results[node.name])}"
                )
            else:
                # This is an input
                assert len(torch_input_names) < len(onx.graph.input), (
                    f"torch_input_names={torch_input_names!r}, "
                    f"onnx_input_names={[n.name for n in onx.graph.input]}, "
                    f"node.name={node.name!r} cannot be an input, "
                    f"placeholders_to_state_dict={sorted(placeholders_to_state_dict)}"
                )
                assert node.name not in skip_mapping_torch_onnx, (
                    f"{node.name!r} is ambiguous, cannot be mapped due to "
                    f"{skip_mapping_torch_onnx}"
                )
                torch_names_to_onnx_names[node.name] = onx.graph.input[
                    len(torch_input_names)
                ].name
                torch_input_names.append(node.name)
        elif node.op == "output":
            torch_output_names = [n.name for n in node.args[0]]
        assert isinstance(node.name, str), (
            f"Unexpected type {type(node.name)} for node={node} (target={node.target}), "
            f"args={node.args}"
        )
        name_to_ep_node[node.name] = i
    assert torch_output_names is not None, "No output node ws found the graph."
    return torch_output_names


def _preparation_with_onnx_model(
    default_device,
    use_tensor: bool,
    onx: onnx.ModelProto,
    already_yielded: Dict[str, Any],
    str_kws: Dict[str, bool],
    positions: Dict[str, Dict[str, int]],
    torch_names_to_onnx_names: Dict[str, str],
    torch_output_names: List[str],
    torch_results: Dict[str, torch.Tensor],
    skip_mapping_torch_onnx: Set[str],
    verbose: int,
    args: Sequence[Any],
    kwargs: Dict[str, Any],
) -> Tuple[Dict[str, str], Dict[str, torch.Tensor], float, float, List[RunAlignedRecord]]:
    for inp in onx.graph.input:
        n = inp.name
        if n in positions:
            positions[n]["onnx"] = -1
        else:
            positions[n] = dict(onnx=-1)
    for inp in onx.graph.initializer:
        n = inp.name
        if n in positions:
            positions[n]["onnx"] = -1
        else:
            positions[n] = dict(onnx=-1)
    for i, node in enumerate(onx.graph.node):
        for n in node.output:
            if n in positions:
                positions[n]["onnx"] = i
            else:
                positions[n] = dict(onnx=i)

    onnx_outputs_names = [o.name for o in onx.graph.output]
    assert torch_output_names is not None and len(torch_output_names) == len(
        onnx_outputs_names
    ), (
        f"Unexpected number of outputs, torch_output_names={torch_output_names}, "
        f"onnx_outputs_names={onnx_outputs_names}"
    )
    mapping_onnx_to_torch = dict(zip(onnx_outputs_names, torch_output_names))

    onnx_args = list(args) if args else []
    if kwargs:
        onnx_args.extend(flatten_object(kwargs, drop_keys=True))
    if verbose:
        print(f"[run_aligned]   args: {string_type(args, **str_kws)}")
        print(f"[run_aligned] kwargs: {string_type(kwargs, **str_kws)}")
        print(f"[run_aligned]   onnx: {string_type(onnx_args, **str_kws)}")
        print(f"[run_aligned] nx: walks through {len(onx.graph.input)} onnx inputs")
    onnx_results: Dict[str, Any] = {}
    for inp, v in zip(onx.graph.input, onnx_args):
        onnx_results[inp.name] = _check_tensor_(
            use_tensor, inp.name, v if use_tensor else to_numpy(v)
        )
        if verbose:
            print(f"[run_aligned-nx] +inp: {inp.name}: {string_type(v, **str_kws)}")

    # alias for initializers
    skip_onnx_name = set()
    init_aliases: Dict[str, str] = {}
    for init in onx.graph.initializer:
        new_names = {
            n
            for n in [
                f"p_{init.name.replace('.', '_')}",
                f"p_{init.name.split('::')[0].split('--')[-1].replace('.', '_')}",
                f"{init.name.split('::')[0].split('--')[-1].replace('.', '_')}",
            ]
            if n != init.name
        }
        drop = False
        for new_name in new_names:
            if new_name in skip_onnx_name:
                drop = True
                break
        if drop:
            skip_onnx_name |= new_names | {init.name}
            for new_name in new_names:
                if new_names in init_aliases:
                    del init_aliases[new_name]
        else:
            for new_name in new_names:
                init_aliases[new_name] = init.name
    rev_init_aliases: Dict[str, Set[str]] = {}
    for k, v in init_aliases.items():
        if v in rev_init_aliases:
            rev_init_aliases[v].add(k)
        else:
            rev_init_aliases[v] = {k}

    # initializers
    if verbose:
        print(f"[run_aligned] nx: handles {len(onx.graph.initializer)} initializers from onnx")
    memory_cpu = 0
    memory_cuda = 0
    records_to_yield = []
    for init in onx.graph.initializer:  # type: ignore
        t = None
        if init.name in torch_results:
            if init.name not in skip_mapping_torch_onnx:
                t = torch_results[init.name]
                torch_names_to_onnx_names[init.name] = init.name
        elif init.name not in skip_onnx_name and init.name in rev_init_aliases:
            new_names = [
                k
                for k in rev_init_aliases[init.name]
                if k in torch_results and k not in skip_mapping_torch_onnx
            ]
            if new_names and len(new_names) == 1:
                new_name = new_names[0]  # type: ignore[assignment, index]
                t = torch_results[new_name]
                if (
                    len(set(t.shape)) == len(t.shape)  # not repeated dimension
                    and t.shape == tuple(init.dims)
                    and torch_dtype_to_onnx_dtype(t.dtype) == init.data_type
                ):
                    torch_names_to_onnx_names[new_name] = init.name
                else:
                    t = None

                # We should check tensors and proto are the same.
        if t is None:
            t = to_tensor(init)
            if default_device and t.numel() >= 1024:
                # Let's force its way to cuda (should check the device has well).
                t = t.to(default_device)
            records_to_yield.append(
                RunAlignedRecord(
                    onnx_id_node=-1,
                    onnx_name=init.name,
                    onnx_op_type="initializer",
                    onnx_shape_type=string_type(t, **str_kws),
                ).check(already_yielded)
            )

        size = t.element_size() * t.numel()
        if t.is_cuda:
            memory_cuda += size
        else:
            memory_cpu += size
        if init.name not in onnx_results:
            # otherwise, it is an input with a default value
            onnx_results[init.name] = _check_tensor_(use_tensor, init.name, t, flip_type=True)
    return mapping_onnx_to_torch, onnx_results, memory_cpu, memory_cuda, records_to_yield


def run_aligned(
    ep: torch.export.ExportedProgram,
    onx: Union[onnx.ModelProto, onnx.FunctionProto],
    run_cls: Callable[
        [
            Union[
                onnx.ModelProto,
                onnx.FunctionProto,
                onnx.GraphProto,
                onnx.NodeProto,
            ]
        ],
        List[Union[np.ndarray, torch.Tensor]],
    ],
    args: Optional[Tuple[torch.Tensor, ...]] = None,
    kwargs: Optional[Dict[str, Any]] = None,
    use_tensor: bool = False,
    atol: Optional[float] = None,
    rtol: Optional[float] = None,
    verbose: int = 0,
    exc: bool = True,
    reset_names: Optional[List[str]] = None,
    replay_configuration: Optional[ReplayConfiguration] = None,
    run_onnx_with_torch_inputs: bool = False,
) -> Iterator[RunAlignedRecord]:
    """
    Runs in parallel both the exported program
    and the onnx proto and looks for discrepancies.
    The function does match on result names so it assumes
    the exported program and the onnx model have the same names
    for equivalent results.

    :param ep: exported program
    :param onx: model or function proto
    :param run_cls: defines the runtime to use for this task
    :param args: input args
    :param kwargs: input kwargs
    :param use_tensor: use torch tensors instead of numpy arrays
        for the onnx runtime
    :param atol: absolute tolerance
    :param rtol: relative tolerance
    :param verbose: verbosity level
    :param exc: stops if an exception
    :param reset_names: list of names, the onnx execution takes the torch outputs instead
        of its own result if the names falls into that set
    :param replay_configuration: configuration to let the user dump any problematic
        piece of the onnx graph he wants to replay in order to investigate later,
        see :class: `ReplayConfiguration
        <onnx_diagnostic.torch_onnx.sbs.ReplayConfiguration>`
    :param run_onnx_with_torch_inputs: run an onnx operator with torch results
        if they available
    :return: a list of :class:`RunAlignedRecord
        <onnx_diagnostic.torch_onnx.sbs_dataclasses.RunAlignedRecord>`

    Example:

    .. runpython::
        :showcode:
        :warningout: UserWarning

        import pandas
        import torch
        from onnx_diagnostic.reference import (
            # This can be replaced by any runtime taking NodeProto as an input.
            ExtendedReferenceEvaluator as ReferenceEvaluator,
        )
        from onnx_diagnostic.torch_onnx.sbs import run_aligned


        class Model(torch.nn.Module):
            def forward(self, x):
                ry = x.abs()
                rz = ry.exp()
                rw = rz + 1
                ru = rw.log() + rw
                return ru


        x = torch.randn((5, 4))
        Model()(x)  # to make sure the model is running
        ep = torch.export.export(
            Model(), (x,), dynamic_shapes=({0: torch.export.Dim("batch")},)
        )
        onx = torch.onnx.export(
            Model(), (x,), dynamic_shapes=({0: torch.export.Dim("batch")},)
        ).model_proto
        results = list(
            run_aligned(ep, onx, ReferenceEvaluator, (x,), atol=1e-5, rtol=1e-5, verbose=1)
        )
        print("------------")
        print("final results")
        df = pandas.DataFrame(results)
        df = df.apply(lambda col: col.fillna("") if col.dtype == "object" else col)
        print(df)

    This example uses :class:`onnx.reference.ReferenceEvaluator` to run the onnx model
    but onnxruntime can also be used through
    :class:`onnx_diagnostic.helpers.ort_session.InferenceSessionForTorch`.
    It relies on :epkg:`onnxruntime` and selects CPU or CUDA depending
    on the device where the inputs are located.

    The :class:`torch.export.ExportedProgram` can be saved on disk
    with ``ep.save("<filename>.pt")`` and restored with
    ``torch.export.load("<filename>.pt")``. That leeds the input to save.
    We can decouple the export and the alignment.

    .. runpython::
        :showcode:
        :warningout: UserWarning

        import onnx
        import torch
        from onnx_diagnostic.torch_export_patches.patch_inputs import use_dyn_not_str


        class Model(torch.nn.Module):
            def forward(self, x):
                ry = x.abs()
                rz = ry.exp()
                rw = rz + 1
                ru = rw.log() + rw
                return ru


        x = torch.randn((5, 4))
        dynamic_shapes = ({0: "batch"},)
        Model()(x)  # to make sure the model is running
        ep = torch.export.export(
            Model(), (x,), dynamic_shapes=use_dyn_not_str(dynamic_shapes)
        )
        onx = torch.onnx.export(
            Model(), (x,), dynamic_shapes=dynamic_shapes
        ).model_proto

        torch.export.save(ep, "test_doc_sbs_example.pt2")
        onnx.save(onx, "test_doc_sbs_example.onnx")
        torch.save((x,), "test_doc_sbs_example.pt")

    Then we can restore all of them and run it.

    .. runpython::
        :showcode:
        :warningout: UserWarning

        import pandas
        import onnx
        import torch
        from onnx_diagnostic.torch_onnx.sbs import run_aligned
        from onnx_diagnostic.reference import OnnxruntimeEvaluator


        ep = torch.export.load("test_doc_sbs_example.pt2")
        onx = onnx.load("test_doc_sbs_example.onnx")
        inputs = torch.load("test_doc_sbs_example.pt")


        results = list(
            run_aligned(
                ep,
                onx,
                OnnxruntimeEvaluator,
                inputs,
                atol=1e-5,
                rtol=1e-5,
                verbose=1,
                use_tensor=True,
            )
        )
        print("------------")
        print("final results")
        df = pandas.DataFrame(results)
        df = df.apply(lambda col: col.fillna("") if col.dtype == "object" else col)
        print(df)

    A command line can also be run:

    .. code-block:: bash

            python -m onnx_diagnostic sbs -i <tensors>.input.pt \\
                                          --ep <exported_program>.pt2 \\
                                          -m <model>.onnx  \\
                                          -o results.xlsx \\
                                          -v 1 --atol=0.1 --rtol=1
    """
    assert callable(run_cls), f"run_cls={run_cls} not a callable"
    already_yielded = {}
    reset_names = set(reset_names) if reset_names else set()
    str_kws = dict(with_shape=True, with_device=True)
    has_cuda = any(
        (isinstance(t, torch.Tensor) and t.is_cuda)
        for t in flatten_object([args, kwargs], drop_keys=True)
    )
    default_device = None
    if has_cuda:
        for t in flatten_object([args, kwargs], drop_keys=True):
            if t is not None and t.is_cuda:
                default_device = t.device
                break
    run_cls_kwargs = {
        "ir_version": onx.ir_version,
        "opsets": {d.domain: d.version for d in onx.opset_import},
        "verbose": max(verbose - 1, 0),
        "providers": (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if has_cuda
            else ["CPUExecutionProvider"]
        ),
    }
    run_cls_kwargs = {
        k: v
        for k, v in run_cls_kwargs.items()
        if k in set(inspect.signature(run_cls).parameters)
    }
    if verbose:
        print(f"[run_aligned] run_cls={run_cls}")
        print(f"[run_aligned] run_cls_kwargs={run_cls_kwargs}")
        if replay_configuration:
            print(f"[run_aligned] replay={replay_configuration}")

    # preparation with ep.graph.nodes
    ep_state_dict = {**ep.state_dict, **dict(ep.named_buffers(), **ep.tensor_constants)}
    placeholders_to_state_dict = {
        **{f"p_{name.replace('.', '_').lower()}": name for name in ep.state_dict},
        **{f"b_{name.replace('.', '_').lower()}": name for name, _ in ep.named_buffers()},
        **{f"c_{name.replace('.', '_').lower()}": name for name in ep.tensor_constants},
    }
    skip_mapping_torch_onnx = _duplicated_values(placeholders_to_state_dict)
    placeholders = {}
    if verbose:
        print(f"[run_aligned] ep: model has {len(ep_state_dict)} torch constants or weights.")

    if verbose:
        print(f"[run_aligned] ep: walks through {len(ep.graph.nodes)} nodes from torch")

    # dictionary mapping result names and their position in both graphs.
    positions: Dict[str, Dict[str, int]] = {}
    ep_graph_nodes = list(ep.graph.nodes)
    torch_results: Dict[str, Any] = {}
    torch_output_names = None
    torch_input_names: List[str] = []
    name_to_ep_node = {}
    torch_names_to_onnx_names = {}
    torch_output_names = _preparation_with_fx_graph(
        ep_graph_nodes,
        name_to_ep_node,
        torch_input_names,
        onx,
        torch_names_to_onnx_names,
        skip_mapping_torch_onnx,
        torch_results,
        placeholders,
        placeholders_to_state_dict,
        ep_state_dict,
        positions,
    )

    # prepration for onnx
    if verbose:
        print(f"[run_aligned] ep: found {len(torch_results)} torch constants or weights.")
        print(f"[run_aligned] ep: found inputs  {torch_input_names}")
        print(f"[run_aligned] ep: found outputs {torch_output_names}")
        print(f"[run_aligned] nx: walks through {len(onx.graph.node)} nodes from onnx")

    mapping_onnx_to_torch, onnx_results, memory_cpu, memory_cuda, records_to_yield = (
        _preparation_with_onnx_model(
            default_device,
            use_tensor,
            onx,
            already_yielded,
            str_kws,
            positions,
            torch_names_to_onnx_names,
            torch_output_names,
            torch_results,
            skip_mapping_torch_onnx,
            verbose,
            args,
            kwargs,
        )
    )
    for record in records_to_yield:
        yield record

    if verbose:
        print(f"[run_aligned] nx: handled {len(onnx_results)} initializers from onnx")
        print(f"[run_aligned] nx: memory cpu {memory_cpu / 2**20:.3f} Mb")
        print(f"[run_aligned] nx: memory cuda {memory_cuda / 2**20:.3f} Mb")
        print(f"[run_aligned] nx: {len(onnx_results)} constants")
        print(f"[run_aligned] nx: {len(onx.graph.input)} inputs")
        print(f"[run_aligned] nx: {len(onx.graph.output)} outputs")
        print(f"[run_aligned] bo: {len(mapping_onnx_to_torch)} outputs")
        print(f"[run_aligned] run_cls_kwargs={run_cls_kwargs}")
        if verbose > 1:
            for k, v in torch_results.items():
                print(f"[run_aligned-ep] +cst: {k}: {string_type(v, **str_kws)}")
            for k, v in onnx_results.items():
                print(f"[run_aligned-nx] +ini: {k}: {string_type(v, **str_kws)}")

    # starts the side-by-side
    if verbose:
        print(
            f"[run_aligned] ep: starts side-by-side with {len(ep_graph_nodes)} "
            f"fx nodes and {len(onx.graph.node)} onnx nodes"
        )
    if verbose == 1:
        import tqdm

        loop = tqdm.tqdm(total=len(ep_graph_nodes) + len(onx.graph.node))
    else:
        loop = None

    already_run: Set[int] = set()
    ep_durations = {}
    status = StatusRunAligned()
    last_position = 0
    for i_torch, node in enumerate(ep_graph_nodes):
        if verbose > 1:
            if node.op == "call_function":
                print(
                    f"[run_aligned] run ep.graph.nodes[{i_torch}]: "
                    f"{node.op}[{node.target}] -> {node.name!r}"
                )
            else:
                print(
                    f"[run_aligned] run ep.graph.nodes[{i_torch}]: {node.op} -> {node.name!r}"
                )
        elif verbose == 1:
            loop.set_description(
                f"ep {i_torch}/{len(ep_graph_nodes)} nx {last_position}/{len(onx.graph.node)} "
                f"{status.to_str()}"
            )
            loop.update(min(1, 1 + i_torch + last_position))

        if node.op == "placeholder":
            is_input = node.name not in placeholders
            if is_input:
                torch_results[node.name] = (
                    onnx_results[torch_names_to_onnx_names[node.name]]
                    if use_tensor
                    else from_numpy(onnx_results[torch_names_to_onnx_names[node.name]])
                )
                assert isinstance(torch_results[node.name], torch.Tensor), (
                    f"torch_results[{node.name}] not a tensor but "
                    f"{type(torch_results[node.name])}, use_tensor={use_tensor}"
                )
                t = torch_results[node.name]
                if verbose > 1:
                    print(f"[run_aligned-ep] =ags: {node.name}={string_type(t, **str_kws)}")
                # Otherwise, it is an input.
                record = RunAlignedRecord(
                    ep_id_node=i_torch,
                    onnx_id_node=-1,
                    ep_name=node.name,
                    onnx_name=torch_names_to_onnx_names[node.name],
                    ep_target="input",
                    onnx_op_type="input",
                    ep_shape_type=string_type(t, **str_kws),
                    onnx_shape_type=string_type(
                        onnx_results[torch_names_to_onnx_names[node.name]], **str_kws
                    ),
                )
                yield record.check(already_yielded)
            else:
                assert node.name in placeholders_to_state_dict, (
                    f"Unable to find placeholder {node.name!r} (node.op={node.op!r}), "
                    f"existing: {sorted(placeholders_to_state_dict)}"
                )
                assert node.name in torch_results, (
                    f"placeholder {node.name!r} (node.op={node.op!r}), "
                    f"should have been added to torch_results: {sorted(torch_results)}"
                )
                t = torch_results[node.name]
                if (
                    node.name in torch_names_to_onnx_names
                    and node.name not in skip_mapping_torch_onnx
                ):
                    if verbose > 1:
                        print(
                            f"[run_aligned-ep] =plh: "
                            f"{node.name}={string_type(t, **str_kws)}"
                        )
                    record = RunAlignedRecord(
                        ep_id_node=i_torch,
                        onnx_id_node=-1,
                        ep_name=node.name,
                        onnx_name=torch_names_to_onnx_names[node.name],
                        ep_target="placeholder",
                        onnx_op_type="initializer",
                        ep_shape_type=string_type(t, **str_kws),
                        onnx_shape_type=string_type(
                            onnx_results[torch_names_to_onnx_names[node.name]], **str_kws
                        ),
                    )
                    if not is_input:
                        record.set_diff(
                            max_diff(
                                t,
                                onnx_results[torch_names_to_onnx_names[node.name]],
                                hist=[0.1, 0.01],
                            )
                        )
                    yield record.check(already_yielded)
                else:
                    if verbose > 1:
                        print(
                            f"[run_aligned-ep] +plh: {node.name}={string_type(t, **str_kws)}"
                        )
                    yield RunAlignedRecord(
                        ep_id_node=i_torch,
                        ep_name=node.name,
                        ep_target="placeholder",
                        ep_shape_type=string_type(t, **str_kws),
                    ).check(already_yielded)
            continue

        outputs = [node.name] if isinstance(node.name, str) else list(node.name)
        args, kwargs = prepare_args_kwargs(torch_results, node)
        begin = time.perf_counter()
        new_outputs = run_fx_node(node, args, kwargs)
        duration = time.perf_counter() - begin
        ep_durations[i_torch] = duration
        if isinstance(new_outputs, (torch.Tensor, int, float, list, tuple)):
            new_outputs = (new_outputs,)

        if new_outputs is None:
            # Probably an assert.
            continue

        for k, v in zip(outputs, new_outputs):
            torch_results[k] = v
        if verbose > 1:
            for k, v in zip(outputs, new_outputs):
                print(f"[run_aligned-ep] +res: {k}={string_type(v, **str_kws)}")

        max_pos = -2
        for n in outputs:
            if n in positions:
                if "onnx" in positions[n]:
                    max_pos = max(max_pos, positions[n]["onnx"])
                if "fx" in positions[n]:
                    if positions[n]["fx"] > i_torch:
                        max_pos = -2
                        break
        if max_pos == -2:
            # we skip.
            continue

        next_to_visit = last_position
        for i_onnx in range(last_position, max_pos + 1):
            if i_onnx in already_run:
                continue
            # The onnx node may produce more than one output, in that
            # case, we need to check the exported program is not behind.
            node = onx.graph.node[i_onnx]
            ep_behind = False
            for iname in node.output:
                if iname in positions and "fx" in positions[iname]:
                    if positions[iname]["fx"] > i_torch:
                        ep_behind = True
                        break
            if ep_behind:
                break

            for r in _loop_onnx_node(
                onx,
                ep_graph_nodes,
                onnx_results,
                mapping_onnx_to_torch,
                torch_results,
                ep_durations,
                use_tensor,
                i_torch,
                i_onnx,
                name_to_ep_node,
                run_cls_kwargs,
                str_kws,
                status,
                already_run,
                torch_names_to_onnx_names,
                verbose,
                exc,
                atol,
                rtol,
                reset_names,
                replay_configuration,
                has_cuda,
                run_cls,
                loop,
                run_onnx_with_torch_inputs,
            ):
                if r:
                    yield r.check(already_yielded)
            next_to_visit = i_onnx + 1

        last_position = next_to_visit

    # complete the execution of the onnx graph
    if verbose > 1:
        print(
            f"[run_aligned] complete execution of onnx graph from pos={last_position} "
            f"to {len(onx.graph.node)}"
        )
    for i_onnx in range(last_position, len(onx.graph.node)):
        if i_onnx in already_run:
            continue
        for r in _loop_onnx_node(
            onx,
            ep_graph_nodes,
            onnx_results,
            mapping_onnx_to_torch,
            torch_results,
            ep_durations,
            use_tensor,
            i_torch,
            i_onnx,
            name_to_ep_node,
            run_cls_kwargs,
            str_kws,
            status,
            already_run,
            torch_names_to_onnx_names,
            verbose,
            exc,
            atol,
            rtol,
            reset_names,
            replay_configuration,
            has_cuda,
            run_cls,
            loop,
            run_onnx_with_torch_inputs,
        ):
            if r:
                yield r.check(already_yielded)

    if loop is not None:
        loop.close()
    if verbose:
        print(f"[run_aligned] done with status={status.to_str()}")
