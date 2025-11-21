import inspect
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union
import onnx
import onnx.helper as oh
import numpy as np
import torch
from ..helpers import string_type, string_diff, max_diff, flatten_object
from ..helpers.onnx_helper import pretty_onnx
from ..helpers.torch_helper import to_numpy, from_numpy, to_tensor


def validate_fx_tensor(
    node: torch.fx.Node, tensor: torch.Tensor, expected_shape: Tuple[Any, ...]
) -> None:
    """
    Validates the shape of tensor is expected.

    :param node: node
    :param tensor: tensor
    :param expected_shape: expected shape
    """
    assert len(tensor.shape) == len(expected_shape), (
        f"Shape mismatch, got {tensor.shape} expected {expected_shape}, "
        f"node.name={node.name!r}, node.target={getattr(node, 'target', None)}, "
        f"node.args={node.args}, node.kwargs={node.kwargs}, "
        f"node.meta={node.meta}"
    )
    for a, b in zip(tensor.shape, expected_shape):
        assert not isinstance(b, int) or a == b or {a, b} == {0, 1}, (
            f"Dimension mismatch, got {tensor.shape} expected {expected_shape}, "
            f"node.name={node.name!r}, node.target={getattr(node, 'target', None)}, "
            f"node.args={node.args}, node.kwargs={node.kwargs}, "
            f"node.meta={node.meta}"
        )


def validate_fx_outputs(node: torch.fx.Node, outputs: Tuple[Any, ...]) -> None:
    """
    Validates the outputs of a node using metadata stored in the node.

    :param node: node
    :param outputs: outputs
    """
    if "val" not in node.meta:
        return
    if isinstance(outputs, torch.Tensor):
        validate_fx_tensor(node, outputs, node.meta["val"].shape)
        return
    if isinstance(outputs, (tuple, list)):
        assert isinstance(node.meta["val"], (list, tuple)), (
            f"Unexpected type {string_type(node.meta['val'])} for node.meta['val'], "
            f"node.name={node.name!r}, node.target={getattr(node, 'target', None)}, "
            f"node.args={node.args}, node.kwargs={node.kwargs}, "
            f"node.meta={node.meta}"
        )
        assert len(outputs) == len(node.meta["val"]), (
            f"Length mismatch, got {len(outputs)} expected {len(node.meta['val'])}, "
            f"node.name={node.name!r}, node.target={getattr(node, 'target', None)}, "
            f"node.args={node.args}, node.kwargs={node.kwargs}, "
            f"node.meta={node.meta}"
        )
        for a, b in zip(outputs, node.meta["val"]):
            validate_fx_tensor(node, a, b.shape)
        return
    if isinstance(outputs, int):
        assert (
            isinstance(node.meta["val"], (torch.SymInt, torch.SymBool, torch.SymFloat))
            or outputs == node.meta["val"]
        ), (
            f"Int mismatch, got {outputs} expected {node.meta['val']}, "
            f"node.name={node.name!r}, node.target={getattr(node, 'target', None)}, "
            f"node.args={node.args}, node.kwargs={node.kwargs}, "
            f"node.meta={node.meta}"
        )
        return
    if outputs is None:
        assert node.meta["val"] is None, (
            f"None mismatch, got {outputs} expected {node.meta['val']}, "
            f"node.name={node.name!r}, node.target={getattr(node, 'target', None)}, "
            f"node.args={node.args}, node.kwargs={node.kwargs}, "
            f"node.meta={node.meta}"
        )
        return
    raise NotImplementedError(
        f"Validation for output type {type(outputs)} is not implemented, "
        f"node.name={node.name!r}, node.target={getattr(node, 'target', None)}, "
        f"node.args={node.args}, node.kwargs={node.kwargs}, "
        f"node.meta={node.meta}"
    )


def run_fx_node(
    node: torch.fx.Node, args: Tuple[Any, ...], kwargs: Optional[Dict[str, Any]] = None
) -> Tuple[Any, ...]:
    """
    Executes a node

    :param node: runs a node
    :param args: unnamed inputs to the node
    :param kwargs: named inputs to the node
    :return: results
    """
    if node.op == "output":
        assert len(args) == 1 and not kwargs, (
            f"Unexpected inputs: args={string_type(args, limit=20)} "
            f"kwargs={string_type(kwargs, limit=20)}"
        )
        return args
    if node.op == "call_function":
        assert callable(node.target), f"{node.target!r} not callable in node {node!r}"
        for a, ea in zip(args, node.args):
            if isinstance(a, torch.Tensor) and hasattr(ea, "meta") and "val" in ea.meta:
                ta = ea.meta["val"]
                assert (
                    isinstance(ta, torch.Tensor)
                    and len(a.shape) == len(ta.shape)
                    and a.dtype == ta.dtype
                ), (
                    f"Unable to run node {node!r}, target={node.target!r}, "
                    f"node.args={node.args!r}, node.kwargs={node.kwargs!r}, "
                    f"args={string_type(args, with_shape=True, with_device=True)}, "
                    f"kwargs={string_type(kwargs, with_shape=True, with_device=True)}"
                )
        try:
            outputs = node.target(*args, **(kwargs or {}))
        except RuntimeError as e:
            raise RuntimeError(
                f"Unable to run node {node!r}, target={node.target!r}, "
                f"args={string_type(args, with_shape=True, with_device=True)}, "
                f"kwargs={string_type(kwargs, with_shape=True, with_device=True)}"
            ) from e
        validate_fx_outputs(node, outputs)
        return outputs
    raise NotImplementedError(
        f"node.op={node.op!r} is not implemented, node.name={node.name!r}"
    )


def _pick_result(torch_results: Dict[str, Any], ref: Any) -> Any:
    "See :func:`prepare_args_kwargs`."
    if isinstance(ref, torch.fx.Node):
        return torch_results[ref.name]
    if isinstance(ref, list):
        return [_pick_result(torch_results, n) for n in ref]
    if isinstance(ref, tuple):
        return tuple(_pick_result(torch_results, n) for n in ref)
    if isinstance(ref, dict):
        return {k: _pick_result(torch_results, v) for k, v in ref.items()}
    if isinstance(ref, (bool, int, float, str, torch.device, torch.dtype)):
        return ref
    if ref is None:
        return None
    if isinstance(ref, torch.layout):
        return ref
    raise NotImplementedError(f"Unable to process args type {type(ref)}")


def prepare_args_kwargs(
    torch_results: Dict[str, Any], node: torch.fx.Node
) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    """
    Prepares args and kwargs before executing a fx node.

    :param torch_results: existing results
    :param node: node to execute
    :return: new args and kwargs
    """
    new_args = _pick_result(torch_results, node.args)
    new_kwargs = _pick_result(torch_results, node.kwargs)
    return new_args, new_kwargs


@dataclass
class RunAlignedRecord:
    ep_id_node: Optional[int] = None
    onnx_id_node: Optional[int] = None
    ep_name: Optional[str] = None
    onnx_name: Optional[str] = None
    ep_target: Optional[str] = None
    onnx_op_type: Optional[str] = None
    onnx_id_output: Optional[int] = None
    shape_type: Optional[str] = None
    err_abs: Optional[float] = None
    err_rel: Optional[float] = None
    err_dev: Optional[float] = None
    err_nan: Optional[float] = None
    ep_time_run: Optional[float] = None
    onnx_time_run: Optional[float] = None

    def set_diff(self, diff: Dict[str, Any]):
        if diff is None:
            return
        if "abs" in diff:
            self.err_abs = diff["abs"]
        if "rel" in diff:
            self.err_rel = diff["rel"]
        if "dev" in diff:
            self.err_dev = diff["dev"]
        if "nan" in diff:
            self.err_nan = diff["nan"]


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
    :return: a list of :class:`RunAlignedRecord`

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

    def _check_tensor_(name, obj, flip_type=False):
        if flip_type:
            if use_tensor:
                if isinstance(obj, np.ndarray):
                    obj = from_numpy(obj)
            else:
                if isinstance(obj, torch.Tensor):
                    obj = to_numpy(obj)

        assert not use_tensor or isinstance(obj, torch.Tensor), (
            f"Unexpected type {type(obj)} for {name!r}. "
            f"use_tensor is True so torch.Tensor is expected."
        )
        assert use_tensor or isinstance(obj, np.ndarray), (
            f"Unexpected type {type(obj)} for {name!r}. "
            f"use_tensor is False so np.array is expected."
        )
        return obj

    def _make_node_from_initializer(proto: onnx.TensorProto) -> onnx.NodeProto:
        return oh.make_node("Constant", [], [proto.name], value=proto)

    def _loop_cmp(
        mapping_onnx_to_torch,
        torch_results,
        onnx_results,
        o,
        r,
        verbose,
        atol,
        rtol,
        i,
        i_onnx,
    ):
        onnx_results[o] = _check_tensor_(o, r)
        if verbose > 1:
            print(f"[run_aligned-nx] +res: {o}={string_type(r, **str_kws)}")

        to = mapping_onnx_to_torch.get(o, o)
        if to in torch_results:
            d = max_diff(torch_results[to], r)
            if verbose > 1:
                if o == to:
                    print(f"[run_aligned-==] cmp {to}: {string_diff(d)}")
                else:
                    print(f"[run_aligned-~~] cmd {to}/{o}: {string_diff(d)}")
                if not (
                    atol is None or rtol is None or (d["abs"] <= atol and d["rel"] <= rtol)
                ):
                    if exc:
                        raise ValueError(
                            f"discrepancies detected for results [{to}/{o}]: "
                            f"{string_diff(d)}"
                            f"\n-- torch_results: {string_type(torch_results[to], **str_kws)}"
                            f"\n-- onnx_results: {string_type(r, **str_kws)}"
                            f"\n-- torch\n{torch_results[to]}\n-- onnx\n{r}"
                        )
                    else:
                        print(f"[run_align-dx] discrepancies {string_diff(d)} - [{to}/{o}]")
            r = RunAlignedRecord(
                ep_id_node=i,
                onnx_id_node=i_onnx,
                ep_name=o,
                onnx_name=to,
                shape_type=string_type(torch_results[to], **str_kws),
            )
            r.set_diff(d)
            return r
        return None

    # preparation with ep.graph.nodes
    ep_state_dict = {**ep.state_dict, **dict(ep.named_buffers(), **ep.tensor_constants)}
    placeholders_to_state_dict = {
        **{f"p_{name.replace('.', '_')}": name for name in ep.state_dict},
        **{f"b_{name.replace('.', '_')}": name for name, _ in ep.named_buffers()},
        **{f"c_{name.replace('.', '_')}": name for name in ep.tensor_constants},
    }
    placeholders = {}
    assert len(placeholders_to_state_dict) == len(ep_state_dict), (
        f"Some names are confusing between {sorted(ep_state_dict)} "
        f"and {sorted(placeholders_to_state_dict)}"
    )
    if verbose:
        print(f"[run_aligned] ep: model has {len(ep_state_dict)} torch constants or weights.")

    if verbose:
        print(f"[run_aligned] ep: walks through {len(ep.graph.nodes)} nodes from torch")
    positions: Dict[str, Any] = {}
    ep_graph_nodes = list(ep.graph.nodes)
    torch_results: Dict[str, Any] = {}
    last_position = 0
    torch_output_names = None
    torch_input_names = []
    name_to_ep_node = {}
    torch_names_to_onnx_names = {}
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
                    f"node.name={node.name!r} cannot be an input"
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

    # prepration for onnx
    if verbose:
        print(f"[run_aligned] ep: found {len(torch_results)} torch constants or weights.")
        print(f"[run_aligned] ep: found inputs  {torch_input_names}")
        print(f"[run_aligned] ep: found outputs {torch_output_names}")
        print(f"[run_aligned] nx: walks through {len(onx.graph.node)} nodes from onnx")
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
        onnx_results[inp.name] = _check_tensor_(inp.name, v if use_tensor else to_numpy(v))
        if verbose:
            print(f"[run_aligned-nx] +inp: {inp.name}: {string_type(v, **str_kws)}")

    # initializers
    if verbose:
        print(f"[run_aligned] nx: handles {len(onx.graph.initializer)} initializers from onnx")
    memory_cpu = 0
    memory_cuda = 0
    for init in onx.graph.initializer:  # type: ignore
        positions[init.name] = -1
        if init.name in torch_results:
            t = torch_results[init.name]
            torch_names_to_onnx_names[init.name] = init.name
            # We should check tensors and proto are the same.
        else:
            new_name = f"p_{init.name.replace('.', '_')}"
            if new_name in torch_results:
                t = torch_results[new_name]
                torch_names_to_onnx_names[init.name] = new_name
                # We should check tensors and proto are the same.
            else:
                t = to_tensor(init)
                if default_device and t.numel() >= 1024:
                    # Let's force its way to cuda (should check the device has well).
                    t = t.to(default_device)
                    yield RunAlignedRecord(
                        onnx_id_node=-1,
                        onnx_name=n,
                        onnx_op_type="initializer",
                        shape_type=string_type(t, **str_kws),
                    )

        size = t.element_size() * t.numel()
        if t.is_cuda:
            memory_cuda += size
        else:
            memory_cpu += size
        if init.name not in onnx_results:
            # otherwise, it is an input with a default value
            onnx_results[init.name] = _check_tensor_(init.name, t, flip_type=True)

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
    if verbose == 1:
        import tqdm

        loop = tqdm.tqdm(list(enumerate(ep_graph_nodes)))
    else:
        loop = list(enumerate(ep_graph_nodes))

    if verbose:
        print(f"[run_aligned] ep: starts side-by-side with {len(ep_graph_nodes)} nodes")
    already_run = set()
    ep_durations = {}
    yielded_nodes = 0
    max_abs = 0
    for i, node in loop:
        if verbose > 1:
            if node.op == "call_function":
                print(
                    f"[run_aligned] run ep.graph.nodes[{i}]: "
                    f"{node.op}[{node.target}] -> {node.name!r}"
                )
            else:
                print(f"[run_aligned] run ep.graph.nodes[{i}]: {node.op} -> {node.name!r}")
        elif verbose == 1:
            loop.set_description(
                f"ep {i}/{len(ep_graph_nodes)} nx {last_position}/{len(onx.graph.node)} "
                f"mapped {yielded_nodes} maxabs {max_abs:1.5f}"
            )

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
                if verbose > 1:
                    t = torch_results[node.name]
                    print(f"[run_aligned-ep] =ags: {node.name}={string_type(t, **str_kws)}")
                # Otherwise, it is an input.
                record = RunAlignedRecord(
                    ep_id_node=i,
                    onnx_id_node=-1,
                    ep_name=node.name,
                    onnx_name=torch_names_to_onnx_names[node.name],
                    ep_target="input",
                    onnx_op_type="input",
                    shape_type=string_type(t, **str_kws),
                )
                yield record
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
                if node.name in torch_names_to_onnx_names:
                    if verbose > 1:
                        print(
                            f"[run_aligned-ep] =plh: "
                            f"{node.name}={string_type(t, **str_kws)}"
                        )
                    record = RunAlignedRecord(
                        ep_id_node=i,
                        onnx_id_node=-1,
                        ep_name=node.name,
                        onnx_name=torch_names_to_onnx_names[node.name],
                        ep_target="placeholder",
                        onnx_op_type="initializer",
                        shape_type=string_type(t, **str_kws),
                    )
                    if not is_input:
                        record.set_diff(
                            max_diff(
                                ep_state_dict[placeholders_to_state_dict[node.name]],
                                onnx_results[node.name],
                                hist=[0.1],
                            )
                        )
                    yield record
                else:
                    if verbose > 1:
                        print(
                            f"[run_aligned-ep] +plh: {node.name}={string_type(t, **str_kws)}"
                        )
                    yield RunAlignedRecord(
                        ep_id_node=i,
                        ep_name=node.name,
                        ep_target="placeholder",
                        shape_type=string_type(torch_results[node.name], **str_kws),
                    )
            continue

        outputs = [node.name] if isinstance(node.name, str) else list(node.name)
        args, kwargs = prepare_args_kwargs(torch_results, node)
        begin = time.perf_counter()
        new_outputs = run_fx_node(node, args, kwargs)
        duration = time.perf_counter() - begin
        ep_durations[i] = duration
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
            if n in positions and "onnx" in positions[n]:
                max_pos = max(max_pos, positions[n]["onnx"])
        if max_pos == -2:
            # we skip.
            continue

        for i_onnx in range(last_position, max_pos + 1):
            if i_onnx in already_run:
                continue
            node = onx.graph.node[i_onnx]
            if verbose > 1:
                print(
                    f"[run_aligned] run onx.graph.node[{i_onnx}]: "
                    f"{node.op_type}({', '.join(node.input)}) -> {', '.join(node.output)}"
                )
            elif verbose == 1:
                loop.set_description(
                    f"ep {i}/{len(ep_graph_nodes)} nx {i_onnx}/{len(onx.graph.node)} "
                    f"mapped {yielded_nodes} maxabs {max_abs:1.5f}"
                )
            ref = run_cls(node, **run_cls_kwargs)
            # We need to clone because the runtime maybe using dlpack to create OrtValue
            feeds = {k: onnx_results[k] for k in node.input if k}
            assert "" not in feeds, f"Unexpected feeds={string_type(feeds, **str_kws)}"
            begin = time.perf_counter()
            try:
                res = ref.run(None, feeds)  # type: ignore[attr-defined]
            except Exception as e:
                raise RuntimeError(
                    f"Unable to run node {node.op_type}, domain={node.domain} "
                    f"with inputs={node.input}, feeds={string_type(feeds, **str_kws)}"
                ) from e
            duration = time.perf_counter() - begin
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
            list_node_output = list(node.output)
            node_output = [o for o in list_node_output if o]
            for o, r in zip(node_output, res):
                if r is None or o is None:
                    continue
                tmp = _loop_cmp(
                    mapping_onnx_to_torch,
                    torch_results,
                    onnx_results,
                    o,
                    r,
                    verbose,
                    atol,
                    rtol,
                    i,
                    i_onnx,
                )
                if tmp is not None:
                    tmp.ep_id_node = name_to_ep_node[tmp.ep_name]
                    tmp.ep_target = str(ep_graph_nodes[tmp.ep_id_node].target)
                    tmp.onnx_op_type = onx.graph.node[tmp.onnx_id_node].op_type
                    tmp.onnx_id_output = list_node_output.index(o)
                    tmp.ep_time_run = ep_durations[tmp.ep_id_node]
                    tmp.onnx_time_run = duration
                    yielded_nodes += 1
                    if tmp.err_abs is not None:
                        max_abs = max(max_abs, tmp.err_abs)
                    yield tmp
            already_run.add(i_onnx)

        last_position = max_pos + 1

    # complete the execution of the onnx graph
    if verbose:
        print(
            f"[run_aligned] complete execution of onnx graph from pos={last_position} "
            f"to {len(onx.graph.node)}"
        )
    for i_onnx in range(last_position, len(onx.graph.node)):
        if i_onnx in already_run:
            continue
        node = onx.graph.node[i_onnx]
        if verbose > 1:
            print(
                f"[run_aligned] run onx.graph.node[{i_onnx}]: "
                f"{node.op_type}({', '.join(node.input)}) -> {', '.join(node.output)}"
            )
        ref = run_cls(node, **run_cls_kwargs)
        feeds = {k: onnx_results[k] for k in node.input if k}
        assert "" not in feeds, f"Unexpected feeds={string_type(feeds, **str_kws)}"
        begin = time.perf_counter()
        res = ref.run(None, feeds)  # type: ignore[attr-defined]
        duration = time.perf_counter() - begin
        list_node_output = list(node.output)
        node_output = [o for o in list_node_output if o]
        for o, r in zip(node_output, res):
            if r is None or o is None:
                continue
            tmp = _loop_cmp(
                mapping_onnx_to_torch,
                torch_results,
                onnx_results,
                o,
                r,
                verbose,
                atol,
                rtol,
                i,
                i_onnx,
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
                yielded_nodes += 1
                if tmp.err_abs is not None:
                    max_abs = max(max_abs, tmp.err_abs)
                yield tmp
        already_run.add(i_onnx)

    if verbose:
        print(f"[run_aligned] done with {yielded_nodes} mapped nodes")
        print(f"[run_aligned] max absolution error={max_abs}")
