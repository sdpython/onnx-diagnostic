import inspect
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union
import onnx
import onnx.helper as oh
import numpy as np
import torch
from ..helpers import string_type, string_diff, max_diff, flatten_object
from ..helpers.onnx_helper import pretty_onnx
from ..helpers.torch_helper import to_numpy, from_numpy


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
        outputs = node.target(*args, **(kwargs or {}))
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
) -> Iterator[Tuple[Any, ...]]:
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
    :param atol: absolute tolerance
    :param rtol: relative tolerance
    :param verbose: verbosity level
    :return: a list of tuples containing the results, they come in tuple

    Example:

    .. runpython::
        :showcode:
        :warningout: UserWarning

        import pandas
        import torch
        from onnx_diagnostic.reference import (
            # This can be replace by any runtime taking NodeProto as an input.
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


        def post_process(obs):
            dobs = dict(zip(["ep_id_node", "onnx_id_node", "ep_name", "onnx_name"], obs))
            dobs["err_abs"] = obs[-1]["abs"]
            dobs["err_rel"] = obs[-1]["rel"]
            return dobs


        x = torch.randn((5, 4))
        Model()(x)  # to make sure the model is running
        ep = torch.export.export(
            Model(), (x,), dynamic_shapes=({0: torch.export.Dim("batch")},)
        )
        onx = torch.onnx.export(
            Model(), (x,), dynamic_shapes=({0: torch.export.Dim("batch")},)
        ).model_proto
        results = list(
            map(
                post_process,
                run_aligned(
                    ep, onx, ReferenceEvaluator, (x,), atol=1e-5, rtol=1e-5, verbose=1
                ),
            ),
        )
        print("------------")
        print("final results")
        df = pandas.DataFrame(results)
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
        from onnx_diagnostic.torch_export_patches.patch_inputs.use_dyn_not_str


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


        def post_process(obs):
            dobs = dict(zip(["ep_id_node", "onnx_id_node", "ep_name", "onnx_name"], obs))
            dobs["err_abs"] = obs[-1]["abs"]
            dobs["err_rel"] = obs[-1]["rel"]
            return dobs


        results = list(
            map(
                post_process,
                run_aligned(
                    ep,
                    onx,
                    OnnxruntimeEvaluator,
                    inputs,
                    atol=1e-5,
                    rtol=1e-5,
                    verbose=1,
                    use_tensor=True,
                ),
            ),
        )
        print("------------")
        print("final results")
        df = pandas.DataFrame(results)
        print(df)
    """
    assert callable(run_cls), f"run_cls={run_cls} not a callable"
    str_kws = dict(with_shape=True, with_device=True, with_min_max=True)
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
        mapping_onnx_to_torch, torch_results, onnx_results, o, r, verbose, atol, rtol
    ):
        onnx_results[o] = _check_tensor_(o, r)
        if verbose:
            print(f"[run_aligned-nx] +res: {o}={string_type(r, **str_kws)}")

        to = mapping_onnx_to_torch.get(o, o)
        if to in torch_results:
            d = max_diff(torch_results[to], r)
            if verbose:
                if o == to:
                    print(f"[run_aligned-==] cmp {to}: {string_diff(d)}")
                else:
                    print(f"[run_aligned-~~] cmd {to}/{o}: {string_diff(d)}")
                if not (
                    atol is None or rtol is None or (d["abs"] <= atol and d["rel"] <= rtol)
                ):
                    raise ValueError(
                        f"discrepancies detected for results [{to}/{o}]: "
                        f"{string_diff(d)}"
                        f"\n-- torch_results: {string_type(torch_results[to], **str_kws)}"
                        f"\n-- onnx_results: {string_type(r, **str_kws)}"
                        f"\n-- torch\n{torch_results[to]}\n-- onnx\n{r}"
                    )
            return (i, i_onnx, o, to, d)
        return None

    if verbose:
        print(f"[run_aligned] walks through {len(ep.graph.nodes)} nodes from torch")
    positions: Dict[str, Any] = {}
    for i, node in enumerate(ep.graph.nodes):
        if isinstance(node.name, str):
            positions[node.name] = dict(fx=i)
        else:
            for n in node.name:
                positions[n] = dict(fx=i)

    if verbose:
        print(f"[run_aligned] walks through {len(onx.graph.node)} nodes from onnx")
    for i, node in enumerate(onx.graph.node):
        for n in node.output:
            if n in positions:
                positions[n]["onnx"] = i
            else:
                positions[n] = dict(onnx=i)

    if verbose:
        print(f"[run_aligned] handles {len(onx.graph.initializer)} initializers from onnx")
    onnx_results: Dict[str, Any] = {}
    for init in onx.graph.initializer:  # type: ignore
        positions[init.name] = -1
        t = run_cls(
            _make_node_from_initializer(init),
            **run_cls_kwargs,
        ).run(  # type: ignore[attr-defined]
            None, {}
        )[
            0
        ]
        if default_device and t.numel() >= 1024:
            # Let's force its way to cuda (should check the device has well).
            t = t.to(default_device)
        onnx_results[init.name] = _check_tensor_(init.name, t, flip_type=True)
        param_name = f"p_{init.name.replace('.', '_')}"
        if param_name == init.name:
            continue
        assert param_name not in onnx_results, (
            f"Some confusion may happen because {init.name!r} -> {param_name!r} "
            f"and onnx_results has {sorted(onnx_results)}"
        )
        onnx_results[param_name] = onnx_results[init.name]

    if verbose:
        print(f"[run_aligned] handles common {len(onnx_results)} initializer from torch")
    # we should be careful, torch may modified inplace the weights,
    # it may be difficult to share weights
    torch_results: Dict[str, Any] = {
        k: (v if use_tensor else from_numpy(v))
        for k, v in onnx_results.items()
        if not k.startswith("init")
    }
    if verbose:
        print(
            f"[run_aligned] handles other constant from {len(ep.graph.nodes)} nodes from torch"
        )
    last_position = 0
    torch_output_names = None
    for node in ep.graph.nodes:
        if node.op == "output":
            torch_output_names = [n.name for n in node.args[0]]
    onnx_outputs_names = [o.name for o in onx.graph.output]
    assert torch_output_names is not None and len(torch_output_names) == len(
        onnx_outputs_names
    ), (
        f"Unexpected number of outputs, torch_output_names={torch_output_names}, "
        f"onnx_outputs_names={onnx_outputs_names}"
    )
    mapping_onnx_to_torch = dict(zip(onnx_outputs_names, torch_output_names))

    if verbose:
        print(f"[run_aligned]  torch {len(torch_results)} constants")
        print(f"[run_aligned]   onnx {len(onnx_results)} constants")
        print(f"[run_aligned] common {len(mapping_onnx_to_torch)} constants")
        for k, v in torch_results.items():
            print(f"[run_aligned-ep] +cst: {k}: {string_type(v, str_kws)}")
        for k, v in onnx_results.items():
            print(f"[run_aligned-nx] +ini: {k}: {string_type(v, str_kws)}")

    onnx_args = list(args) if args else []
    if kwargs:
        onnx_args.extend(flatten_object(kwargs, drop_keys=True))
    if verbose:
        print(f"[run_aligned]   args: {string_type(args, **str_kws)}")
        print(f"[run_aligned] kwargs: {string_type(kwargs, **str_kws)}")
        print(f"[run_aligned]   onnx: {string_type(onnx_args, **str_kws)}")
        print(f"[run_aligned] walks through {len(onx.graph.input)} onnx inputs")
    for inp, v in zip(onx.graph.input, onnx_args):
        onnx_results[inp.name] = _check_tensor_(inp.name, v if use_tensor else to_numpy(v))
        if verbose:
            print(f"[run_aligned-nx] +inp: {inp.name}: {string_type(v, **str_kws)}")

    for i, node in enumerate(ep.graph.nodes):
        if verbose:
            if node.op == "call_function":
                print(
                    f"[run_aligned] run ep.graph.nodes[{i}]: "
                    f"{node.op}[{node.target}] -> {node.name!r}"
                )
            else:
                print(f"[run_aligned] run ep.graph.nodes[{i}]: {node.op} -> {node.name!r}")

        if node.op == "placeholder":
            if node.name in onnx_results:
                torch_results[node.name] = (
                    onnx_results[node.name]
                    if use_tensor
                    else torch.from_numpy(onnx_results[node.name])
                )
                if verbose:
                    t = torch_results[node.name]
                    print(f"[run_aligned-ep] +plh: {node.name}={string_type(t, **str_kws)}")
                continue
            raise AssertionError(
                f"unable to process node {node.op} -> {node.name!r} "
                f"not in {sorted(onnx_results)}, "
                f"args={string_type(args, **str_kws)}, "
                f"kwargs={string_type(kwargs, **str_kws)}, "
                f"onx.graph.input={[i.name for i in onx.graph.input]}"
            )

        outputs = [node.name] if isinstance(node.name, str) else list(node.name)
        args, kwargs = prepare_args_kwargs(torch_results, node)
        new_outputs = run_fx_node(node, args, kwargs)
        if isinstance(new_outputs, (torch.Tensor, int, float, list)):
            new_outputs = (new_outputs,)

        if new_outputs is None:
            # Probably an assert.
            continue

        for k, v in zip(outputs, new_outputs):
            torch_results[k] = v
        if verbose:
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
            node = onx.graph.node[i_onnx]
            if verbose:
                print(
                    f"[run_aligned] run onx.graph.node[{i_onnx}]: "
                    f"{node.op_type}({', '.join(node.input)}) -> {', '.join(node.output)}"
                )
            ref = run_cls(node, **run_cls_kwargs)
            feeds = {k: onnx_results[k] for k in node.input}
            res = ref.run(None, feeds)  # type: ignore[attr-defined]
            assert (
                not has_cuda
                or not any(t is not None and t.is_cuda for t in feeds.values())
                or any(
                    t is not None
                    and t.is_cuda
                    and t.dtype in {torch.float32, torch.float16, torch.bfloat16}
                    for t in res
                )
                or node.op_type in {"Shape", "Size"}  # on CPU no matter what
            ), (
                f"One input is on cuda but there is no float output on cuda, "
                f"feeds={string_type(feeds, with_device=True, with_shape=True)}, "
                f"res={string_type(res, with_device=True, with_shape=True)}, "
                f"node is {pretty_onnx(node)}"
            )
            for o, r in zip(node.output, res):
                tmp = _loop_cmp(
                    mapping_onnx_to_torch,
                    torch_results,
                    onnx_results,
                    o,
                    r,
                    verbose,
                    atol,
                    rtol,
                )
                if tmp is not None:
                    yield tmp

        last_position = max_pos + 1

    # complete the execution of the onnx graph
    if verbose:
        print(f"[run_aligned] complete execution of onnx graph from pos={last_position}")
    for i_onnx in range(last_position, len(onx.graph.node)):
        node = onx.graph.node[i_onnx]
        if verbose:
            print(
                f"[run_aligned] run onx.graph.node[{i_onnx}]: "
                f"{node.op_type}({', '.join(node.input)}) -> {', '.join(node.output)}"
            )
        ref = run_cls(node, **run_cls_kwargs)
        feeds = {k: onnx_results[k] for k in node.input}
        res = ref.run(None, feeds)  # type: ignore[attr-defined]
        for o, r in zip(node.output, res):
            tmp = _loop_cmp(
                mapping_onnx_to_torch, torch_results, onnx_results, o, r, verbose, atol, rtol
            )
            if tmp is not None:
                yield tmp
