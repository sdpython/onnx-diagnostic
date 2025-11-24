import inspect
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Self, Set, Tuple, Union
import onnx
import onnx.helper as oh
import numpy as np
import torch
from ..helpers import string_type, string_diff, max_diff, flatten_object
from ..helpers.onnx_helper import pretty_onnx, extract_subset_of_nodes, make_submodel
from ..helpers.torch_helper import to_numpy, from_numpy, to_tensor, torch_dtype_to_onnx_dtype


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
class ReplayConfiguration:
    """
    Configuration specifying how to replay or dump pieces of
    onnx graph in order to replay them later and investigate
    later possible sources of discrepancies.

    :param dump_folder: where to dump the onnx model corresponding to the
        pieces to investigate
    :param selected_names: list of results names to dump
    :param selected_op_types: list of onnx operators to dump
    :param threshold: only keep those whose discrepancies is greater than that threshold
    """

    dump_folder: str
    selected_names: Optional[Set[str]] = None
    selected_op_types: Optional[Set[str]] = None
    threshold: float = 0.1

    def __post_init__(self):
        assert self.dump_folder, "dump_folder is empty and this is not allowed for the replay"

    def select(
        self,
        name: Optional[str] = None,
        op_type: Optional[str] = None,
        err_abs: Optional[float] = None,
    ) -> bool:
        """
        Returns true or false whether or not a piece of the onnx model should be dumped,
        around a particular node. The results is True if one of the condition is true:

        * ``name in self.selected_names``
        * ``op_type in self.selected_op_types``
        * ``err_abs >= self.threshold``

        :param name: result name
        :param op_type: operator type
        :param err_abs: measured discrepancy
        :return: True if this should be dumped
        """
        if name and self.selected_names and name in self.selected_names:
            return True
        if op_type and self.selected_op_types and op_type in self.selected_op_types:
            return True
        if err_abs is not None and self.threshold is not None and err_abs >= self.threshold:
            return True
        return False

    def dump(
        self,
        name: str,
        onnx_id_node: int,
        model: onnx.ModelProto,
        onnx_results: Dict[str, Any],
        torch_results: Dict[str, torch.Tensor],
        onnx_name_to_ep_name: Dict[str, str],
        verbose: int = 0,
    ) -> Optional[str]:
        """
        Dumps the minimal graph which can be replayed outside the model.

        :param name: name of the result to look into
        :param onnx_id_node: index of the node which produces it model `model`
        :param model: onnx model
        :param onnx_results: all known onnx results
        :param torch_results: all known torch results
        :param onnx_name_to_ep_name: correspondence between onnx_node name
            and exported program name
        :param verbose: verbosity level
        :return: the folder created to dump everything
        """
        if verbose:
            print(f"[ReplayConfiguration.dump] extract subset of node for {name!r}")
        nodes = extract_subset_of_nodes(
            model=model,
            name=name,
            node_index=onnx_id_node,
            cut_points=set(onnx_name_to_ep_name),
        )
        if not nodes:
            if verbose:
                print(
                    f"[ReplayConfiguration.dump] could not extract subset of node for {name!r}"
                )
            return None
        if verbose:
            print(f"[ReplayConfiguration.dump] make model with {len(nodes)} nodes")
        submodel = make_submodel(
            nodes,
            ir_version=model.ir_version,
            opset_imports=model.opset_import,
            output_names=[name],
            type_rank_fn=lambda name: (
                torch_dtype_to_onnx_dtype(onnx_results[name].dtype),
                len(onnx_results[name].shape),
            ),
        )
        input_names = [n.name for n in submodel.graph.input]
        if verbose:
            print(f"[ReplayConfiguration.dump] model inputs {input_names}")
        folder = os.path.join(self.dump_folder, name.replace(":", "_").replace("/", "_"))
        os.makedirs(folder, exist_ok=True)
        if verbose:
            print(f"[ReplayConfiguration.dump] dumps into folder {folder!r}")
        onnx.save(submodel, os.path.join(folder, "model.onnx"))
        torch_inputs = {}
        for n in input_names:
            if n in onnx_name_to_ep_name:
                torch_inputs[n] = torch_results[onnx_name_to_ep_name[n]]
            else:
                # It is possible that this result only exists in the onnx worlds.
                pass
        onnx_inputs = {n: onnx_results[n] for n in input_names}
        assert (
            name in onnx_name_to_ep_name
        ), f"Unable to find {name!r} in {onnx_name_to_ep_name}"
        expected_outputs_and_mapping = dict(
            expected=(torch_results[onnx_name_to_ep_name[name]],),
            mapping={
                k: onnx_name_to_ep_name[k] for k in input_names if k in onnx_name_to_ep_name
            },
        )
        torch.save(torch_inputs, os.path.join(folder, "torch_inputs.pt"))
        torch.save(onnx_inputs, os.path.join(folder, "onnx_inputs.pt"))
        torch.save(
            expected_outputs_and_mapping, os.path.join(folder, "torch_outputs_and_mapping.pt")
        )
        if verbose:
            print(f"[ReplayConfiguration.dump] done {folder!r}")
        return folder


@dataclass
class RunAlignedRecord:
    """
    The side-by-side ran by function :func:`run_aligned
    <onnx_diagnostic.torch_onnx.sbs.run_aligned>`
    yields instances of this type. If both `ep_name`
    and `onnx_name` are specified, then both results
    appear in the exported program (torch) and the onnx model.

    :param ep_id_node: node index in the exported program
    :param onnx_id_node: node index in the onnx model, -1 for an initializer
    :param ep_name: result name in the exported program
    :param onnx_name: result name in the onnx model, usually same as `ep_name`
        except for initializer
    :param ep_target: target name in the exported program producing the result
    :param onnx_op_type: operator type in the onnx model producing the result
    :param onnx_id_output: usually 0 unless this node has multiple output,
        in that case, it is the output index
    :param ep_shape_type: shape and type of the results in the exported program
    :param onnx_shape_type: shape and type of the results in the onnx mode,
        it should be the same as `ep_shape_type`, anything different probably
        means a bug
    :param err_abs: maximum absolute error for the considered result
        between the exported program and the onnx model
    :param err_rel: maximum relative error
    :param err_dev: 0 if the device is the same, 1 if not
    :param err_nan: number of nan values disagreeing
    :param err_h01: number of values for which the discrepancy is above 0.1
    :param ep_time_run: execution time for the exported program
    :param onnx_time_run: execution time for the onnx model, that includes
        the creation of the onnx model so that's probably not very usable
    """

    ep_id_node: Optional[int] = None
    onnx_id_node: Optional[int] = None
    ep_name: Optional[str] = None
    onnx_name: Optional[str] = None
    ep_target: Optional[str] = None
    onnx_op_type: Optional[str] = None
    onnx_id_output: Optional[int] = None
    ep_shape_type: Optional[str] = None
    onnx_shape_type: Optional[str] = None
    err_abs: Optional[float] = None
    err_rel: Optional[float] = None
    err_dev: Optional[float] = None
    err_nan: Optional[float] = None
    err_h01: Optional[float] = None
    ep_time_run: Optional[float] = None
    onnx_time_run: Optional[float] = None

    def __post_init__(self):
        "Validation."
        assert self.ep_id_node is None or self.ep_id_node >= 0, (
            f"Node id are always positive in the exported program but "
            f"ep_id_node={self.ep_id_node}"
        )

    def set_diff(self, diff: Dict[str, Any]):
        """Sets error."""
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
        if "rep" in diff:
            self.err_h01 = diff["rep"][">0.1"]
        return self

    @property
    def key(
        self,
    ) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[str], Optional[str]]:
        "Creates a unique identifier."
        return (
            self.ep_id_node,
            self.onnx_id_node,
            self.onnx_id_output,
            self.ep_name,
            self.onnx_name,
        )

    def check(
        self,
        already_yielded: Dict[
            Tuple[Optional[int], Optional[int], Optional[int], Optional[str], Optional[str]],
            int,
        ],
    ) -> Self:
        "Checks a record was not already yielded."
        if self.onnx_op_type == "reset":
            # no record for this one
            return self
        key = self.key
        assert key not in already_yielded, (
            f"Record with key={key} was already yielded, "
            f"number of records={len(already_yielded)} and previous "
            f"record at position {already_yielded[key]} (self={self})"
        )
        already_yielded[key] = len(already_yielded)
        return self


@dataclass
class StatusRunAligned:
    """
    Information to display while running the side-by-side

    :param max_abs: maximum absolute seen so far
    :param n_inf: number of infinite values seen so far
    :param n_nan: number of nan values seen so for
    :param yielded_nodes: number of yielded pair of nodes seen so far
    :param last_replay: last result dumped on disk for later replay
    """

    max_abs: float = 0.0
    n_inf: int = 0
    n_nan: int = 0
    yielded_nodes: int = 0
    last_replay: str = ""

    def to_str(self) -> str:
        "Nice display."
        s = (
            f"yielded={self.yielded_nodes} maxabs={self.max_abs:1.3f} "
            f"#inf={self.n_inf} #nan={self.n_nan}"
        )
        if self.last_replay:
            return f"{s} -PLAY({self.last_replay})"
        return s

    def update(self, err_abs: float):
        "Updates all attributes with the latest measure."
        if np.isinf(err_abs) or np.isnan(err_abs):
            self.n_inf += 1
        elif err_abs > 1e6:
            self.n_nan += 1
        else:
            self.max_abs = max(self.max_abs, err_abs)


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
    already_yielded = {}  # type: ignore[var-annotated]
    reset_names = set(reset_names) if reset_names else set()  # type: ignore[assignment]
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
            d = max_diff(torch_results[to], r, hist=[0.1])
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
                ep_name=to,
                onnx_name=o,
                ep_shape_type=string_type(torch_results[to], **str_kws),
                onnx_shape_type=string_type(r, **str_kws),
            )
            r.set_diff(d)
            mapping_onnx_to_torch[to] = o
            return r
        return None

    def _loop_onnx_node(
        onx,
        ep_graph_nodes,
        onnx_results,
        mapping_onnx_to_torch,
        torch_results,
        ep_durations,
        use_tensor,
        i,
        i_onnx,
        name_to_ep_node,
        run_cls_kwargs,
        str_kws,
        status,
        already_run,
        torch_names_to_onnx_names,
        verbose,
    ):

        if i_onnx in already_run:
            yield None
        node = onx.graph.node[i_onnx]
        if verbose > 1:
            print(
                f"[run_aligned] run onx.graph.node[{i_onnx}]: "
                f"{node.op_type}({', '.join(node.input)}) -> {', '.join(node.output)}"
            )
        elif verbose == 1:
            loop.set_description(
                f"ep {i}/{len(ep_graph_nodes)} nx {i_onnx}/{len(onx.graph.node)} "
                f"{status.to_str()}"
            )

        ref = run_cls(node, **run_cls_kwargs)
        # We need to clone because the runtime maybe using dlpack to create OrtValue
        feeds = (
            {k: onnx_results[k].clone() for k in node.input if k}
            if use_tensor
            else {k: onnx_results[k].copy() for k in node.input if k}
        )
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
            if r is None or not o:
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
                status.yielded_nodes += 1
                if tmp.err_abs is not None:
                    status.update(tmp.err_abs)
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
                        r,
                        verbose,
                        atol,
                        rtol,
                        i,
                        i_onnx,
                    )
                    if tmp is not None:
                        tmp.onnx_op_type = "reset"
                        tmp.onnx_id_output = list_node_output.index(o)
                        status.yielded_nodes += 1
                        yield tmp
        already_run.add(i_onnx)

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

    # preparation with ep.graph.nodes
    ep_state_dict = {**ep.state_dict, **dict(ep.named_buffers(), **ep.tensor_constants)}
    placeholders_to_state_dict = {
        **{f"p_{name.replace('.', '_')}": name for name in ep.state_dict},
        **{f"b_{name.replace('.', '_')}": name for name, _ in ep.named_buffers()},
        **{f"c_{name.replace('.', '_')}": name for name in ep.tensor_constants},
    }
    skip_mapping_torch_onnx = _duplicated_values(placeholders_to_state_dict)
    placeholders = {}
    if verbose:
        print(f"[run_aligned] ep: model has {len(ep_state_dict)} torch constants or weights.")

    if verbose:
        print(f"[run_aligned] ep: walks through {len(ep.graph.nodes)} nodes from torch")
    positions: Dict[str, Any] = {}
    ep_graph_nodes = list(ep.graph.nodes)
    torch_results: Dict[str, Any] = {}
    last_position = 0
    torch_output_names = None
    torch_input_names: List[str] = []
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
    for init in onx.graph.initializer:  # type: ignore
        positions[init.name] = -1
        t = None
        if init.name in torch_results:
            if init.name not in skip_mapping_torch_onnx:
                t = torch_results[init.name]
                torch_names_to_onnx_names[init.name] = init.name
        elif init.name not in skip_onnx_name and init.name in rev_init_aliases:
            new_names = [  # type: ignore[assignment]
                k
                for k in rev_init_aliases[init.name]
                if k in torch_results and k not in skip_mapping_torch_onnx
            ]
            if new_names and len(new_names) == 1:
                new_name = new_names[0]  # type: ignore[assignment, index]
                t = torch_results[new_name]
                if (
                    t.shape == tuple(init.dims)
                    and torch_dtype_to_onnx_dtype(t.dtype) == init.data_type
                ):
                    torch_names_to_onnx_names[new_name] = init.name

                # We should check tensors and proto are the same.
        if t is None:
            t = to_tensor(init)
            if default_device and t.numel() >= 1024:
                # Let's force its way to cuda (should check the device has well).
                t = t.to(default_device)
            yield RunAlignedRecord(
                onnx_id_node=-1,
                onnx_name=init.name,
                onnx_op_type="initializer",
                onnx_shape_type=string_type(t, **str_kws),
            ).check(already_yielded)

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
    already_run: Set[int] = set()
    ep_durations = {}
    status = StatusRunAligned()
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
                f"{status.to_str()}"
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
                t = torch_results[node.name]
                if verbose > 1:
                    print(f"[run_aligned-ep] =ags: {node.name}={string_type(t, **str_kws)}")
                # Otherwise, it is an input.
                record = RunAlignedRecord(
                    ep_id_node=i,
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
                        ep_id_node=i,
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
                                hist=[0.1],
                            )
                        )
                    yield record.check(already_yielded)
                else:
                    if verbose > 1:
                        print(
                            f"[run_aligned-ep] +plh: {node.name}={string_type(t, **str_kws)}"
                        )
                    yield RunAlignedRecord(
                        ep_id_node=i,
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
            for r in _loop_onnx_node(
                onx,
                ep_graph_nodes,
                onnx_results,
                mapping_onnx_to_torch,
                torch_results,
                ep_durations,
                use_tensor,
                i,
                i_onnx,
                name_to_ep_node,
                run_cls_kwargs,
                str_kws,
                status,
                already_run,
                torch_names_to_onnx_names,
                verbose,
            ):
                if r:
                    yield r.check(already_yielded)

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
        for r in _loop_onnx_node(
            onx,
            ep_graph_nodes,
            onnx_results,
            mapping_onnx_to_torch,
            torch_results,
            ep_durations,
            use_tensor,
            i,
            i_onnx,
            name_to_ep_node,
            run_cls_kwargs,
            str_kws,
            status,
            already_run,
            torch_names_to_onnx_names,
            verbose,
        ):
            if r:
                yield r.check(already_yielded)

    if verbose:
        print(f"[run_aligned] done with status={status.to_str()}")
