import os
import textwrap
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from typing import Self
except ImportError:
    # python <= 3.10
    Self = "Self"  # type: ignore[assignment]
import onnx
import numpy as np
import torch
from ..helpers.onnx_helper import extract_subset_of_nodes, make_submodel, from_array_extended
from ..helpers.torch_helper import torch_dtype_to_onnx_dtype


def make_torch_inputs(
    input_names: List[str],
    onnx_name_to_ep_name: Dict[str, str],
    onnx_results: Dict[str, torch.Tensor],
    torch_results: Dict[str, torch.Tensor],
    submodel: Optional[onnx.ModelProto],
) -> Tuple[List[torch.Tensor], Set[str]]:
    """
    Gathers torch tensors instead of onnx tensors (tensors produced by the onnx model)

    :param input_names: tensors to gather
    :param onnx_name_to_ep_name: mapping between onnx name to names in the exported program
    :param onnx_results: all onnx results (produced by the onnx model)
    :param torch_results: all tensors produced by the exported program
    :param submodel: onnx model, any tensor missing in `torch_results` is
        add as an initializer to this model
    :return: the list of tensors, the set of inputs for which there was no tensor coming
        from the exported program
    """
    torch_inputs = {}
    removed_inputs = set()
    for n in input_names:
        if n in onnx_name_to_ep_name:
            torch_inputs[n] = torch_results[onnx_name_to_ep_name[n]]
        else:
            removed_inputs.add(n)
            if submodel is not None:
                # We add that input as an initializer because it is probably a constant.
                submodel.graph.initializer.append(from_array_extended(onnx_results[n], name=n))
            else:
                torch_inputs[n] = onnx_results[n]
    return torch_inputs, removed_inputs


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

    def get_replay_code(self) -> str:
        """
        Returns a code letting the user replay the onnx model.
        It looks like the following. It may have to be adapted.

        .. runpython::
            :showcode:

            from onnx_diagnostic.torch_onnx.sbs_dataclasses import ReplayConfiguration

            rc = ReplayConfiguration(dump_folder="unused")
            print(rc.get_replay_code())
        """
        return textwrap.dedent(
            """
            import onnx
            import torch
            from onnx_diagnostic.helpers import max_diff, string_diff, string_type
            from onnx_diagnostic.helpers.torch_helper import study_discrepancies
            from onnx_diagnostic.helpers.onnx_helper import pretty_onnx
            from onnx_diagnostic.reference import OnnxruntimeEvaluator

            skws = dict(with_shape=True, with_device=True)

            torch_inputs = torch.load("torch_inputs.pt")
            onnx_inputs = torch.load("onnx_inputs.pt")
            expected_outputs_and_mapping = torch.load("torch_outputs_and_mapping.pt")
            expected = expected_outputs_and_mapping["expected"]
            mapping = expected_outputs_and_mapping["mapping"]

            print(f"-- torch_inputs={string_type(torch_inputs, **skws)}")
            print(f"-- onnx_inputs={string_type(onnx_inputs, **skws)}")
            print(f"-- expected={string_type(expected, **skws)}")
            print(f"-- mapping={mapping}")

            print()
            print("-- model.onnx")
            print()

            model = onnx.load("model.onnx")
            print(pretty_onnx(model))

            print()
            print("-- range of inputs --")
            print()

            for k, v in onnx_inputs.items():
                print(f"--   {k}: {string_type(v, **skws, with_min_max=True)}")

            print()
            print("-- discrepancies of inputs --")
            print()

            ep_feeds = {}
            for k, v in onnx_inputs.items():
                tk = mapping.get(k, k)
                tkv = torch_inputs[k] if k in torch_inputs else torch_inputs[tk]
                ep_feeds[k] = tkv
                diff = max_diff(v, tkv)
                print(
                    f"--   {k} -> {tk} ep:{string_type(tkv, **skws)} "
                    f"nx:{string_type(v, **skws)} / diff {string_diff(diff)}"
                )

            print()
            print("-- SVD --")
            print()

            for k, v in onnx_inputs.items():
                if len(v.shape) == 2:
                    U, S, Vt = torch.linalg.svd(v.to(torch.float32))
                    print(f" -- {k}: {S[:5]}")

            print()
            print("-- run with onnx_inputs --")
            print()

            sess = OnnxruntimeEvaluator(model, whole=True)
            feeds = onnx_inputs
            obtained = sess.run(None, feeds)
            print(f"-- obtained={string_type(obtained, **skws)}")
            diff = max_diff(expected, tuple(obtained), hist=[0.1, 0.01])
            print(f"-- diff: {string_diff(diff)}")
            print()
            print("-- plots --")

            for i in range(len(expected)):
                study_discrepancies(
                    expected[i],
                    obtained[i],
                    title=f"study output {i}",
                    name=f"disc{i}.png",
                    bins=50,
                )

            print()
            print("-- run with torch_inputs --")
            print()

            obtained = sess.run(None, ep_feeds)
            print(f"-- obtained={string_type(obtained, **skws)}")
            diff = max_diff(expected, tuple(obtained), hist=[0.1, 0.01])
            print(f"-- diff: {string_diff(diff)}")

            print()
            print("-- end --")
            print()
            """
        )

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

        torch_inputs, removed_inputs = make_torch_inputs(
            input_names, onnx_name_to_ep_name, onnx_results, torch_results, submodel
        )

        if removed_inputs:
            input_names = [i for i in input_names if i not in removed_inputs]
            new_inputs = [i for i in submodel.graph.input if i.name not in removed_inputs]
            del submodel.graph.input[:]
            submodel.graph.input.extend(new_inputs)
            if verbose:
                print(f"[ReplayConfiguration.dump] removed input {removed_inputs}")
                print(f"[ReplayConfiguration.dump] final model inputs {input_names}")

        onnx.save(submodel, os.path.join(folder, "model.onnx"))
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
        with open(os.path.join(folder, "replay.py"), "w") as f:
            f.write(self.get_replay_code())
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
    err_abs2: Optional[float] = None
    err_rel2: Optional[float] = None
    err_dev2: Optional[float] = None
    err_nan2: Optional[float] = None
    err_h012: Optional[float] = None

    def __post_init__(self):
        "Validation."
        assert self.ep_id_node is None or self.ep_id_node >= 0, (
            f"Node id are always positive in the exported program but "
            f"ep_id_node={self.ep_id_node}"
        )

    def set_diff(self, diff: Dict[str, Any]) -> Self:
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

    def set_diff2(self, diff: Dict[str, Any]) -> Self:
        """Sets error."""
        if diff is None:
            return
        if "abs" in diff:
            self.err_abs2 = diff["abs"]
        if "rel" in diff:
            self.err_rel2 = diff["rel"]
        if "dev" in diff:
            self.err_dev2 = diff["dev"]
        if "nan" in diff:
            self.err_nan2 = diff["nan"]
        if "rep" in diff:
            self.err_h012 = diff["rep"][">0.1"]
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
