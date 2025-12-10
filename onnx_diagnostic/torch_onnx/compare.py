import enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import onnx
from ..helpers.onnx_helper import onnx_dtype_name


_NOT_SO_FAR_OPS = [
    {"MatMul", "Gemm", "FusedMatMul"},
    {"Conv", "FusedConv"},
    {"MaxPool"},
]


def _sum_sets(sets):
    t = set()
    for s in sets:
        t |= s
    return t


_ALL_NOT_SO_FAR_OPS = _sum_sets(_NOT_SO_FAR_OPS)


def _align(res: str, limit: int) -> str:
    if len(res) == limit:
        return res
    if len(res) > limit:
        return res[:limit]
    return res + " " * (limit - len(res))


class ObsType(enum.IntEnum):
    """Observation kind."""

    RESULT = 1
    INITIALIZER = 2
    SPARSE_INITIALIZER = 4
    INPUT = 8
    OUTPUT = 16
    NODE = 32

    def __repr__(self):
        return f"{self.__class__.__name__}.{self._name_}"


@dataclass
class ObsCompare:
    """
    The description of an observation, a node, an input, an output, an initializer.

    :param position: index of this observation in the original model
    :param kind: node type, see :class:`ObsType`
    :param name_or_outputs: name of an initializer or the outputs of a node
    :param itype: onnx type
    :param index: index of an input or output
    :param shape: shape
    :param op_type: node op_type
    :param comment: comment, unused
    """

    position: int
    kind: ObsType
    name_or_outputs: Tuple[str]
    itype: int = 0
    index: int = 0
    shape: Optional[Tuple[Tuple[Union[int, str], ...]]] = None
    op_type: str = ""
    comment: str = ""

    def __str__(self) -> str:
        "usual"
        els = [
            _align(f"{self.position:04d}", 4),
            _align(self.kind._name_, 6),
            _align(onnx_dtype_name(self.itype) if self.itype else "?", 8),
            _align("?" if self.shape is None else "x".join(map(str, self.shape)), 18),
            _align(self.op_type or "", 15),
            _align(", ".join(self.name_or_outputs), 35),
        ]
        return " ".join(els)

    @classmethod
    def to_str(cls, obs: Optional["ObsCompare"]) -> str:
        assert not obs or isinstance(obs, ObsCompare), f"unexpected type {type(obs)}"
        if obs:
            return str(obs)
        return " " * (4 + 6 + 8 + 18 + 15 + 35 + 5)

    def distance(self, obs: "ObsCompare") -> float:
        """Computes a cost between two observations."""
        if self.kind != obs.kind:
            return 1e6
        d: float = 0
        if self.itype != obs.itype:
            d += 1e5
        if self.kind == ObsType.NODE:
            cost = 9997
            d = 0
            if self.op_type != obs.op_type:
                if self.op_type in _ALL_NOT_SO_FAR_OPS or obs.op_type in _ALL_NOT_SO_FAR_OPS:
                    d += 1e2
                    for aset in _NOT_SO_FAR_OPS:
                        if self.op_type in aset and obs.op_type in aset:
                            cost = 97
                        elif self.op_type in aset or obs.op_type in aset:
                            d += 5e4
                else:
                    d += 9e2
            if len(self.name_or_outputs) == 1 and len(obs.name_or_outputs) == 1:
                if self.name_or_outputs[0] != obs.name_or_outputs[0]:
                    n1 = self.name_or_outputs[0]
                    n2 = obs.name_or_outputs[0]
                    n1 = n1.replace("_", "")
                    n2 = n2.replace("_", "")
                    if n1 == n2:
                        d += 1
                    elif (n1.startswith(("val_", "_onx_")) or "::" in n1 or "--" in n1) and (
                        n2.startswith(("val_", "_onx_")) or "::" in n2 or "--" in n2
                    ):
                        # These are name given the exporter
                        # and not inspired from the model itself.
                        d += cost / 100
                    else:
                        d += cost
            else:
                a = set(self.name_or_outputs) & set(obs.name_or_outputs)
                b = set(self.name_or_outputs) | set(obs.name_or_outputs)
                d += cost * (len(b) - len(a))
            return d
        if self.kind == ObsType.INPUT:
            return (
                999.7
                if self.itype != obs.itype
                or self.shape != obs.shape
                or self.index != obs.index
                else 0
            )
        if self.kind == ObsType.INITIALIZER or self.kind == ObsType.SPARSE_INITIALIZER:
            return 1e3 if self.itype != obs.itype or self.shape != obs.shape else 0
        if self.kind == ObsType.OUTPUT:
            return (
                999.1
                if self.itype != obs.itype
                or self.shape != obs.shape
                or self.index != obs.index
                else 0
            )
        return 1e8

    @classmethod
    def obs_sequence_from_model(
        cls,
        model: Union[onnx.ModelProto, onnx.GraphProto],
    ) -> List["ObsCompare"]:
        """
        Creates a sequence of observations bases on a model.

        :param model: model
        :return: sequence of observations
        """
        graph = model if isinstance(model, onnx.GraphProto) else model.graph

        shapes = {}
        types = {}
        for info in [*graph.value_info, *graph.input, *graph.output]:
            if info.type.tensor_type:
                t = info.type.tensor_type
                shapes[info.name] = tuple((d.dim_param or d.dim_value) for d in t.shape.dim)
                types[info.name] = t.elem_type

        seq = []
        for init in graph.initializer:
            obs = ObsCompare(
                position=len(seq),
                kind=ObsType.INITIALIZER,
                itype=init.data_type,
                shape=tuple(init.dims),
                name_or_outputs=(init.name,),
            )
            seq.append(obs)
        for i, inp in enumerate(graph.input):
            obs = ObsCompare(
                position=len(seq),
                kind=ObsType.INPUT,
                itype=inp.type.tensor_type.elem_type,
                index=i,
                shape=tuple(
                    (d.dim_param or d.dim_value) for d in inp.type.tensor_type.shape.dim
                ),
                name_or_outputs=(inp.name,),
            )
            seq.append(obs)
        for node in graph.node:
            obs = ObsCompare(
                position=len(seq),
                kind=ObsType.NODE,
                itype=types.get(node.output[0], 0),
                index=i,
                shape=shapes.get(node.output[0], None),
                name_or_outputs=tuple(node.output),
                op_type=node.op_type,
            )
            seq.append(obs)
        for i, inp in enumerate(graph.output):
            obs = ObsCompare(
                position=len(seq),
                kind=ObsType.OUTPUT,
                itype=inp.type.tensor_type.elem_type,
                index=i,
                shape=tuple(
                    (d.dim_param or d.dim_value) for d in inp.type.tensor_type.shape.dim
                ),
                name_or_outputs=(inp.name,),
            )
            seq.append(obs)
        return seq


@dataclass
class ObsComparePair:
    """
    Defines a pair of comparison objects

    :param side1: object from first side
    :param side2: object from first side
    :param distance: distance
    """

    side1: Optional[ObsCompare]
    side2: Optional[ObsCompare]
    distance: float

    def __str__(self) -> str:
        "nice display"
        return (
            f"{self.distance:.4e} | "
            f"{ObsCompare.to_str(self.side1)} | {ObsCompare.to_str(self.side2)}"
        )

    @classmethod
    def to_str(cls, seq: List["ObsComparePair"]) -> str:
        """Displays every pair in text."""
        return "\n".join([f"{str(pair)}" for pair in seq])

    @classmethod
    def distance_sequence(cls, s1: List["ObsCompare"], s2: List["ObsCompare"]) -> Tuple[
        float,
        List[Tuple[int, int]],
        List["ObsComparePair"],
    ]:
        """
        Computes the distance between two sequences of results.

        :param s1: first sequence
        :param s2: second sequence
        :return: distance and alignment

        An example:

        .. runpython::
            :showcode:

            import torch
            from onnx_diagnostic.export.api import to_onnx
            from onnx_diagnostic.torch_onnx.compare import ObsComparePair, ObsCompare


            class Model(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.conv1 = torch.nn.Conv2d(3, 16, 5)
                    self.fc1 = torch.nn.Linear(144, 64)
                    self.fc2 = torch.nn.Linear(64, 128)
                    self.fc3 = torch.nn.Linear(128, 10)

                def forward(self, x):
                    x = torch.nn.functional.max_pool2d(
                        torch.nn.functional.relu(self.conv1(x)),
                        (4, 4),
                    )
                    # x = F.max_pool2d(F.relu(self.conv2(x)), 2)
                    x = torch.flatten(x, 1)
                    x = torch.nn.functional.relu(self.fc1(x))
                    x = torch.nn.functional.relu(self.fc2(x))
                    y = self.fc3(x)
                    return y


            model = Model()
            x = torch.randn((2, 3, 16, 17), dtype=torch.float32)
            dynamic_shapes = ({0: "batch", 3: "dim"},)
            onnx_optimized = to_onnx(
                model, (x,), dynamic_shapes=dynamic_shapes, exporter="custom", optimize=True
            ).model_proto
            onnx_not_optimized = to_onnx(
                model, (x,), dynamic_shapes=dynamic_shapes, exporter="custom", optimize=False
            ).model_proto
            seq1 = ObsCompare.obs_sequence_from_model(onnx_not_optimized)
            seq2 = ObsCompare.obs_sequence_from_model(onnx_optimized)
            _dist, _path, pair_cmp = ObsComparePair.distance_sequence(seq1, seq2)
            text = ObsComparePair.to_str(pair_cmp)
            print(text)
        """
        delay = max(50, abs(len(s2) - len(s1)) + 1)
        distance: Dict[Tuple[int, int], Union[int, float]] = {(-1, -1): 0}
        predecessor: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {(-1, -1): None}
        insert_cost = 1e3
        for i in range(len(s1)):
            for j in range(max(0, i - delay), min(len(s2), i + delay)):
                best = distance.get((i, j), 1e100)
                pred = None
                ki, kj = i - 1, j - 1
                if (ki, kj) in distance:
                    d = distance[ki, kj] + s1[i].distance(s2[j])
                    if d < best:
                        best = d
                        pred = (ki, kj)
                ki, kj = i - 1, j
                if (ki, kj) in distance:
                    d = distance[ki, kj] + insert_cost + 1
                    if d < best:
                        best = d
                        pred = (ki, kj)
                ki, kj = i, j - 1
                if (ki, kj) in distance:
                    d = distance[ki, kj] + insert_cost + 0.1
                    if d < best:
                        best = d
                        pred = (ki, kj)
                distance[i, j] = best
                predecessor[i, j] = pred

        # reverse
        way = []
        last: Optional[Tuple[int, int]] = len(s1) - 1, len(s2) - 1
        while last is not None:
            way.append(last)
            last = predecessor[last]
        indices = list(reversed(way))[1:]
        obs_path: List[ObsComparePair] = []
        last = -1, -1
        for i, j in indices:
            di = i - last[0]
            dj = j - last[1]
            cost = distance.get((i, j), np.nan)
            if di == dj == 1:
                obs_path.append(ObsComparePair(s1[i], s2[j], distance=cost))
            elif di == 0:
                obs_path.append(ObsComparePair(None, s2[j], distance=cost))
            elif dj == 0:
                obs_path.append(ObsComparePair(s1[i], None, distance=cost))
            else:
                raise RuntimeError(f"issue with di={di}, dj={dj}")
            last = i, j
        return distance[len(s1) - 1, len(s2) - 1], indices, obs_path
