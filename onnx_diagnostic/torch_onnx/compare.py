import enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import onnx
from ..helpers.onnx_helper import onnx_dtype_name


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

    :param kind: node type, see :class:`ObsType`
    :param name_or_outputs: name of an initializer or the outputs of a node
    :param itype: onnx type
    :param index: index of an input or output
    :param shape: shape
    :param op_type: node op_type
    :param comment: comment, unused
    """

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
            _align(self.kind._name_, 6),
            _align(onnx_dtype_name(self.itype) if self.itype else "?", 8),
            _align("?" if self.shape is None else "x".join(map(str, self.shape)), 18),
            _align(self.op_type or "", 15),
            _align(", ".join(self.name_or_outputs), 35),
        ]
        return " ".join(els)

    def distance(self, obs: "ObsCompare") -> float:
        """Computes a cost between two observations."""
        if self.kind != obs.kind:
            return 1e6
        if self.itype != obs.itype:
            return 1e5
        if self.kind == ObsType.NODE:
            if self.op_type != obs.op_type:
                return 1e4
            if len(self.name_or_outputs) == 1:
                return 0 if self.name_or_outputs == obs.name_or_outputs else 1e2
            a = set(self.name_or_outputs) & set(obs.name_or_outputs)
            b = set(self.name_or_outputs) | set(obs.name_or_outputs)
            return 1e2 * (len(b) - len(a))
        if self.kind == ObsType.INPUT:
            return (
                999.7
                if self.itype != obs.itype
                or self.shape != obs.shape
                or self.index != obs.index
                else 0
            )
        if self.kind == ObsType.INITIALIZER or self.kind == ObsType.SPARSE_INITIALIZER:
            return 1e3 if self.itype != obs.itype or self.shape or obs.shape else 0
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
    def distance_sequence(cls, s1: List["ObsCompare"], s2: List["ObsCompare"]) -> Tuple[
        float,
        List[Tuple[int, int]],
        List[Tuple[Optional["ObsCompare"], Optional["ObsCompare"]]],
    ]:
        """
        Computes the distance between two sequences of results.

        :param s1: first sequence
        :param s2: second sequence
        :return: distance and alignment
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
        last: Optional[Tuple[Optional[int], Optional[int]]] = len(s1) - 1, len(s2) - 1
        while last is not None:
            way.append(last)
            last = predecessor[last]
        indices = list(reversed(way))[1:]
        obs_path: List[Optional[Tuple[int, int]]] = []
        last = -1, -1
        for i, j in indices:
            di = i - last[0]
            dj = j - last[1]
            if di == dj == 1:
                obs_path.append((s1[i], s2[j]))
            elif di == 0:
                obs_path.append((None, s2[j]))
            elif dj == 0:
                obs_path.append((s1[i], None))
            else:
                raise RuntimeError(f"issue with di={di}, dj={dj}")
            last = i, j
        return distance[len(s1) - 1, len(s2) - 1], indices, obs_path

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
        for info in graph.value_info:
            if info.type.tensor_type:
                t = info.type.tensor_type
                shapes[info.name] = tuple((d.dim_param or d.dim_value) for d in t.shape.dim)
                types[info.name] = t.elem_type

        seq = []
        for init in graph.initializer:
            obs = ObsCompare(
                kind=ObsType.INITIALIZER,
                itype=init.data_type,
                name_or_outputs=(init.name,),
            )
            seq.append(obs)
        for i, inp in enumerate(graph.input):
            obs = ObsCompare(
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
