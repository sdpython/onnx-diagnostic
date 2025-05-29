import enum
from typing import Any, Dict, Optional, Set, Tuple, Union
import onnx
import torch
from ..helpers import string_type


class RuntimeValueKind(enum.IntEnum):
    INITIALIZER = 1
    INPUT = 2
    OUTPUT = 4
    RESULT = 12


class RuntimeValue:
    """Describes a value used during the execution of a model."""

    def __init__(
        self,
        name: str,
        dtype: Optional[Any] = None,
        shape: Optional[Tuple[Union[str, int], ...]] = None,
        value: Optional[torch.Tensor] = None,
        first_used: Optional[int] = None,
        last_used: Optional[int] = None,
        created: Optional[int] = None,
        is_shape: Optional[bool] = None,
        kind: Optional[RuntimeValueKind] = None,
    ):
        self.name = name
        self.dtype = dtype
        self.shape = shape
        self.value = value
        self.first_used = first_used
        self.last_used = last_used
        self.created = created
        self.is_shape = is_shape
        self.kind = kind

    def __repr__(self) -> str:
        "usual"
        ad = {}
        for att in [
            "name",
            "dtype",
            "shape",
            "first_used",
            "last_used",
            "is_shape",
            "kind",
            "created",
        ]:
            v = getattr(self, att)
            if v is not None:
                ad[att] = v
        if self.value is not None:
            ad["value"] = string_type(self.value, with_shape=True)
        msg = ", ".join(f"{name}={t}" for name, t in ad.items())
        return f"{self.__class__.__name__}({msg})"


def get_hidden_inputs(graph: onnx.GraphProto) -> Set[str]:
    """
    Returns the hidden inputs (inputs coming from an upper context)
    used by a subgraph.
    """
    hidden = set()
    memo = set(i.name for i in graph.initializer)
    memo |= set(i.name for i in graph.sparse_initializer)
    for node in graph.node:
        for i in node.input:
            if i not in memo:
                hidden.add(i)
        for att in node.attribute:
            if att.type == onnx.AttributeProto.GRAPH and att.g:
                hid = get_hidden_inputs(att.g)
                less = set(h for h in hid if h not in memo)
                hidden |= less
        memo |= set(node.output)
    return hidden


def first_used_last_used(proto: onnx.ModelProto) -> Dict[str, RuntimeValue]:
    """
    Builds first used, last used informations for every result
    in the model.

    :param proto: model
    :return: dictionary of RuntimeValue
    """
    values = {}
    for init in proto.graph.initializer:
        values[init.name] = RuntimeValue(
            init.name, kind=RuntimeValueKind.INITIALIZER, created=-1
        )
    for init in proto.graph.sparse_initializer:
        values[init.name] = RuntimeValue(
            init.name, created=-1, kind=RuntimeValueKind.INITIALIZER
        )
    for inp in proto.graph.input:
        values[inp.name] = RuntimeValue(inp.name, created=-1, kind=RuntimeValueKind.INPUT)
    for it, node in enumerate(proto.graph.node):
        for i in node.input:
            if values[i].first_used is None:
                values[i].first_used = it
            values[i].last_used = it
        for att in node.attribute:
            if att.type == onnx.AttributeProto.GRAPH:
                for n in get_hidden_inputs(att.g):
                    if values[n].first_used is None:
                        values[n].first_used = it
                    values[n].last_used = it
        for o in node.output:
            values[o] = RuntimeValue(o, created=it, kind=RuntimeValueKind.RESULT)
    for out in proto.graph.output:
        values[out.name].kind = RuntimeValueKind.OUTPUT
        values[out.name].last_used = len(proto.graph.node)
    return values
