import enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import onnx
import torch
from ..helpers import string_type


class RuntimeValueKind(enum.IntEnum):
    "Kind of result."

    RESULT = 1
    INITIALIZER = 3
    INPUT = 5
    OUTPUT = 9

    def to_str(self) -> str:
        for k, v in self.__class__.__dict__.items():
            if v == int(self):
                return k
        raise RuntimeError(f"Unable to display {self!r}")


class RuntimeDevice(enum.IntEnum):
    "Device definition"

    UNKNOWN = 0
    NEW = 1
    CPU = 2
    CUDA = 4

    def to_str(self) -> str:
        for k, v in self.__class__.__dict__.items():
            if v == int(self):
                return k
        raise RuntimeError(f"Unable to display {self!r}")


class RuntimeValue:
    """Describes a value used during the execution of a model."""

    def __init__(
        self,
        name: str,
        dtype: Optional[Any] = None,
        shape: Optional[Tuple[Union[str, int], ...]] = None,
        value: Optional[Any] = None,
        first_used: Optional[int] = None,
        last_used: Optional[int] = None,
        created: Optional[int] = None,
        is_shape: Optional[bool] = None,
        kind: Optional[RuntimeValueKind] = None,
        device: Optional[RuntimeDevice] = None,
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
        self.device = device

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
            "device",
        ]:
            v = getattr(self, att)
            if v is not None:
                ad[att] = v
        if self.value is not None:
            ad["value"] = string_type(self.value, with_shape=True)
        msg = ", ".join(
            f"{name}={t.to_str()}" if hasattr(t, "to_str") else f"{name}={t}"
            for name, t in ad.items()
        )
        return f"{self.__class__.__name__}({msg})"

    @property
    def has_value(self) -> bool:
        "Tells if value is specified."
        return self.value is not None

    def set_value(self, value: torch.Tensor):
        """Sets the value."""
        assert value is not None, "Use clean_value to set a value to None"
        self.value = value
        if self.dtype:
            assert (
                self.dtype == value.dtype
            ), f"Unexpected dtype={value.dtype}, previous dtype was {self.dtype}"
        else:
            self.dtype = value.dtype
        self.shape = tuple(map(int, value.shape))

    def clean_value(self):
        """Sets value to None."""
        self.value = None

    @property
    def is_output(self) -> bool:
        "Tells if it is an output."
        return self.kind == RuntimeValueKind.OUTPUT

    @property
    def is_input(self) -> bool:
        "Tells if it is an input."
        return self.kind == RuntimeValueKind.INPUT

    @property
    def is_initializer(self) -> bool:
        "Tells if it is an initializer."
        return self.kind == RuntimeValueKind.INITIALIZER


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


def set_is_shape(
    node: onnx.NodeProto, values: Dict[str, RuntimeValue], drop: Optional[Set[str]] = None
) -> List[str]:
    """
    Sets attribute ``is_shape`` for outputs of a node.

    :param node: node to process
    :param values: stored results, values in this dictionary are updated
    :param drop: variables not to consider because the come from the graph
        holding this subgraph
    :return: list of modified results
    """
    if not node.input:
        # Constant
        return []
    drop = drop or set()
    if node.op_type in ("Shape", "Size") and node.domain == "":
        values[node.output[0]].is_shape = True
        return [node.output[0]]
    is_shapes = [values[i].is_shape for i in node.input if i not in drop]
    if any(is_shapes):
        if is_shapes[0] and len(node.output) == 1:
            values[node.output[0]].is_shape = True
            return [node.output[0]]
    else:
        for o in node.output:
            values[o].is_shape = False
        return list(node.output)
    return []


def first_used_last_used(
    proto: Union[onnx.FunctionProto, onnx.GraphProto, onnx.ModelProto],
    constant_as_initializer: bool = False,
) -> Dict[str, RuntimeValue]:
    """
    Builds first used, last used information for every result
    in the model.

    :param proto: model, graph or function
    :param constant_as_initializer: outputs of node Constant is tagged as INITIALIZER
    :return: dictionary of RuntimeValue
    """
    values = {}
    if isinstance(proto, onnx.ModelProto):
        initializer = proto.graph.initializer
        sparse_initializer = proto.graph.sparse_initializer
        _input = proto.graph.input
        output = proto.graph.output
        _node = proto.graph.node
        allow_unknown = False
    elif isinstance(proto, onnx.GraphProto):
        initializer = proto.initializer
        sparse_initializer = proto.sparse_initializer
        _input = proto.input
        output = proto.output
        _node = proto.node
        allow_unknown = True
    else:
        initializer = []
        sparse_initializer = []
        _input = proto.input
        output = proto.output
        _node = proto.node
        allow_unknown = False

    for init in initializer:
        values[init.name] = RuntimeValue(
            init.name, kind=RuntimeValueKind.INITIALIZER, created=-1
        )
    for init in sparse_initializer:
        values[init.name] = RuntimeValue(
            init.name, created=-1, kind=RuntimeValueKind.INITIALIZER
        )
    for inp in _input:
        n = inp if isinstance(inp, str) else inp.name
        values[n] = RuntimeValue(n, created=-1, kind=RuntimeValueKind.INPUT)
    drop = set()
    for it, node in enumerate(_node):
        for i in node.input:
            if i not in values:
                assert allow_unknown, f"Input {i!r} is unknown."
                # This input comes from a context and the model is a GraphProto
                drop.add(i)
                continue
            if values[i].first_used is None:
                values[i].first_used = it
            values[i].last_used = it
        for att in node.attribute:
            if att.type == onnx.AttributeProto.GRAPH:
                for n in get_hidden_inputs(att.g):
                    if values[n].first_used is None:
                        values[n].first_used = it
                    values[n].last_used = it
        is_constant = node.op_type == "Constant" and node.domain == ""
        for o in node.output:
            values[o] = RuntimeValue(
                o,
                created=it,
                kind=(
                    RuntimeValueKind.INITIALIZER
                    if is_constant and constant_as_initializer
                    else RuntimeValueKind.RESULT
                ),
            )
        set_is_shape(node, values, drop=drop)

    for out in output:
        n = out if isinstance(out, str) else out.name
        values[n].kind = RuntimeValueKind.OUTPUT
        values[n].last_used = len(_node)
    return values
