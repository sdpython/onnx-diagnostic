from typing import Any, Dict, Optional
import onnx
from . import OpRun


class OpRunControlFlow(OpRun):
    """Common ancestor for control flows."""

    @classmethod
    def has_subgraphs(cls) -> bool:
        """Returns True if the kernel has subgraphs."""
        return True

    def __init__(
        self,
        node: onnx.NodeProto,
        version: Optional[int] = None,
        parent: Optional["onnx_diagnostic.reference.TorchOnnxEvaluator"] = None,  # noqa: F821
    ):
        super().__init__(node, version)
        assert (
            parent is not None
        ), f"parent must be specified for operator {self.__class__.__name__!r}"
        for att in node.attribute:
            if att.type == onnx.AttributeProto.GRAPH:
                rt = parent.__class__(
                    att.g,
                    providers=parent.providers,
                    opsets=parent.opsets,
                    local_functions=parent.functions,
                )
                setattr(self, att.name, rt)


class If_1(OpRunControlFlow):
    "If"

    def run(self, cond, context: Optional[Dict[str, Any]] = None):
        rt = self.then_branch if cond.tensor.item() else self.else_branch  # type: ignore[attr-defined]
        return rt.run_with_values(context=context)
