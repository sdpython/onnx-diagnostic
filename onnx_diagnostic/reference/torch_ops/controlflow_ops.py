from typing import Any, Dict, Optional
import onnx
import torch
from . import OpRunKernel, OpRunTensor


class OpRunControlFlow(OpRunKernel):
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
                    verbose=parent.verbose,
                )
                setattr(self, att.name, rt)


class If_1(OpRunControlFlow):
    "If"

    def run(self, cond, context: Optional[Dict[str, Any]] = None):
        rt = self.then_branch if cond.tensor.item() else self.else_branch  # type: ignore[attr-defined]
        return rt.run_with_values(context=context)


class Loop_16(OpRunControlFlow):
    "Loop"

    def __init__(
        self,
        node: onnx.NodeProto,
        version: Optional[int] = None,
        parent: Optional["onnx_diagnostic.reference.TorchOnnxEvaluator"] = None,  # noqa: F821
    ):
        super().__init__(node, version, parent)
        self.output_index = {n: i for i, n in enumerate(self.body.output_names)}
        self.N = len(self.body.input_names) - 2
        self.K = len(self.body.output_names) - self.N - 1

    def run(self, M, cond, *args, context: Optional[Dict[str, Any]] = None):
        if args:
            v_initial = args[0]
            args = args[1:]
        else:
            v_initial = None
        assert M is None or hasattr(
            M, "dtype"
        ), f"M must be empty or an array but its type is {type(M)}."
        body = self.body
        loop_inputs = body.input_names
        inputs = dict.fromkeys(loop_inputs)
        if v_initial is not None:
            inputs[loop_inputs[2]] = v_initial
        cond_name = body.output_names[0]
        if args:
            begin = len(loop_inputs) - len(args)
            all_inputs = loop_inputs[begin:]
            for name, val in zip(all_inputs, args):
                inputs[name] = val
        if context is not None:
            for a in context:
                inputs[a] = context[a]

        k_carried_away = [[] for i in range(self.K)]  # type: ignore
        it = 0
        while (cond is None or cond.tensor is None or cond.tensor.item()) and (
            M is None or M.tensor is None or it < M.tensor.item()
        ):
            if len(body.input_names) > 0 and body.input_names[0] is not None:
                inputs[body.input_names[0]] = OpRunTensor(
                    torch.tensor(it, dtype=None if M is None else M.dtype)
                )
            if len(body.input_names) > 1 and body.input_names[1] is not None:
                inputs[body.input_names[1]] = cond
            outputs = list(
                self.body.run_with_values(
                    *[inputs[k] for k in self.body.input_names], context=context
                )
            )
            if self.K > 0:
                for k in range(self.K):
                    k_carried_away[k].append(outputs[-self.K + k])
            index_cond = self.output_index[cond_name]
            cond = outputs[index_cond]
            assert (
                cond is not None
            ), f"Condition {cond_name!r} returned by the subgraph cannot be None."
            for i, o in zip(body.input_names[2:], body.output_names[1:]):
                inputs[i] = outputs[self.output_index[o]]
            it += 1

        if it == 0:
            outputs = [inputs[i] for i in body.input_names[2:]]
        else:
            outputs = outputs[1 : 1 + self.N]
        outputs.extend([OpRunTensor(torch.cat(x, axis=0)) for x in k_carried_away])
        while len(outputs) < len(self.body.output_names):
            outputs.append(OpRunTensor(torch.empty(())))
        return tuple(outputs)
