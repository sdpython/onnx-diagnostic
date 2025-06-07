import torch
from . import OpRunKernel, OpRunTensor


class OpRunBinary(OpRunKernel):
    "Binary Op"

    def run(self, x: OpRunTensor, y: OpRunTensor) -> OpRunTensor:
        if x.get_device() != y.get_device():
            if x.get_device() >= 0:
                y = y.to(x.device)
            else:
                x = x.to(y.device)
        return self._run(x, y)

    def _run(self, x: OpRunTensor, y: OpRunTensor) -> OpRunTensor:
        raise NotImplementedError(f"Operator {self.__class__.__name__!r} is not complete.")


class And_1(OpRunBinary):
    """And"""

    def _run(self, x: OpRunTensor, y: OpRunTensor) -> OpRunTensor:
        return OpRunTensor(x.tensor & y.tensor)


class Add_1(OpRunBinary):
    """Add"""

    def _run(self, x: OpRunTensor, y: OpRunTensor) -> OpRunTensor:
        return OpRunTensor(x.tensor + y.tensor)


class Div_1(OpRunBinary):
    """Div"""

    def _run(self, x: OpRunTensor, y: OpRunTensor) -> OpRunTensor:
        return OpRunTensor(x.tensor / y.tensor)


class Equal_1(OpRunBinary):
    """Equal"""

    def _run(self, x: OpRunTensor, y: OpRunTensor) -> OpRunTensor:
        return OpRunTensor(x.tensor == y.tensor)


class Greater_1(OpRunBinary):
    """Greater"""

    def _run(self, x: OpRunTensor, y: OpRunTensor) -> OpRunTensor:
        return OpRunTensor(x.tensor > y.tensor)


class GreaterOrEqual_1(OpRunBinary):
    """GreaterOrEqual"""

    def _run(self, x: OpRunTensor, y: OpRunTensor) -> OpRunTensor:
        return OpRunTensor(x.tensor >= y.tensor)


class Less_1(OpRunBinary):
    """Less"""

    def _run(self, x: OpRunTensor, y: OpRunTensor) -> OpRunTensor:
        return OpRunTensor(x.tensor < y.tensor)


class LessOrEqual_1(OpRunBinary):
    """LessOrEqual"""

    def _run(self, x: OpRunTensor, y: OpRunTensor) -> OpRunTensor:
        return OpRunTensor(x.tensor <= y.tensor)


class MatMul_1(OpRunBinary):
    """MatMul"""

    def _run(self, x: OpRunTensor, y: OpRunTensor) -> OpRunTensor:
        return OpRunTensor(x.tensor @ y.tensor)


class Mul_1(OpRunBinary):
    """Mul"""

    def _run(self, x: OpRunTensor, y: OpRunTensor) -> OpRunTensor:
        return OpRunTensor(x.tensor * y.tensor)


class Or_1(OpRunBinary):
    """Or"""

    def _run(self, x: OpRunTensor, y: OpRunTensor) -> OpRunTensor:
        return OpRunTensor(x.tensor | y.tensor)


class Pow_12(OpRunBinary):
    """Pow"""

    def _run(self, x: OpRunTensor, y: OpRunTensor) -> OpRunTensor:
        return OpRunTensor(torch.pow(x.tensor, y.tensor))


class Sub_1(OpRunBinary):
    """Sub"""

    def _run(self, x: OpRunTensor, y: OpRunTensor) -> OpRunTensor:
        return OpRunTensor(x.tensor - y.tensor)
