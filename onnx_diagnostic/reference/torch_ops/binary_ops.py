import torch
from . import OpRun, OpRunValue


class And_1(OpRun):
    """And"""

    def run(self, x: OpRunValue, y: OpRunValue) -> OpRunValue:
        return OpRunValue(x.tensor & y.tensor)


class Add_1(OpRun):
    """Add"""

    def run(self, x: OpRunValue, y: OpRunValue) -> OpRunValue:
        return OpRunValue(x.tensor + y.tensor)


class Div_1(OpRun):
    """Div"""

    def run(self, x: OpRunValue, y: OpRunValue) -> OpRunValue:
        return OpRunValue(x.tensor / y.tensor)


class Equal_1(OpRun):
    """Equal"""

    def run(self, x: OpRunValue, y: OpRunValue) -> OpRunValue:
        return OpRunValue(x.tensor == y.tensor)


class Greater_1(OpRun):
    """Greater"""

    def run(self, x: OpRunValue, y: OpRunValue) -> OpRunValue:
        return OpRunValue(x.tensor > y.tensor)


class GreaterOrEqual_1(OpRun):
    """GreaterOrEqual"""

    def run(self, x: OpRunValue, y: OpRunValue) -> OpRunValue:
        return OpRunValue(x.tensor >= y.tensor)


class Less_1(OpRun):
    """Less"""

    def run(self, x: OpRunValue, y: OpRunValue) -> OpRunValue:
        return OpRunValue(x.tensor < y.tensor)


class LessOrEqual_1(OpRun):
    """LessOrEqual"""

    def run(self, x: OpRunValue, y: OpRunValue) -> OpRunValue:
        return OpRunValue(x.tensor <= y.tensor)


class MatMul_1(OpRun):
    """MatMul"""

    def run(self, x: OpRunValue, y: OpRunValue) -> OpRunValue:
        return OpRunValue(x.tensor @ y.tensor)


class Mul_1(OpRun):
    """Mul"""

    def run(self, x: OpRunValue, y: OpRunValue) -> OpRunValue:
        return OpRunValue(x.tensor * y.tensor)


class Or_1(OpRun):
    """Or"""

    def run(self, x: OpRunValue, y: OpRunValue) -> OpRunValue:
        return OpRunValue(x.tensor | y.tensor)


class Pow_12(OpRun):
    """Pow"""

    def run(self, x: OpRunValue, y: OpRunValue) -> OpRunValue:
        return OpRunValue(torch.pow(x.tensor, y.tensor))


class Sub_1(OpRun):
    """Sub"""

    def run(self, x: OpRunValue, y: OpRunValue) -> OpRunValue:
        return OpRunValue(x.tensor - y.tensor)
