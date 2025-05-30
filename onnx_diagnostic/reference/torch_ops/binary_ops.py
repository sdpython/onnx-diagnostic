from . import OpRun, OpRunValue


class Add_1(OpRun):
    """Add"""

    def run(self, x: OpRunValue, y: OpRunValue) -> OpRunValue:
        return OpRunValue(x.tensor + y.tensor)


class Mul_1(OpRun):
    """Mul"""

    def run(self, x: OpRunValue, y: OpRunValue) -> OpRunValue:
        return OpRunValue(x.tensor * y.tensor)


class Div_1(OpRun):
    """Div"""

    def run(self, x: OpRunValue, y: OpRunValue) -> OpRunValue:
        return OpRunValue(x.tensor / y.tensor)


class Sub_1(OpRun):
    """Sub"""

    def run(self, x: OpRunValue, y: OpRunValue) -> OpRunValue:
        return OpRunValue(x.tensor - y.tensor)
