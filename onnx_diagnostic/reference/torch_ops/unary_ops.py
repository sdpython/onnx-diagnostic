from . import OpRun, OpRunValue


class Neg_1(OpRun):
    """Neg"""

    def run(self, x: OpRunValue) -> OpRunValue:
        return OpRunValue(-x.tensor)


class Not_1(OpRun):
    """Not"""

    def run(self, x: OpRunValue) -> OpRunValue:
        return OpRunValue(~x.tensor)


class Reciprocal_1(OpRun):
    """REciprocal"""

    def run(self, x: OpRunValue) -> OpRunValue:
        return OpRunValue(1 / x.tensor)
