import torch
from . import OpRun, OpRunValue


class Abs_1(OpRun):
    """Abs"""

    def run(self, x: OpRunValue) -> OpRunValue:
        return OpRunValue(torch.abs(x.tensor))


class Cos_1(OpRun):
    """Cos"""

    def run(self, x: OpRunValue) -> OpRunValue:
        return OpRunValue(x.tensor.cos())


class Exp_1(OpRun):
    """Exp"""

    def run(self, x: OpRunValue) -> OpRunValue:
        return OpRunValue(x.tensor.exp())


class Log_1(OpRun):
    """Log"""

    def run(self, x: OpRunValue) -> OpRunValue:
        return OpRunValue(x.tensor.log())


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


class Sigmoid_6(OpRun):
    """Sqrt"""

    def run(self, x: OpRunValue) -> OpRunValue:
        return OpRunValue(torch.sigmoid(x.tensor))


class Sin_1(OpRun):
    """Sin"""

    def run(self, x: OpRunValue) -> OpRunValue:
        return OpRunValue(x.tensor.sin())


class Sqrt_1(OpRun):
    """Sqrt"""

    def run(self, x: OpRunValue) -> OpRunValue:
        return OpRunValue(x.tensor.sqrt())
