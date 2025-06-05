import torch
from . import OpRun, OpRunTensor


class Abs_1(OpRun):
    """Abs"""

    def run(self, x: OpRunTensor) -> OpRunTensor:
        return OpRunTensor(torch.abs(x.tensor))


class Cos_1(OpRun):
    """Cos"""

    def run(self, x: OpRunTensor) -> OpRunTensor:
        return OpRunTensor(x.tensor.cos())


class Erf_9(OpRun):
    """Erf"""

    def run(self, x: OpRunTensor) -> OpRunTensor:
        return OpRunTensor(x.tensor.erf())


class Exp_1(OpRun):
    """Exp"""

    def run(self, x: OpRunTensor) -> OpRunTensor:
        return OpRunTensor(x.tensor.exp())


class Identity_1(OpRun):
    "Identity"

    def run(self, x: OpRunTensor) -> OpRunTensor:
        return OpRunTensor(x.tensor)


class Log_1(OpRun):
    """Log"""

    def run(self, x: OpRunTensor) -> OpRunTensor:
        return OpRunTensor(x.tensor.log())


class Neg_1(OpRun):
    """Neg"""

    def run(self, x: OpRunTensor) -> OpRunTensor:
        return OpRunTensor(-x.tensor)


class Not_1(OpRun):
    """Not"""

    def run(self, x: OpRunTensor) -> OpRunTensor:
        return OpRunTensor(~x.tensor)


class Reciprocal_1(OpRun):
    """REciprocal"""

    def run(self, x: OpRunTensor) -> OpRunTensor:
        return OpRunTensor(1 / x.tensor)


class Sigmoid_6(OpRun):
    """Sqrt"""

    def run(self, x: OpRunTensor) -> OpRunTensor:
        return OpRunTensor(torch.sigmoid(x.tensor))


class Sin_1(OpRun):
    """Sin"""

    def run(self, x: OpRunTensor) -> OpRunTensor:
        return OpRunTensor(x.tensor.sin())


class Sqrt_1(OpRun):
    """Sqrt"""

    def run(self, x: OpRunTensor) -> OpRunTensor:
        return OpRunTensor(x.tensor.sqrt())
