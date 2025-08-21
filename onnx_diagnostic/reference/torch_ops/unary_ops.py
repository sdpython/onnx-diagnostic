import torch
from . import OpRunKernel, OpRunTensor


class Abs_1(OpRunKernel):
    """Abs"""

    def run(self, x: OpRunTensor) -> OpRunTensor:
        return OpRunTensor(torch.abs(x.tensor))


class Cos_1(OpRunKernel):
    """Cos"""

    def run(self, x: OpRunTensor) -> OpRunTensor:
        return OpRunTensor(x.tensor.cos())


class Erf_9(OpRunKernel):
    """Erf"""

    def run(self, x: OpRunTensor) -> OpRunTensor:
        return OpRunTensor(x.tensor.erf())


class Exp_1(OpRunKernel):
    """Exp"""

    def run(self, x: OpRunTensor) -> OpRunTensor:
        return OpRunTensor(x.tensor.exp())


class Identity_1(OpRunKernel):
    "Identity"

    def run(self, x: OpRunTensor) -> OpRunTensor:
        return OpRunTensor(x.tensor)


class IsNaN_9(OpRunKernel):
    """IsNaN"""

    def run(self, x: OpRunTensor) -> OpRunTensor:
        return OpRunTensor(x.tensor.isnan())


class Log_1(OpRunKernel):
    """Log"""

    def run(self, x: OpRunTensor) -> OpRunTensor:
        return OpRunTensor(x.tensor.log())


class Neg_1(OpRunKernel):
    """Neg"""

    def run(self, x: OpRunTensor) -> OpRunTensor:
        return OpRunTensor(-x.tensor)


class Not_1(OpRunKernel):
    """Not"""

    def run(self, x: OpRunTensor) -> OpRunTensor:
        return OpRunTensor(~x.tensor)


class Reciprocal_1(OpRunKernel):
    """REciprocal"""

    def run(self, x: OpRunTensor) -> OpRunTensor:
        return OpRunTensor(1 / x.tensor)


class Sigmoid_6(OpRunKernel):
    """Sqrt"""

    def run(self, x: OpRunTensor) -> OpRunTensor:
        return OpRunTensor(torch.sigmoid(x.tensor))


class Sin_1(OpRunKernel):
    """Sin"""

    def run(self, x: OpRunTensor) -> OpRunTensor:
        return OpRunTensor(x.tensor.sin())


class Sqrt_1(OpRunKernel):
    """Sqrt"""

    def run(self, x: OpRunTensor) -> OpRunTensor:
        return OpRunTensor(x.tensor.sqrt())
