from . import OpRun


class Add_1(OpRun):
    """Add"""

    def run(self, x, y):
        return x + y


class Mul_1(OpRun):
    """Mul"""

    def run(self, x, y):
        return x * y


class Div_1(OpRun):
    """Div"""

    def run(self, x, y):
        return x / y


class Sub_1(OpRun):
    """Sub"""

    def run(self, x, y):
        return x - y
