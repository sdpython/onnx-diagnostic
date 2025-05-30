from . import OpRun, OpRunValue


class Squeeze_13(OpRun):
    "Squeeze"

    def run(self, data: OpRunValue, axes: OpRunValue) -> OpRunValue:
        return OpRunValue(data.tensor.squeeze(axes.as_tuple_int))
