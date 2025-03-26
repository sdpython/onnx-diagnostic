"""
.. _l-plot-failing-reference-evaluator:

Intermediate results with (ONNX) ReferenceEvaluator
===================================================

Let's assume :epkg:`onnxruntime` crashes without telling why or where.
The first thing is do is to locate where. For that, we run a python runtime
which is going to run until it fails.

A failing model
+++++++++++++++

The issue here is a an operator ``Cast`` trying to convert a result
into a non-existing type.
"""

import numpy as np
import onnx
import onnx.helper as oh
import onnxruntime
from onnx_diagnostic import doc
from onnx_diagnostic.helpers import from_array_extended
from onnx_diagnostic.reference import ExtendedReferenceEvaluator

TFLOAT = onnx.TensorProto.FLOAT

model = oh.make_model(
    oh.make_graph(
        [
            oh.make_node("Mul", ["X", "Y"], ["xy"], name="n0"),
            oh.make_node("Sigmoid", ["xy"], ["sy"], name="n1"),
            oh.make_node("Add", ["sy", "one"], ["C"], name="n2"),
            oh.make_node("Cast", ["C"], ["X999"], to=999, name="failing"),
            oh.make_node("CastLike", ["X999", "Y"], ["Z"], name="n4"),
        ],
        "-nd-",
        [
            oh.make_tensor_value_info("X", TFLOAT, ["a", "b", "c"]),
            oh.make_tensor_value_info("Y", TFLOAT, ["a", "b", "c"]),
        ],
        [oh.make_tensor_value_info("Z", TFLOAT, ["a", "b", "c"])],
        [from_array_extended(np.array([1], dtype=np.float32), name="one")],
    ),
    opset_imports=[oh.make_opsetid("", 18)],
    ir_version=9,
)

# %%
# We check it is failing.

try:
    onnxruntime.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
except onnxruntime.capi.onnxruntime_pybind11_state.Fail as e:
    print(e)


# %%
# ExtendedReferenceEvaluator
# ++++++++++++++++++++++++++
#
# This class extends :class:`onnx.reference.ReferenceEvaluator`
# with operators outside the standard but defined by :epkg:`onnxruntime`.
# `verbose=10` tells the class to print as much as possible,
# `verbose=0` prints nothing. Intermediate values for more or less verbosity.

ref = ExtendedReferenceEvaluator(model, verbose=10)
feeds = dict(
    X=np.random.rand(3, 4).astype(np.float32), Y=np.random.rand(3, 4).astype(np.float32)
)
try:
    ref.run(None, feeds)
except Exception as e:
    print("ERROR", type(e), e)

# %%
# We can see it run until it reaches `Cast` and stops.
# The error message is not always obvious to interpret.
# It gets improved every time from time to time.
# This runtime is useful when it fails for a numerical reason.
# It is possible to insert prints in the python code to print
# more information or debug if needed.

doc.plot_legend("Python Runtime for ONNX", "ExtendedReferenceEvalutor", "lightgrey")
