"""
.. _l-plot-failing-onnxruntime-evaluator:

Intermediate results with onnxruntime
=====================================

Example :ref:`l-plot-failing-reference-evaluator` demonstrated
how to run a python runtime on a model but it may very slow sometimes
and it could show some discrepancies if the only provider is not CPU.
Let's use :class:`OnnxruntimeEvaluator <onnx_diagnostic.reference.OnnxruntimeEvaluator>`.
It splits the model into node and runs them independently until it succeeds
or fails. This class converts every node into model based on the types
discovered during the execution. It relies on :class:`InferenceSessionForTorch
<onnx_diagnostic.ort_session.InferenceSessionForTorch>` or
:class:`InferenceSessionForNumpy
<onnx_diagnostic.ort_session.InferenceSessionForNumpy>`
for the execution. This example uses torch tensor and
bfloat16.

A failing model
+++++++++++++++

The issue here is a an operator ``Cast`` trying to convert a result
into a non-existing type.
"""

import onnx
import onnx.helper as oh
import torch
import onnxruntime
from onnx_diagnostic import doc
from onnx_diagnostic.ext_test_case import has_cuda
from onnx_diagnostic.helper import from_array_extended
from onnx_diagnostic.reference import OnnxruntimeEvaluator

TBFLOAT16 = onnx.TensorProto.BFLOAT16

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
            oh.make_tensor_value_info("X", TBFLOAT16, ["a", "b", "c"]),
            oh.make_tensor_value_info("Y", TBFLOAT16, ["a", "b", "c"]),
        ],
        [oh.make_tensor_value_info("Z", TBFLOAT16, ["a", "b", "c"])],
        [from_array_extended(torch.tensor([1], dtype=torch.bfloat16), name="one")],
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
# OnnxruntimeEvaluator
# ++++++++++++++++++++++++++
#
# This class extends :class:`onnx.reference.ReferenceEvaluator`
# with operators outside the standard but defined by :epkg:`onnxruntime`.
# `verbose=10` tells the class to print as much as possible,
# `verbose=0` prints nothing. Intermediate values for more or less verbosity.

ref = OnnxruntimeEvaluator(model, verbose=10)
feeds = dict(
    X=torch.rand((3, 4), dtype=torch.bfloat16), Y=torch.rand((3, 4), dtype=torch.bfloat16)
)
try:
    ref.run(None, feeds)
except Exception as e:
    print("ERROR", type(e), e)


# %%
# :epkg:`onnxruntime` may not support bfloat16 on CPU.
# See :epkg:`onnxruntime kernels`.

if has_cuda():
    ref = OnnxruntimeEvaluator(model, providers="cuda", verbose=10)
    feeds = dict(
        X=torch.rand((3, 4), dtype=torch.bfloat16), Y=torch.rand((3, 4), dtype=torch.bfloat16)
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

doc.plot_legend("onnxruntime\nrunning\nstep by step", "OnnxruntimeEvaluator", "lightgrey")
