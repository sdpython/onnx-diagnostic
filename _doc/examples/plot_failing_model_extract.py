"""
.. _l-plot-failing-model-extract:

Find where a model is failing by running submodels
==================================================

Let's assume :epkg:`onnxruntime` crashes without telling why or where.
The first thing is do is to locate where. For that, we extract every submodel
starting from the inputs and running the first *n* nodes of the model.
The model is likely to fail for some *n*. Then the failing is known.

This method only works if the model only contains operator coming
from the main domain *ai.onnx* otherwise shape inference stops
at the first non standard operator and the algorithm fails at
producing :class:`onnx.ModelProto` including the non standard operators.

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
from onnx_diagnostic.helpers.onnx_helper import from_array_extended
from onnx_diagnostic.ort_session import investigate_onnxruntime_issue

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
# Shape Inference
# +++++++++++++++
#
# Building submodels requires to known the output type.
# We run shape inference on the model.
shaped_model = onnx.shape_inference.infer_shapes(model)


# %%
# Looping over the nodes
# ++++++++++++++++++++++
#
#

failing = investigate_onnxruntime_issue(shaped_model, providers="cpu", verbose=1, quiet=True)

# %%
# Let's print the failing node.
print(failing)


# %%
# Detect an issue with shape Inference
# ++++++++++++++++++++++++++++++++++++
#
# We could have caught the error sooner by asking shape inference
# to raise an exception if one node could not be processed.
# It means either the node is a custom node
# and shape inference has no way to guess the output type and shape
# for this node or shape inference failed.

try:
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
except onnx.onnx_cpp2py_export.shape_inference.InferenceError as e:
    print(e)

# %%

doc.plot_legend("run onnx model\nuntil it fails", "onnxruntime.InferenceSession", "lightgrey")
