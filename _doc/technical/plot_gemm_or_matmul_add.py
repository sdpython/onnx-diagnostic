"""
.. _l-plot-gemm-or-matmul-add:

====================
Gemm or Matmul + Add
====================

Order of computation matters. ``1 + 1e-20 - 1 != 1 - 1 + 1e-20`` if the
precision of the computation is taken into account.
What an operator Gemm in :epkg:`onnxruntime`, the most simple
way to represent a linear neural layer.

A model with three choices
==========================
"""

import numpy as np
import pandas
import matplotlib.pyplot as plt
import onnx
import onnx.helper as oh
import torch
from onnx_diagnostic.helpers import max_diff
from onnx_diagnostic.helpers.onnx_helper import pretty_onnx
from onnx_diagnostic.reference import OnnxruntimeEvaluator
from onnxruntime import (
    InferenceSession,
    SessionOptions,
    __version__ as version_onnxruntime,
    GraphOptimizationLevel,
)

print(f"onnxruntime version = {version_onnxruntime}")

# %%
# The version is important. Numerical differences are observed
# with onnxruntime<=1.22. Let's see how to make them happen.


def make_model_gemm(itype: int) -> onnx.ModelProto:
    return oh.make_model(
        oh.make_graph(
            [
                oh.make_node("Gemm", ["A", "X", "B"], ["Ygemmfused"]),
                oh.make_node("Gemm", ["A", "X"], ["gmm"]),
                oh.make_node("Add", ["gmm", "B"], ["Ygemm"]),
                oh.make_node("MatMul", ["A", "X"], ["mm"]),
                oh.make_node("Add", ["mm", "B"], ["Ymm"]),
                oh.make_node("FusedMatMul", ["A", "X"], ["fmm"], domain="com.microsoft"),
                oh.make_node("Add", ["fmm", "B"], ["Yfused"]),
            ],
            "test",
            [
                oh.make_tensor_value_info("A", itype, ["a", "b"]),
                oh.make_tensor_value_info("X", itype, ["b", "c"]),
                oh.make_tensor_value_info("B", itype, ["c"]),
            ],
            [
                oh.make_tensor_value_info("Ygemmfused", itype, ["a", "c"]),
                oh.make_tensor_value_info("Yfused", itype, ["a", "c"]),
                oh.make_tensor_value_info("Ygemm", itype, ["a", "c"]),
                oh.make_tensor_value_info("Ymm", itype, ["a", "c"]),
            ],
        ),
        opset_imports=[oh.make_opsetid("", 22)],
        ir_version=10,
    )


def matrix_diff(tensors):
    mat = np.zeros((len(tensors), len(tensors)), dtype=np.float32)
    for i, t in enumerate(tensors):
        for j in range(i + 1, len(tensors)):
            mat[i, j] = max_diff(t, tensors[j])["abs"]
            mat[j, i] = mat[i, j]
    return mat


itype = onnx.TensorProto.FLOAT16
dtype = np.float16
model = make_model_gemm(itype)

A = np.random.randn(512, 256).astype(dtype)
X = np.random.randn(256, 256).astype(dtype)
B = np.random.randn(256).astype(dtype)
feeds = dict(A=A, X=X, B=B)

# %%
# We disable all the optimization made by onnxruntime to make
# the computation follows what we want to verify.
opts = SessionOptions()
opts.graph_optimization_level = GraphOptimizationLevel.ORT_DISABLE_ALL
opts.optimized_model_filepath = "plot_gemm_or_matmul.optimized.onnx"
sess = InferenceSession(model.SerializeToString(), opts, providers=["CPUExecutionProvider"])
results = [A @ X + B, *sess.run(None, feeds)]
diffs = matrix_diff(results)

print(diffs)

# %%
onx = onnx.load(opts.optimized_model_filepath)
print(pretty_onnx(onx))

# %%
# It seems some cast were still inserted.

# %%
# Let's try with CUDA and float32 if it is available.

A = torch.randn((512, 512), dtype=torch.float32)
X = torch.randn((512, 512), dtype=torch.float32)
B = torch.randn((512), dtype=torch.float32)

for itype, dtype, device in [
    (onnx.TensorProto.FLOAT16, torch.float16, "cpu"),
    (onnx.TensorProto.FLOAT, torch.float32, "cpu"),
    (onnx.TensorProto.FLOAT16, torch.float16, "cuda"),
    (onnx.TensorProto.FLOAT, torch.float32, "cuda"),
]:
    if device == "cuda" and not torch.cuda.is_available():
        continue
    a = A.to(dtype).to(device)
    x = X.to(dtype).to(device)
    b = B.to(dtype).to(device)
    feeds = dict(A=a, X=x, B=b)
    model = make_model_gemm(itype)

    sess = OnnxruntimeEvaluator(model, whole=True)
    results = sess.run(None, feeds)
    diffs = matrix_diff(results)
    print(f"------ dtype={dtype}, device={device!r}")
    print(diffs)

# %%
# A weird bias
# ============
#
# In the previous example, the coefficients of the bias
# are simular to the others coefficients. What if we make them
# a lot higher.

B = (torch.arange(512, dtype=torch.float32) + 1) / 512 * 16384
labels = ["torch", *[o.name for o in model.graph.output]]

for itype, dtype, device in [
    (onnx.TensorProto.FLOAT, torch.float32, "cpu"),
    (onnx.TensorProto.FLOAT16, torch.float16, "cpu"),
    (onnx.TensorProto.FLOAT, torch.float32, "cuda"),
    (onnx.TensorProto.FLOAT16, torch.float16, "cuda"),
]:
    if device == "cuda" and not torch.cuda.is_available():
        continue
    a = A.to(dtype).to(device)
    x = X.to(dtype).to(device)
    b = B.to(dtype).to(device)
    feeds = dict(A=a, X=x, B=b)
    model = make_model_gemm(itype)

    filename = f"plot_gemm_or_matmul.{itype}.{device}.onnx"
    sess = OnnxruntimeEvaluator(
        model,
        whole=True,
        graph_optimization_level=GraphOptimizationLevel.ORT_DISABLE_ALL,
        optimized_model_filepath=filename,
    )
    has_cast = "Cast" in [n.op_type for n in onnx.load(filename).graph.node]
    results = [a @ x + b, *sess.run(None, feeds)]
    diffs = matrix_diff(results)
    df = pandas.DataFrame(diffs, columns=labels, index=labels)
    print(f"------ has_cast={has_cast}, dtype={dtype}, device={device!r}, max(b)={b.max()}")
    print(df)

# %%
# Cast is inserted on CPU because some kernel are not available for
# float16. Even though, we can see huge discrepancies happening.
#
# bias value vs discrepancies
# ===========================


m1, m2 = results[0:2]
diff = torch.abs(m1.to(torch.float32) - m2.to(torch.float32)).max(dim=0)[0]
print(f"max(diff)={diff.max()}")

fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(B.tolist(), (diff.detach().cpu() + torch.rand(512) * 0.5).tolist(), ".")
ax.set_title("Discrepancies (y) VS Bias (x)")
fig.savefig("plot_gemm_or_matmul_add.png")

# %%
# Discrepancies do not happen all the time but it is very likely to happen.
# Fused Gemm should be avoided when the bias is very different from the multiplied
# matrix and avoided in the generic case.
