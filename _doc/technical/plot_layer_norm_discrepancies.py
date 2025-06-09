"""
LayerNormalization implementation cannot be exchanged
=====================================================

This example applies what was illustrated
:ref:`l-plot-parallelized-reduction`, reduction operations
are sensitive to parallelization.

Methodology
+++++++++++

We consider a simple model with a LayerNormalization followed by a MatMul.
Each operator can be run with :epkg:`onnxruntime` or :epkg:`pytorch`.
We compare the four combinations.

The model
+++++++++
"""

import itertools
import numpy as np
import pandas
import onnx
import onnx.helper as oh
import onnxruntime
import torch
from onnx_array_api.plotting.graphviz_helper import plot_dot
from onnx_diagnostic.doc import rotate_align, save_fig, plot_histogram, title
from onnx_diagnostic.ext_test_case import unit_test_going
from onnx_diagnostic.helpers import max_diff, string_diff, string_type
from onnx_diagnostic.helpers.onnx_helper import onnx_dtype_name, onnx_dtype_to_np_dtype
from onnx_diagnostic.helpers.torch_helper import onnx_dtype_to_torch_dtype
from onnx_diagnostic.helpers.doc_helper import LayerNormalizationOrt, MatMulOrt
from onnx_diagnostic.reference import TorchOnnxEvaluator

TFLOAT = onnx.TensorProto.FLOAT
TFLOAT16 = onnx.TensorProto.FLOAT16


def get_model(itype: int = TFLOAT16):
    return oh.make_model(
        oh.make_graph(
            [
                oh.make_node("LayerNormalization", ["X", "scale", "bias"], ["norm"], axis=-1),
                oh.make_node("MatMul", ["norm", "weights"], ["mm"]),
                oh.make_node("Add", ["mm", "bias2"], ["Z"]),
            ],
            "layer_norm_matmul_add",
            [
                oh.make_tensor_value_info("X", itype, ["a", "b", "c"]),
                oh.make_tensor_value_info("scale", itype, ["c"]),
                oh.make_tensor_value_info("bias", itype, ["c"]),
                oh.make_tensor_value_info("weights", itype, ["c", "c"]),
                oh.make_tensor_value_info("bias2", itype, ["c"]),
            ],
            [oh.make_tensor_value_info("Z", itype, ["a", "b", "c"])],
        ),
        ir_version=9,
        opset_imports=[oh.make_opsetid("", 18)],
    )


model = get_model()
plot_dot(model)

# %%
# Let's compare two runtimes
# ++++++++++++++++++++++++++
#
# That will be :epkg:`onnxruntime` and
# :class:`onnx_diagnostic.reference.TorchOnnxEvaluator`.

last_dim = 64 if unit_test_going() else 1152


def make_feeds(last_dim: int):
    return {
        "X": (torch.rand((32, 1024, last_dim), dtype=torch.float16) - 0.5) * 120,
        "scale": torch.rand((last_dim,), dtype=torch.float16),
        "bias": torch.rand((last_dim,), dtype=torch.float16),
        "weights": torch.rand((last_dim, last_dim), dtype=torch.float16),
        "bias2": torch.rand((last_dim,), dtype=torch.float16),
    }


def cast_feeds(itype, provider, feeds):
    ttype = onnx_dtype_to_torch_dtype(itype)
    np_dtype = onnx_dtype_to_np_dtype(itype)
    np_feeds = {k: v.detach().numpy() for k, v in feeds.items()}
    if provider == "CUDA":
        if not torch.cuda.is_available():
            return None, None
        tch_feeds = {k: v.to("cuda") for k, v in feeds.items()}
        ort_feeds = np_feeds
    else:
        tch_feeds = feeds.copy()
        tch_feeds["X"] = tch_feeds["X"][:2]  # too long otherwise
        ort_feeds = np_feeds.copy()
        ort_feeds["X"] = ort_feeds["X"][:2]
    tch_feeds = {k: v.to(ttype) for k, v in tch_feeds.items()}
    ort_feeds = {k: v.astype(np_dtype) for k, v in ort_feeds.items()}
    return tch_feeds, ort_feeds


feeds = make_feeds(last_dim)
kws = dict(with_shape=True, with_min_max=True, with_device=True)
data = []
baseline = {}

for provider, itype in itertools.product(["CPU", "CUDA"], [TFLOAT, TFLOAT16]):
    tch_feeds, ort_feeds = cast_feeds(itype, provider, feeds)
    if tch_feeds is None:
        continue

    model = get_model(itype)
    print()
    print(f"-- running on {provider} with {onnx_dtype_name(itype)}")
    print("-- running with torch")
    torch_sess = TorchOnnxEvaluator(model, providers=[f"{provider}ExecutionProvider"])
    expected = torch_sess.run(None, tch_feeds)
    baseline[itype, provider, "torch"] = expected
    print(f"-- torch: {string_type(expected, **kws)}")

    print("-- running with ort")
    ort_sess = onnxruntime.InferenceSession(
        model.SerializeToString(), providers=[f"{provider}ExecutionProvider"]
    )
    got = ort_sess.run(None, ort_feeds)
    baseline[itype, provider, "ort"] = got
    print(f"-- ort: {string_type(got, **kws)}")
    diff = max_diff(expected, got, hist=True)
    print(f"-- diff {string_diff(diff)}")

    # memorize the data
    diff["dtype"] = onnx_dtype_name(itype)
    diff["provider"] = provider
    diff.update(diff["rep"])
    del diff["rep"]
    del diff["dnan"]
    del diff[">100.0"]
    del diff[">10.0"]
    data.append(diff)

# %%
df = pandas.DataFrame(data).set_index(["provider", "dtype"])
print(df)

# %%
# Visually.

save_fig(
    rotate_align(
        df[["abs"]].plot.bar(title="Discrepancies ORT / torch for LayerNorm(X) @ W + B")
    ),
    "plot_layer_norm_discrepancies_1.png",
)

# %%
# The discrepancies are significant on CUDA, higher for float16.
# Let's see which operator is responsible for them,
# *LayerNormalization* or *MatMul*.

# %%
# Distribution of the results
# +++++++++++++++++++++++++++

tensor = baseline[TFLOAT16, "CPU", "ort"][0].ravel().astype(np.float32)
print(pandas.DataFrame({"expected": tensor}).describe())

# %%
# Histogram.

save_fig(
    title(plot_histogram(tensor), "Distribution of the computed results"),
    "plot_layer_norm_discrepancies_hist.png",
)


# %%
# The discrepancies come from?
# ++++++++++++++++++++++++++++
#
# We mix torch and onnxruntime to execute the kernels.

data = []

for mod, provider, itype in itertools.product(
    ["ORT-ORT", "ORT-TORCH", "TORCH-ORT", "TORCH-TORCH"], ["CPU", "CUDA"], [TFLOAT, TFLOAT16]
):
    ttype = onnx_dtype_to_torch_dtype(itype)
    np_dtype = onnx_dtype_to_np_dtype(itype)
    tch_feeds, _ = cast_feeds(itype, provider, feeds)
    if tch_feeds is None:
        continue

    ker1, ker2 = mod.split("-")
    custom_kernels = (
        {("", "LayerNormalization"): LayerNormalizationOrt} if ker1 == "ORT" else {}
    ) | ({("", "MatMul"): MatMulOrt} if ker2 == "ORT" else {})

    model = get_model(itype)
    print()
    print(f"-- {mod} running on {provider} with {onnx_dtype_name(itype)}")
    sess = TorchOnnxEvaluator(
        model,
        custom_kernels=custom_kernels,
        providers=[f"{provider}ExecutionProvider"],
    )
    got = sess.run(None, tch_feeds)
    print(f"-- {mod}: {string_type(got, **kws)}")

    difft = max_diff(baseline[itype, provider, "torch"], got)
    print(f"-- diff with torch {string_diff(difft)}")
    diffo = max_diff(baseline[itype, provider, "ort"], got)
    print(f"-- diff with ort {string_diff(diffo)}")

    data.append(
        dict(
            model=mod,
            dtype=onnx_dtype_name(itype),
            provider=provider,
            diff_ort=diffo["abs"],
            diff_torch=difft["abs"],
        )
    )

# %%
df = pandas.DataFrame(data).set_index(["dtype", "provider", "model"])
df = df.sort_index()
print(df)

# %%
# Visually.

save_fig(
    rotate_align(
        df[["diff_ort", "diff_torch"]].plot.bar(
            title="ORT/Torch or Torch/ORT for LayerNorm(X) @ W + B",
            figsize=(10, 4),
        )
    ),
    "plot_layer_norm_discrepancies_2.png",
)

# %%
# Conclusion
# ++++++++++
#
# :epkg:`torch` seems able to replicate the same results if the same computation
# is run multiple times. :epkg:`onnxruntime` is only able to do that on CUDA.
# With float16 and CUDA, LayerNormalization seems to introduce some discrepancies.
