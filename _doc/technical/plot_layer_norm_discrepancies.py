"""
LayerNormalization implementation cannot be exchanged
=====================================================

This example applies what was illustrated
:ref:`l-plot-parallelized-reduction`, reduction operations
are sensitive to parallelization.

We consider a small model including a layer normalization
followed by a matrix multiplication and we show that replacing
a kernel by another one may significantly impact the output.

The model
+++++++++
"""

import pandas
import onnx
import onnx.helper as oh
import onnxruntime
import torch
from onnx_array_api.plotting.graphviz_helper import plot_dot
from onnx_diagnostic.helpers import max_diff, string_diff, string_type
from onnx_diagnostic.reference import TorchOnnxEvaluator

TFLOAT16 = onnx.TensorProto.FLOAT16

model = oh.make_model(
    oh.make_graph(
        [
            oh.make_node("LayerNormalization", ["X", "scale", "bias"], ["norm"], axis=-1),
            oh.make_node("MatMul", ["norm", "weights"], ["mm"]),
            oh.make_node("Add", ["mm", "bias2"], ["Z"]),
        ],
        "layer_norm_matmul_add",
        [
            oh.make_tensor_value_info("X", TFLOAT16, ["a", "b", "c"]),
            oh.make_tensor_value_info("scale", TFLOAT16, ["c"]),
            oh.make_tensor_value_info("bias", TFLOAT16, ["c"]),
            oh.make_tensor_value_info("weights", TFLOAT16, ["c", "c"]),
            oh.make_tensor_value_info("bias2", TFLOAT16, ["c"]),
        ],
        [oh.make_tensor_value_info("Z", TFLOAT16, ["a", "b", "c"])],
    ),
    ir_version=9,
    opset_imports=[oh.make_opsetid("", 18)],
)

plot_dot(model)

# %%
# Let's compare two runtimes
# ++++++++++++++++++++++++++
#
# That will be :epkg:`onnxruntime` and
# :class:`onnx_diagnostic.reference.TorchOnnxEvaluator`.

feeds = {
    "X": (torch.rand((32, 1024, 1152), dtype=torch.float16) - 0.5) * 120,
    "scale": torch.rand((1152,), dtype=torch.float16),
    "bias": torch.rand((1152,), dtype=torch.float16),
    "weights": torch.rand((1152, 1152), dtype=torch.float16),
    "bias2": torch.rand((1152,), dtype=torch.float16),
}
np_feeds = {k: v.detach().numpy() for k, v in feeds.items()}
kws = dict(with_shape=True, with_min_max=True, with_device=True)
data = []

for provider in ["CPU", "CUDA"]:
    if provider == "CUDA":
        if not torch.cuda.is_available():
            continue
        tch_feeds = {k: v.to("cuda") for k, v in feeds.items()}
        ort_feeds = np_feeds
    else:
        tch_feeds = feeds.copy()
        tch_feeds["X"] = tch_feeds["X"][:2]  # too long otherwise
        ort_feeds = np_feeds.copy()
        ort_feeds["X"] = ort_feeds["X"][:2]
    print()
    print(f"-- running on {provider}")
    print("-- running with torch")
    torch_sess = TorchOnnxEvaluator(model, providers=[f"{provider}ExecutionProvider"])
    expected = torch_sess.run(None, tch_feeds)
    print(f"-- torch: {string_type(expected, **kws)}")

    print("-- running with ort")
    ort_sess = onnxruntime.InferenceSession(
        model.SerializeToString(), providers=[f"{provider}ExecutionProvider"]
    )
    got = ort_sess.run(None, ort_feeds)
    print(f"-- ort: {string_type(got, **kws)}")
    diff = max_diff(expected, got, hist=True)
    print(f"-- diff {string_diff(diff)}")

    # memorize the data
    diff["provider"] = provider
    diff.update(diff["rep"])
    del diff["rep"]
    data.append(diff)

# %%
df = pandas.DataFrame(data).set_index("provider")
print(df)
