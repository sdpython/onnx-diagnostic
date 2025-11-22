"""
.. _l-plot-matmul-reverse-engineering:

=================
More about Linear
=================

"""

import cpuinfo
import pandas
import onnx
import onnx.helper as oh
from tqdm import tqdm
import torch
from onnx_diagnostic.ext_test_case import unit_test_going
from onnx_diagnostic.helpers import max_diff
from onnx_diagnostic.reference import OnnxruntimeEvaluator
from onnxruntime import __version__ as version_onnxruntime

print(f"onnxruntime version = {version_onnxruntime}")
print(f"cpu name = {cpuinfo.get_cpu_info()['brand_raw']}")
if torch.cuda.is_available():
    print(f"gpu name = {torch.cuda.get_device_name(0)}")
    print(f"cuda version = {torch.version.cuda}")

# %%
# The version is important. Numerical differences are observed
# with onnxruntime<=1.22. Let's see how to make them happen.


def make_model_gemm(itype: int) -> onnx.ModelProto:
    return oh.make_model(
        oh.make_graph(
            [oh.make_node("Gemm", ["A", "X", "B"], ["Y"])],
            "test",
            [
                oh.make_tensor_value_info("A", itype, ["a", "b"]),
                oh.make_tensor_value_info("X", itype, ["b", "c"]),
                oh.make_tensor_value_info("B", itype, ["c"]),
            ],
            [oh.make_tensor_value_info("Y", itype, ["a", "c"])],
        ),
        opset_imports=[oh.make_opsetid("", 22)],
        ir_version=10,
    )


def make_grid(N, bucket):
    a = torch.ones((N, N), dtype=torch.float32)
    n = N // bucket + (1 if N % bucket else 0)
    b = torch.ones((N,), dtype=torch.float32)
    mp = 8
    for i in range(n):
        for j in range(n):
            p = (i + j) % mp + 2
            val = float(2**p) * 0.1234
            a[
                i * bucket : min((i + 1) * bucket, N),
                (n - j - 2) * bucket : min((n - j - 1) * bucket, N),
            ] = val
        val = float(2 ** (i % mp)) + 0.1234
        b[i * bucket : min((i + 1) * bucket, N)] = val
    a -= a.mean()
    b -= b.mean()
    a /= a.std()
    b /= b.std()
    return a, -a, -b


print("N = 8, bucket = 2")
print(make_grid(8, 2)[0])

# %%
# We try different grid settings.

if torch.cuda.is_available():
    itype, dtype, device = onnx.TensorProto.FLOAT16, torch.float16, "cuda"
    data = []
    bar = tqdm(list(range(20, 1200, 100 if unit_test_going() else 1)))
    for i in bar:
        A, X, B = make_grid(1280, i)
        a = A.to(dtype).to(device)
        x = X.to(dtype).to(device)
        b = B.to(dtype).to(device)
        feeds = dict(A=a, X=x, B=b)
        model = make_model_gemm(itype)
        expected = torch.nn.functional.linear(a, x.T, b)
        sess = OnnxruntimeEvaluator(model, whole=True)
        results = sess.run(None, feeds)
        diff = max_diff(expected, results[0], hist=[0.1, 1.0])
        e32 = expected.to(torch.double)
        bar.set_description(f"err={diff['abs']:1.3f}")
        data.append(
            dict(
                M=A.shape[0],
                N=X.shape[1],
                K=A.shape[1],
                B=i,
                err=diff["abs"],
                nerr1=diff["rep"][">0.1"],
                mean=expected.to(torch.float32).mean().item(),
            )
        )

    df = pandas.DataFrame(data)
    print(df.tail())
    df[df["err"] > 0].to_excel("plot_matmul_reverse_engineering.cuda.xlsx")
    ax = df[["B", "err"]].set_index("B").plot(title="ERR / regularity size")
    ax.figure.savefig("plot_matmul_reverse_engineering.cuda.png")
