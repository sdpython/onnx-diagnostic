"""
Compares two ONNX models.
"""

print("-- import onnx")
import onnx

print("-- import onnx.helper")
from onnx.helper import tensor_dtype_to_np_dtype

print("-- import onnxruntime")
import onnxruntime

print("-- import torch")
import torch

print("-- import transformers")
import transformers

print("-- import huggingface_hub")
import huggingface_hub

print("-- import onnx-diagnostic.helper")
from onnx_diagnostic.helpers.helper import flatten_object, string_type, max_diff, string_diff

print("-- import onnx-diagnostic.torch_models")
from onnx_diagnostic.torch_models.hghub import get_untrained_model_with_inputs

print("-- done")

model_id = "arnir0/Tiny-LLM"
onnx1 = (
    "dump_test/arnir0_Tiny-LLM-custom-default-f16-cuda-op20/"
    "arnir0_Tiny-LLM-custom-default-f16-cuda-op20.onnx"
)
onnx2 = (
    "dump_test/arnir0_Tiny-LLM-custom-default-f16-cuda-op21/"
    "arnir0_Tiny-LLM-custom-default-f16-cuda-op21.onnx"
)
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

print(f"-- load {onnx1!r}")
onx1 = onnx.load(onnx1)
print(f"-- load {onnx2!r}")
onx2 = onnx.load(onnx2)

print(f"-- getting inputs for model_id {model_id!r}")
data = get_untrained_model_with_inputs(model_id)
inputs = data["inputs"]
print(f"-- inputs: {string_type(inputs, with_shape=True)}")
flatten_inputs = flatten_object(inputs, drop_keys=True)
print(f"-- flat inputs: {string_type(flatten_inputs, with_shape=True)}")

names = [i.name for i in onx1.graph.input]
itypes = [i.type.tensor_type.elem_type for i in onx1.graph.input]
assert names == [
    i.name for i in onx2.graph.input
], f"Not the same names for both models {names} != {[i.name for i in onx2.graph.input]}"
feeds = {
    n: t.numpy().astype(tensor_dtype_to_np_dtype(itype))
    for n, itype, t in zip(names, itypes, flatten_inputs)
}
print(f"-- feeds: {string_type(feeds, with_shape=True)}")

print(f"-- creating session 1 from {onnx1!r}")
opts = onnxruntime.SessionOptions()
opts.optimized_model_filepath = "debug1_full.onnx"
opts.log_severity_level = 0
opts.log_verbosity_level = 0
sess1 = onnxruntime.InferenceSession(onnx1, opts, providers=providers)
print(f"-- creating session 2 from {onnx2!r}")
opts.optimized_model_filepath = "debug2_full.onnx"
opts.log_severity_level = 0
opts.log_verbosity_level = 0
sess2 = onnxruntime.InferenceSession(onnx2, opts, providers=providers)

print("-- run session1")
expected1 = sess1.run(None, feeds)
print(f"-- got {string_type(expected1, with_shape=True)}")
print("-- run session2")
expected2 = sess2.run(None, feeds)
print(f"-- got {string_type(expected2, with_shape=True)}")

print("-- compute differences")
diff = max_diff(expected1, expected2)
print(f"-- diff={string_diff(diff)}")


def get_names(onx: onnx.ModelProto) -> list[str]:
    names = []
    for node in onx.graph.node:
        for o in node.output:
            names.append((o, node.op_type, node.name))
    return names


if diff["abs"] > 0.1:
    print("--")
    print("-- import select_model_inputs_outputs")
    from onnx_extended.tools.onnx_nodes import select_model_inputs_outputs

    print("-- looking into intermediate results")
    names1 = get_names(onx1)
    names2 = get_names(onx1)
    common = [n for n in names1 if n in (set(names1) & set(names2))]
    print(f"-- {len(common)} names / {len(names1)}-{len(names2)}")
    print(f"-- first names {common[:5]}")
    for name, op_type, op_name in common:
        x1 = select_model_inputs_outputs(onx1, [name])
        x2 = select_model_inputs_outputs(onx2, [name])
        s1 = onnxruntime.InferenceSession(x1.SerializeToString(), providers=providers)
        s2 = onnxruntime.InferenceSession(x2.SerializeToString(), providers=providers)
        e1 = s1.run(None, feeds)
        e2 = s2.run(None, feeds)
        diff = max_diff(e1, e2)
        print(
            f"-- name={name!r}: diff={string_diff(diff)} "
            f"- op_type={op_type!r}, op_name={op_name!r}"
        )
        if diff["abs"] > 0.1:
            opts = onnxruntime.SessionOptions()
            opts.optimized_model_filepath = "debug1.onnx"
            onnxruntime.InferenceSession(x1.SerializeToString(), opts, providers=providers)
            opts.optimized_model_filepath = "debug2.onnx"
            onnxruntime.InferenceSession(x2.SerializeToString(), opts, providers=providers)
            print("--")
            print("-- break here")
            print(f"-- feeds {string_type(feeds, with_shape=True)}")
            print(f"-- e1={string_type(e1, with_shape=True, with_min_max=True)}")
            print(f"-- e2={string_type(e2, with_shape=True, with_min_max=True)}")
            break
