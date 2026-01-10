"""
.. _l-plot-intermediate-results:

Dumps intermediate results of a torch model
===========================================

Looking for discrepancies is quickly annoying. Discrepancies
come from two results obtained with the same models
implemented in two different ways, :epkg:`pytorch` and :epkg:`onnx`.
Models are big so where do they come from? That's the
unavoidable question. Unless there is an obvious reason,
the only way is to compare intermediate outputs alon the computation.
The first step into that direction is to dump the intermediate results
coming from :epkg:`pytorch`.
We use :func:`onnx_diagnostic.helpers.torch_helper.steal_forward` for that.

A simple LLM Model
++++++++++++++++++

See :func:`onnx_diagnostic.helpers.torch_helper.dummy_llm`
for its definition. It is mostly used for unit test or example.
"""

import numpy as np
import pandas
import onnx
import torch
import onnxruntime
from onnx_diagnostic import doc
from onnx_diagnostic.helpers import max_diff, string_diff, string_type
from onnx_diagnostic.helpers.torch_helper import dummy_llm, steal_forward
from onnx_diagnostic.helpers.mini_onnx_builder import create_input_tensors_from_onnx_model
from onnx_diagnostic.reference import OnnxruntimeEvaluator, ReportResultComparison


model, inputs, ds = dummy_llm(dynamic_shapes=True)

# %%
# We use float16.
model = model.to(torch.float16)

# %%
# Let's check.

print(f"type(model)={type(model)}")
print(f"inputs={string_type(inputs, with_shape=True)}")
print(f"ds={string_type(ds, with_shape=True)}")

# %%
# It contains the following submodules.

for name, mod in model.named_modules():
    print(f"- {name}: {type(mod)}")

# %%
# Steal and dump the output of submodules
# +++++++++++++++++++++++++++++++++++++++
#
# The following context spies on the intermediate results
# for the following module and submodules. It stores
# in one onnx file all the input/output for those.

with steal_forward(
    [
        ("model", model),
        ("model.decoder", model.decoder),
        ("model.decoder.attention", model.decoder.attention),
        ("model.decoder.feed_forward", model.decoder.feed_forward),
        ("model.decoder.norm_1", model.decoder.norm_1),
        ("model.decoder.norm_2", model.decoder.norm_2),
    ],
    dump_file="plot_dump_intermediate_results.inputs.onnx",
    verbose=1,
    storage_limit=2**28,
):
    expected = model(*inputs)


# %%
# Restores saved inputs/outputs
# +++++++++++++++++++++++++++++
#
# All the intermediate tensors were saved in one unique onnx model,
# every tensor is stored in a constant node.
# The model can be run with any runtime to restore the inputs
# and function :func:`create_input_tensors_from_onnx_model
# <onnx_diagnostic.helpers.mini_onnx_builder.create_input_tensors_from_onnx_model>`
# can restore their names.

saved_tensors = create_input_tensors_from_onnx_model(
    "plot_dump_intermediate_results.inputs.onnx"
)
for k, v in saved_tensors.items():
    print(f"{k} -- {string_type(v, with_shape=True)}")

# %%
# Let's explained the naming convention.
#
# ::
#
#    ('model.decoder.norm_2', 0, 'I') -- ((T1s2x30x16,),{})
#                |            |   |
#                |            |   +--> input, the format is args, kwargs
#                |            |
#                |            +--> iteration, 0 means the first time the execution
#                |                 went through that module
#                |                 it is possible to call multiple times,
#                |                 the model to store more
#                |
#                +--> the name given to function steal_forward
#
# The same goes for output except ``'I'`` is replaced by ``'O'``.
#
# ::
#
#    ('model.decoder.norm_2', 0, 'O') -- T1s2x30x16
#
# This trick can be used to compare intermediate results coming
# from pytorch to any other implementation of the same model
# as long as it is possible to map the stored inputs/outputs.

# %%
# Conversion to ONNX
# ++++++++++++++++++
#
# The difficult point is to be able to map the saved intermediate
# results to intermediate results in ONNX.
# Let's create the ONNX model.

ep = torch.export.export(model, inputs, dynamic_shapes=ds)
epo = torch.onnx.export(ep)
epo.optimize()
epo.save("plot_dump_intermediate_results.onnx")

# %%
# Discrepancies
# +++++++++++++
#
# We have a torch model, intermediate results and an ONNX graph
# equivalent to the torch model.
# Let's see how we can check the discrepancies.
# First the discrepancies of the whole model.

sess = onnxruntime.InferenceSession(
    "plot_dump_intermediate_results.onnx", providers=["CPUExecutionProvider"]
)
feeds = dict(
    zip([i.name for i in sess.get_inputs()], [t.detach().cpu().numpy() for t in inputs])
)
got = sess.run(None, feeds)
diff = max_diff(expected, got)
print(f"discrepancies torch/ORT: {string_diff(diff)}")

# %%
# What about intermediate results?
# Let's use a runtime still based on :epkg:`onnxruntime`
# running an eager evaluation.

sess_eager = OnnxruntimeEvaluator(
    "plot_dump_intermediate_results.onnx",
    providers=["CPUExecutionProvider"],
    torch_or_numpy=True,
)
feeds_tensor = dict(zip([i.name for i in sess.get_inputs()], inputs))
got = sess_eager.run(None, feeds_tensor)
diff = max_diff(expected, got)
print(f"discrepancies torch/eager ORT: {string_diff(diff)}")

# %%
# They are almost the same. That's good.
# Let's now dig into the intermediate results.
# They are compared to the outputs stored in saved_tensors
# during the execution of the model.
baseline = {}
for k, v in saved_tensors.items():
    if k[-1] == "I":  # inputs are excluded
        continue
    if isinstance(v, torch.Tensor):
        baseline[f"{k[0]}.{k[1]}".replace("model.decoder", "decoder")] = v

report_cmp = ReportResultComparison(baseline)
sess_eager.run(None, feeds_tensor, report_cmp=report_cmp)

# %%
# Let's see the results.

data = report_cmp.data
df = pandas.DataFrame(data)
piv = df.pivot(index=("run_index", "run_name"), columns="ref_name", values="abs")
print(piv)

# %%
# Let's clean a little bit.
piv[piv >= 1] = np.nan
print(piv.dropna(axis=0, how="all"))

# %%
# We can identity which results is mapped to which expected tensor.

# %%
# Picture of the model
# ++++++++++++++++++++

onx = onnx.load("plot_dump_intermediate_results.onnx")
doc.plot_dot(onx)

# %%
doc.plot_legend("steal and dump\nintermediate\nresults", "steal_forward", "blue")
