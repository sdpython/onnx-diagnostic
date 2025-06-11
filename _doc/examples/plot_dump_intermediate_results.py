"""
.. _l-plot-intermediate-results:

Dumps intermediate results of a torch model
===========================================


codellama/CodeLlama-7b-Python-hf
++++++++++++++++++++++++++++++++

"""

import onnx
import torch
from onnx_array_api.plotting.graphviz_helper import plot_dot
from onnx_diagnostic import doc
from onnx_diagnostic.helpers import string_type
from onnx_diagnostic.helpers.torch_helper import dummy_llm
from onnx_diagnostic.helpers.mini_onnx_builder import create_input_tensors_from_onnx_model
from onnx_diagnostic.helpers.torch_helper import steal_forward


model, inputs, ds = dummy_llm(dynamic_shapes=True)

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
    model(*inputs)


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
#                +--> the name given to steal forward
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

epo = torch.onnx.export(model, inputs, dynamic_shapes=ds, dynamo=True)
epo.optimize()
epo.save("plot_dump_intermediate_results.onnx")

# %%
# It looks like the following.
onx = onnx.load("plot_dump_intermediate_results.onnx")
plot_dot(onx)

# %%
doc.plot_legend("steal and dump\nintermediate\nresults", "steal_forward", "blue")
