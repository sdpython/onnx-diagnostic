"""
Builds dynamic shapes from any input
====================================

Getting dynamic shapes right for :func:`torch.export.export` when the inputs
includes a custom class such as a :class:`transformers.cache_utils.DynamicCache`.
:func:`torch.export.export` cannot use a DynamicCache filled with dynamic shapes
but instead it uses a kind of unserialized serialized form of it.

Standard inputs for a LLM with a dynamic cache
++++++++++++++++++++++++++++++++++++++++++++++
"""

import pprint
import torch
from onnx_diagnostic import doc
from onnx_diagnostic.helpers import string_type
from onnx_diagnostic.helpers.cache_helper import make_dynamic_cache
from onnx_diagnostic.export.shape_helper import all_dynamic_shape_from_inputs
from onnx_diagnostic.torch_models.hghub import get_untrained_model_with_inputs
from onnx_diagnostic.torch_export_patches import torch_export_patches

bsize, nheads, slen, dim = 2, 1, 30, 96

inputs = dict(
    input_ids=torch.randint(15, size=(2, 3), dtype=torch.int64),
    attention_mask=torch.randint(1, size=(2, 33), dtype=torch.int64),
    position_ids=torch.arange(3, dtype=torch.int64),
    past_key_values=make_dynamic_cache(
        [(torch.randn(bsize, nheads, slen, dim), torch.randn(bsize, nheads, slen, dim))]
    ),
)

print(string_type(inputs, with_shape=True))

# %%
# Function :func:`onnx_diagnostic.export.shape_helper.all_dynamic_shape_from_inputs`
# produces the corresponding dynamic shapes assuming they are all dynamic.
ds = all_dynamic_shape_from_inputs(inputs)
pprint.pprint(ds)

# %%
# What about a StaticCache?
# +++++++++++++++++++++++++
#
# We use :func:`onnx_diagnostic.torch_models.hghub.get_untrained_model_with_inputs` to get
# a consistent configuration with a static cache.

data = get_untrained_model_with_inputs(
    "arnir0/Tiny-LLM",
    model_kwargs=dict(cache_implementation="static"),
    inputs_kwargs=dict(cls_cache="StaticCache"),
)
inputs = data["inputs"]
print(string_type(inputs, with_shape=True))

# %%
# And the input shapes.
ds = all_dynamic_shape_from_inputs(inputs)
if ds["past_key_values"]:
    print("transformers implemented serialization function for StaticCache.")
else:
    print("We need to use serialization function implemented in this package.")
    with torch_export_patches(patch_transformers=True):
        ds = all_dynamic_shape_from_inputs(inputs)

# %%
# That gives.
pprint.pprint(ds)

# %%
# We can compare with the ones returned by the function.
pprint.pprint(data["dynamic_shapes"])


# %%

doc.plot_legend("dynamic shapes\nfrom inputs", "dynamic shapes", "green")
