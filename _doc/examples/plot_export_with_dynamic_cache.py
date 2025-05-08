"""
.. _l-plot-export-with-dynamic-shape:

===================================================
Export with DynamicCache and guessed dynamic shapes
===================================================

Every LLMs implemented in :epkg:`transformers` use cache.
One of the most used is :class:`transformers.cache_utils.DynamicCache`.
The cache size is dynamic to cope with the growing context.
The example shows a tool which determines the dynamic shapes
for :func:`torch.export.export` based on a set of valid inputs.

DynamicCache
============

:func:`torch.export.export` serializes caches and any custom class
if these serialization functions are provided with is the case for
:class:`transformers.cache_utils.DynamicCache` and ``transformers>=4.50``.
The dynamic shapes must be provided following the serialized form.
"""

import pprint
import torch
from onnx_diagnostic import doc
from onnx_diagnostic.ext_test_case import has_transformers
from onnx_diagnostic.helpers import string_type
from onnx_diagnostic.helpers.cache_helper import (
    flatten_unflatten_for_dynamic_shapes,
    make_dynamic_cache,
)
from onnx_diagnostic.export import ModelInputs
from onnx_diagnostic.torch_export_patches import torch_export_patches


class Model(torch.nn.Module):
    def forward(self, cache, z):
        return (
            z
            + cache.key_cache[0]
            + cache.key_cache[1]
            + cache.value_cache[0]
            + cache.value_cache[1]
        )


model = Model()

n_layers = 2
bsize, nheads, slen, dim = 2, 4, 3, 7
cache = make_dynamic_cache(
    [
        (torch.randn(bsize, nheads, slen, dim), torch.randn(bsize, nheads, slen, dim))
        for i in range(n_layers)
    ]
)
z = torch.randn((1, 1, 1, 7))
model(cache, z)  # to check it works.

# %%
# The cache looks like this:

print(string_type(cache, with_shape=True))


# %% Let's create another set of inputs.

cache2 = make_dynamic_cache(
    [
        (
            torch.randn(bsize + 1, nheads, slen + 1, dim + 1),
            torch.randn(bsize + 1, nheads, slen + 1, dim + 1),
        )
        for i in range(n_layers)
    ]
)
inputs = [
    (cache, z),
    (cache2, torch.randn((1, 1, 1, 8))),
]

# %%
# And the second set of inputs looks like:
print(string_type(inputs[1], with_shape=True))

# %%
# .. _l-guess-dynamic-shapes-example:
#
# Guess the dynamic shapes
# ========================
#
# The following tool can be used to guess the dynamic shapes
# the way :func:`torch.export.export` expects them.

mi = ModelInputs(Model(), inputs)
ds = mi.guess_dynamic_shapes()

pprint.pprint(ds)

# %%
# And finally the export.
# The export is simple if ``transformers>=4.50``, otherwise,
# transformers needs to be patched.
# :func:`onnx_diagnostic.torch_export_patches.torch_export_patches`
# registers functions to serialize ``DynamicCache``. This one is modified to make
# the shape inference implemented in :epkg:`torch` happy.

if has_transformers("4.50"):
    ep = torch.export.export(model, inputs[0], dynamic_shapes=ds[0], strict=False)
else:
    with torch_export_patches(patch_transformers=True) as modificator:
        ep = torch.export.export(
            model, modificator(inputs[0]), dynamic_shapes=ds[0], strict=False
        )
print(ep)

# %%
# Use string instead of DYNAMIC
# +++++++++++++++++++++++++++++
#
# ONNX exporter considers strings instead of DYNAMIC or AUTO
# to give names to every dimension.

dss = mi.guess_dynamic_shapes(auto="dim")
pprint.pprint(dss)


# %%
# Do we need to guess?
# ++++++++++++++++++++
#
# Function :func:`onnx_diagnostic.helpers.string_type` is using
# the serialization functions to print out the DynamicCache the was
# :func:`torch.export.export` expects them.

print(string_type(cache, with_shape=True))

# %%
# You can also use function
# :func:`onnx_diagnostic.helpers.cache_helper.flatten_unflatten_for_dynamic_shapes`
# to show a DynamicCache restructured the way :func:`torch.export.export` expects
# it to be without the custom class.

print(string_type(flatten_unflatten_for_dynamic_shapes(cache), with_shape=True))

# %%
# This code works for any custom class if it was registered
# with :func:`torch.utils._pytree.register_pytree_node`.


doc.plot_legend("dynamic shapes\nfor DynamicCache", "torch.export.export", "tomato")
