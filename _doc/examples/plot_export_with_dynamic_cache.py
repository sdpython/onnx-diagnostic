"""
.. _l-plot-export-with-dynamic-shape:

===========================================
Export with DynamicCache and dynamic shapes
===========================================

Every LLMs implemented in :epkg:`transformers` use cache.
One of the most used is :class:`transformers.cache_utils.DynamicCache`.
The cache size is dynamic to cope with the growing context.
The example shows a tool which determines the dynamic shapes
for :func:`torch.export.export` based on a set of valid inputs.

Simple Examples
===============

We first look at examples playing positional and names parameters
to understand how :func:`torch.export.export` works.

args
++++
"""

import pprint
import torch
from onnx_diagnostic import doc
from onnx_diagnostic.cache_helpers import make_dynamic_cache
from onnx_diagnostic.helpers import string_type
from onnx_diagnostic.export import ModelInputs

# %%
# We need addition import in case ``transformers<4.50``.
# Exporting DynamicCache is not supported before that.
from onnx_diagnostic.ext_test_case import has_transformers
from onnx_diagnostic.torch_export_patches import bypass_export_some_errors


class Model(torch.nn.Module):
    def forward(self, x, y):
        return x + y


model = Model()
x = torch.randn((5, 6))
y = torch.randn((1, 6))
model(x, y)  # to check it works

ep = torch.export.export(model, (x, y))
print(ep)

# %%
# As expected there is no dynamic shapes.
# We use :class:`onnx_diagnostic.export.ModelInputs`
# to define them from two set of valid inputs.
# These inputs must have different value for the dynamic
# dimensions.

inputs = [(x, y), (torch.randn((7, 8)), torch.randn((1, 8)))]
mi = ModelInputs(Model(), inputs)
ds = mi.guess_dynamic_shapes()
pprint.pprint(ds)

# %%
# The function returns a tuple with two objects.
# The first one for the positional arguments, the other one
# for the named arguments. There is no named arguments. We
# we used the first result to export.

ep = torch.export.export(model, (x, y), dynamic_shapes=ds[0])
print(ep)

# %%
# kwargs
# ++++++
#
# We do the same with named arguments.


class Model(torch.nn.Module):
    def forward(self, x, y):
        return x + y


model = Model()
x = torch.randn((5, 6))
y = torch.randn((1, 6))
model(x=x, y=y)  # to check it works

# %%
# Two sets of valid inputs.
inputs = [dict(x=x, y=y), dict(x=torch.randn((7, 8)), y=torch.randn((1, 8)))]
mi = ModelInputs(Model(), inputs)
ds = mi.guess_dynamic_shapes()
pprint.pprint(ds)

# %%
# And we export.
ep = torch.export.export(model, (), kwargs=dict(x=x, y=y), dynamic_shapes=ds[1])
print(ep)

# %%
# args and kwargs
# +++++++++++++++
#
# :func:`torch.export.export` does not like having dynami shapes
# for both args and kwargs. We need to define them using one mechanism.


class Model(torch.nn.Module):
    def forward(self, x, y):
        return x + y


model = Model()
x = torch.randn((5, 6))
y = torch.randn((1, 6))
model(x, y=y)  # to check it works

# %%
# Two sets of valid inputs with positional and names arguments.

inputs = [((x,), dict(y=y)), ((torch.randn((7, 8)),), dict(y=torch.randn((1, 8))))]
mi = ModelInputs(Model(), inputs)
ds = mi.guess_dynamic_shapes()
pprint.pprint(ds)

# %%
# This does not work with :func:`torch.export.export` so
# we use a method to move the positional dynamic shapes to
# named one. The method relies on the signature of the
# forward method.

new_args, new_kwargs, new_ds = mi.move_to_kwargs(*mi.inputs[0], ds)
pprint.pprint(new_ds)

# %%
# And we export.

ep = torch.export.export(model, new_args, kwargs=new_kwargs, dynamic_shapes=new_ds[1])
print(ep)

# %%
# DynamicCache
# ============
#
# :func:`torch.export.export` serializes caches and any custom class
# if these serialization functions are provided with is the case for
# :class:`transformers.cache_utils.DynamicCache` and ``transformers>=4.50``.
# The dynamic shapes must be provided following the serialized form.


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
# And the first set of inputs looks like:
print(string_type(inputs[0], with_shape=True))

# %%
# We can now compute the dynamic shapes.

mi = ModelInputs(Model(), inputs)
ds = mi.guess_dynamic_shapes()
pprint.pprint(ds)

# %%
# And finally the export.
# The export is simple if ``transformers>=4.50``, otherwise,
# transformers needs to be patched.
# :func:`onnx_diagnostic.torch_export_patches.bypass_export_some_errors`
# registers functions to serialize ``DynamicCache``. This one is modified to make
# the shape inference implemented in :epkg:`torch` happy.

if has_transformers("4.50"):
    ep = torch.export.export(model, inputs[0], dynamic_shapes=ds[0], strict=False)
else:
    with bypass_export_some_errors(patch_transformers=True) as modificator:
        ep = torch.export.export(
            model, modificator(inputs[0]), dynamic_shapes=ds[0], strict=False
        )
print(ep)

# %%

doc.plot_legend("dynamic shapes", "torch.export.export", "tomato")
