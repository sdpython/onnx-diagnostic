"""
.. _l-plot-export-with-args-kwargs:

==========================================
Dynamic Shapes for ``*args``, ``**kwargs``
==========================================

Quick tour of dynamic shapes.
We first look at examples playing positional and names parameters
to understand how :func:`torch.export.export` works.

args
====
"""

import pprint
import torch
from onnx_diagnostic import doc
from onnx_diagnostic.export import ModelInputs


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
# ======
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
# ===============
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

doc.plot_legend("dynamic shapes\n*args, **kwargs", "torch.export.export", "tomato")
