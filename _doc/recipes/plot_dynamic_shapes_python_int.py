"""
.. _l-plot-dynamic-shapes-python-int:

Do not use python int with dynamic shapes
=========================================

:func:`torch.export.export` uses :class:`torch.SymInt` to operate on shapes and
optimizes the graph it produces. It checks if two tensors share the same dimension,
if the shapes can be broadcast, ... To do that, python types must not be used
or the algorithm looses information.

Wrong Model
+++++++++++
"""

import math
import torch
from onnx_diagnostic import doc
from onnx_diagnostic.torch_export_patches import torch_export_patches


class Model(torch.nn.Module):
    def dim(self, i, divisor):
        return int(math.ceil(i / divisor))  # noqa: RUF046

    def forward(self, x):
        new_shape = (self.dim(x.shape[0], 8), x.shape[1])
        return torch.zeros(new_shape)


model = Model()
x = torch.rand((10, 15))
y = model(x)
print(f"x.shape={x.shape}, y.shape={y.shape}")

# %%
# Export
# ++++++

DYN = torch.export.Dim.DYNAMIC
ep = torch.export.export(model, (x,), dynamic_shapes=(({0: DYN, 1: DYN}),))
print(ep)

# %%
# The last dimension became static. We must not use int.
# :func:`math.ceil` should be avoided as well since it is a python operation.
# The exporter may fail to detect it is operating on shapes.
#
# Rewrite
# +++++++


class RewrittenModel(torch.nn.Module):
    def dim(self, i, divisor):
        return (i + divisor - 1) // divisor

    def forward(self, x):
        new_shape = (self.dim(x.shape[0], 8), x.shape[1])
        return torch.zeros(new_shape)


rewritten_model = RewrittenModel()
y = rewritten_model(x)
print(f"x.shape={x.shape}, y.shape={y.shape}")

# %%
# Export
# ++++++

ep = torch.export.export(rewritten_model, (x,), dynamic_shapes=({0: DYN, 1: DYN},))
print(ep)


# %%
# Find the error
# ++++++++++++++
#
# Function :func:`onnx_diagnostic.torch_export_patches.torch_export_patches`
# has a parameter ``stop_if_static`` which patches torch to raise exception
# when something like that is happening.


with torch_export_patches(stop_if_static=True):
    ep = torch.export.export(model, (x,), dynamic_shapes=({0: DYN, 1: DYN},))
    print(ep)

# %%
doc.plot_legend("dynamic shapes\ndo not cast to\npython int", "dynamic shapes", "yellow")
