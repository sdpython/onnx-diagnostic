"""
.. _l-plot-export-dim1:

0, 1, 2 for a Dynamic Dimension in the dummy example to export a model
======================================================================

:func:`torch.export.export` does not work if a tensor given to the function
has 0 or 1 for dimension declared as dynamic dimension.

Simple model, no dimension with 0 or 1
++++++++++++++++++++++++++++++++++++++
"""

import torch
from onnx_diagnostic import doc


class Model(torch.nn.Module):
    def forward(self, x, y, z):
        return torch.cat((x, y), axis=1) + z


model = Model()
x = torch.randn(2, 3)
y = torch.randn(2, 5)
z = torch.randn(2, 8)
model(x, y, z)

DYN = torch.export.Dim.DYNAMIC
ds = {0: DYN, 1: DYN}

ep = torch.export.export(model, (x, y, z), dynamic_shapes=(ds, ds, ds))
print(ep.graph)

# %%
# Same model, a dynamic dimension = 1
# +++++++++++++++++++++++++++++++++++

z = z[:1]

DYN = torch.export.Dim.DYNAMIC
ds = {0: DYN, 1: DYN}

try:
    ep = torch.export.export(model, (x, y, z), dynamic_shapes=(ds, ds, ds))
    print(ep.graph)
except Exception as e:
    print("ERROR", e)

# %%
# It failed. Let's try a little trick.

# %%
# Same model, a dynamic dimension = 1 and backed_size_oblivious=True
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

try:
    with torch.fx.experimental._config.patch(backed_size_oblivious=True):
        ep = torch.export.export(model, (x, y, z), dynamic_shapes=(ds, ds, ds))
        print(ep.graph)
except RuntimeError as e:
    print("ERROR", e)

# %%
# It worked.

doc.plot_legend("dynamic dimension\nworking with\n0 or 1", "torch.export.export", "green")
