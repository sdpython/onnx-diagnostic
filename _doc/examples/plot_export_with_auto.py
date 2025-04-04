"""
.. _l-plot-sxport-with-auto:

Use DYNAMIC or AUTO when exporting if dynamic shapes has constraints
====================================================================

Setting the dynamic shapes is not always easy.
Here are a few tricks to make it work.

dx + dy not allowed?
++++++++++++++++++++
"""

import torch
from onnx_diagnostic import doc


class Model(torch.nn.Module):
    def forward(self, x, y, z):
        return torch.cat((x, y), axis=1) + z[:, ::2]


model = Model()
x = torch.randn(2, 3)
y = torch.randn(2, 5)
z = torch.randn(2, 16)
model(x, y, z)


print(torch.export.export(model, (x, y, z)).graph)

# %%
# Everything is fine so far. With dynamic shapes now.
# dx + dy is not allowed...

batch = torch.export.Dim("batch")
dx = torch.export.Dim("dx")
dy = torch.export.Dim("dy")

try:
    dz = dx + dy
    raise AssertionError("able to add dynamic dimensions, please update the tutorial")
except NotImplementedError as e:
    print(f"unable to add dynamic dimensions because {type(e)}, {e}")

# %%
# Then we could make it a different one.

dz = torch.export.Dim("dz")
try:
    ep = torch.export.export(
        model,
        (x, y, z),
        dynamic_shapes={
            "x": {0: batch, 1: dx},
            "y": {0: batch, 1: dy},
            "z": {0: batch, 1: dz},
        },
    )
    print(ep)
    raise AssertionError("able to export this model, please update the tutorial")
except torch._dynamo.exc.UserError as e:
    print(f"unable to use Dim('dz') because {type(e)}, {e}")

# %%
# That works. We could also use
# ``torch.export.Dim.DYNAMIC`` or ``torch.export.Dim.AUTO``
# for the dimension we cannot set.

DYNAMIC = torch.export.Dim.DYNAMIC
ep = torch.export.export(
    model,
    (x, y, z),
    dynamic_shapes={
        "x": {0: DYNAMIC, 1: dx},
        "y": {0: DYNAMIC, 1: dy},
        "z": {0: DYNAMIC, 1: DYNAMIC},
    },
)

print(ep)

# %%
# The same result can be obtained with ``torch.export.Dim.AUTO``.

AUTO = torch.export.Dim.AUTO
ep = torch.export.export(
    model,
    (x, y, z),
    dynamic_shapes=({0: AUTO, 1: AUTO}, {0: AUTO, 1: AUTO}, {0: AUTO, 1: AUTO}),
)
print(ep)

# %%

doc.plot_legend("torch.export.Dim\nor DYNAMIC\nor AUTO", "torch.export.export", "tomato")
