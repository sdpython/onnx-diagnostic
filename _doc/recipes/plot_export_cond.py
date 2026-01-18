"""
.. _l-plot-export-cond:

Export a model with a control flow (If)
=======================================

Control flow cannot be exported with a change.
The code of the model can be changed or patched
to introduce function :func:`torch.cond`.

A model with a test
+++++++++++++++++++
"""

import torch
from onnx_diagnostic import doc
from onnx_diagnostic.torch_export_patches import torch_export_rewrite

# %%
# We define a model with a control flow (-> graph break)


class ForwardWithControlFlowTest(torch.nn.Module):
    def forward(self, x):
        if x.sum():
            return x * 2
        else:
            return -x


class ModelWithControlFlow(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(3, 2),
            torch.nn.Linear(2, 1),
            ForwardWithControlFlowTest(),
        )

    def forward(self, x):
        out = self.mlp(x)
        return out


model = ModelWithControlFlow()

# %%
# Let's check it runs.
x = torch.randn(1, 3)
model(x)

# %%
# As expected, it does not export.
try:
    torch.export.export(model, (x,))
    raise AssertionError("This export should failed unless pytorch now supports this model.")
except Exception as e:
    print(e)


# %%
# Suggested Patch
# +++++++++++++++
#
# Let's avoid the graph break by replacing the forward.


def new_forward(x):
    def identity2(x):
        return x * 2

    def neg(x):
        return -x

    return torch.cond(x.sum() > 0, identity2, neg, (x,))


print("the list of submodules")
for name, mod in model.named_modules():
    print(name, type(mod))
    if isinstance(mod, ForwardWithControlFlowTest):
        mod.forward = new_forward

# %%
# Let's see what the fx graph looks like.

ep = torch.export.export(model, (x,))
print(ep.graph)

# %%
# Automated Rewrite of the Control Flow
# +++++++++++++++++++++++++++++++++++++
#
# Functions :func:`torch_export_rewrite
# <onnx_diagnostic.torch_export_patches.torch_export_rewrite>`
# or :func:`torch_export_patches <onnx_diagnostic.torch_export_patches.torch_export_patches>`
# can automatically rewrite a method of a class or a function,
# the method to rewrite is specified parameter ``rewrite``.
# It is experimental. The function contains options to
# rewrite one test but not another one already supported by the exporter.
# It may give a first version of the rewritten code if only a manual
# rewriting can make the model exportable.

with torch_export_rewrite(rewrite=[ForwardWithControlFlowTest.forward], verbose=2) as f:
    ep = torch.export.export(model, (x,))

# %%
# This gives:

print(ep.graph)

# %%

doc.plot_legend("If -> torch.cond", "torch.export.export", "yellowgreen")
