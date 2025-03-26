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


# %%
# We define a model with a control flow (-> graph break)


class ForwardWithControlFlowTest(torch.nn.Module):
    def forward(self, x):
        if x.sum():
            return x * 2
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

doc.plot_legend("If -> torch.cond", "torch.export.export", "tomato")
