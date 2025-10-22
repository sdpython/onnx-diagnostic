"""
Dynamic Shapes and Broadcasting
===============================

:func:`torch.exoprt.export` makes strict assumption on dynamic shapes
to the generic case. Let's consider two tensors with only one dimension.
``x * y`` allows four configurations:

* ``shape(x) = (1,)`` and ``shape(y) = (1,)``
* ``shape(x) = (1,)`` and ``shape(y) = (p,)``
* ``shape(x) = (q,)`` and ``shape(y) = (1,)``
* ``shape(x) = (p,)`` and ``shape(y) = (p,)``

The expected shape for ``shape(x * y)`` is ``(max(p,q),)``.

Simple Case
+++++++++++

"""

import torch
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from onnx_diagnostic.torch_export_patches import torch_export_patches
from experimental_experiment.torch_interpreter.tracing import CustomTracer


class Model(torch.nn.Module):
    def forward(self, x, y):
        return x * y


Dim = torch.export.Dim

ep = torch.export.export(
    Model(),
    (torch.tensor([2, 3], dtype=torch.float32), torch.tensor([2, 3], dtype=torch.float32)),
    dynamic_shapes=({0: Dim.DYNAMIC}, {0: Dim.DYNAMIC}),
)
print(ep)

# %%
# We see clearly that the export assummed that ``x`` ad ``y`` had the same shape.
# No other configuration seemed to work at export time,
# including ``with torch.fx.experimental._config.patch(backed_size_oblivious=True):``
# the shape of one tensor equal to ``(1,)``.

output = [n for n in ep.graph.nodes if n.op == "output"][0]
print("output is ", output.name, " arg is", output.args[0])

# %%
# The final shape is:

shape = output.args[0][0].meta["val"].shape
print("output shape is ", shape)

# %%
# Tracing
# +++++++
#
# Let's compare with what a simple tracing would do. :class:`CustomTracer
# <experimental_experiment.torch_interpreter.tracing.CustomTracer>`
# is a customize tracer built on top of :class:`torch.fx.Tracer`.

graph = CustomTracer().trace(Model())
print(graph)

# %%
output = [n for n in graph.nodes if n.op == "output"][0]
print("output is ", output.name, " arg is", output.args[0])
print("The tracer leaves no trace:", output.args[0].__dict__)

# %%
# Shape propagation
# +++++++++++++++++

gm = torch.fx.GraphModule(Model(), graph)

shape_env = ShapeEnv()
fake_mode = FakeTensorMode(shape_env=shape_env)
# d1 = shape_env.create_unbacked_symint()
# d2 = shape_env.create_unbacked_symint()
fake_inputs = fake_mode.from_tensor(
    torch.zeros((2,), dtype=torch.float32), static_shapes=False
), fake_mode.from_tensor(torch.zeros((2,), dtype=torch.float32), static_shapes=False)

print("fake_inputs are ", fake_inputs)
res = FakeTensorProp(gm, fake_mode).propagate(*fake_inputs)
print("output is", res)

# %%
# Handle Different Shapes
# +++++++++++++++++++++++

fake_inputs = fake_mode.from_tensor(
    torch.zeros((2,), dtype=torch.float32), static_shapes=False
), fake_mode.from_tensor(torch.zeros((1,), dtype=torch.float32), static_shapes=False)

print("fake_inputs are ", fake_inputs)
res = FakeTensorProp(gm, fake_mode).propagate(*fake_inputs)
print("output is", res)

# %%
# Conclusion
# ++++++++++
#
# We need to give distinct dimensions to get distinct names.

fake_inputs = fake_mode.from_tensor(
    torch.zeros((2,), dtype=torch.float32), static_shapes=False
), fake_mode.from_tensor(torch.zeros((3,), dtype=torch.float32), static_shapes=False)
print("fake_inputs are ", fake_inputs)


# %%
try:
    res = FakeTensorProp(gm, fake_mode).propagate(*fake_inputs)
except Exception as e:
    print(e)

# %%
# By applying the patches:

with torch_export_patches():
    res = FakeTensorProp(gm, fake_mode).propagate(*fake_inputs)
    print("output is", res)

# %%
# This is what we want. Let's go back to :func:`torch.export.export`

with torch_export_patches():
    ep = torch.export.export(
        Model(),
        (
            torch.tensor([2, 3], dtype=torch.float32),
            torch.tensor([2, 3, 4], dtype=torch.float32),
        ),
        dynamic_shapes=({0: Dim.DYNAMIC}, {0: Dim.DYNAMIC}),
    )
    print(ep)

# %%
output = [n for n in ep.graph.nodes if n.op == "output"][0]
print("output is ", output.name, " arg is", output.args[0])
shape = output.args[0][0].meta["val"].shape
print("output shape is ", shape)
