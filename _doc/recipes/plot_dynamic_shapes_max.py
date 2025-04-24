"""
Cannot export ``torch.sym_max(x.shape[0], y.shape[0])``
=======================================================

This is related to the following issues:
`Cannot export torch.sym_max(x.shape[0], y.shape[0])
<https://github.com/pytorch/pytorch/issues/150851>`_.

The algorithm trying to automatically infer shapes after every operator
in the exported program is something very aggreessive. Here is a case where
it takes a wrong decision and how to get around it.

**This bug was fixed after 4/24/2025**.

Wrong Model
+++++++++++
"""

import torch
from onnx_diagnostic import doc


class Model(torch.nn.Module):
    def forward(self, x, y, fact):
        s1 = max(x.shape[0], y.shape[0])
        s2 = max(x.shape[1], y.shape[1])
        # Shapes cannot be known here.
        z = torch.zeros((s1, s2), dtype=x.dtype)
        z[: x.shape[0], : x.shape[1]] = x
        z[: y.shape[0], : y.shape[1]] += y
        return z * fact


model = Model()
x = torch.arange(6).reshape((2, 3))
y = torch.arange(6).reshape((3, 2)) * 10
fact = torch.tensor([[1, 2, 3]], dtype=x.dtype)
z = model(x, y, fact)
print(f"x.shape={x.shape}, y.shape={y.shape}, z.shape={z.shape}")

# %%
# Export
# ++++++
DYN = torch.export.Dim.DYNAMIC

ep = torch.export.export(
    model, (x, y, fact), dynamic_shapes=({0: DYN, 1: DYN}, {0: DYN, 1: DYN}, {1: DYN})
)
print(ep)

# %%
# But does it really work? Let's print the shapes.
model_ep = ep.module()
ez = model_ep(x, y, fact)
print("case 1:", z.shape, ez.shape)

# %%
# Case with different shapes.

x = torch.arange(4).reshape((2, 2))
y = torch.arange(9).reshape((3, 3))
try:
    ez = model_ep(x, y, fact)
    print("case 2:", model(x, y, fact).shape, ez.shape)
except Exception as e:
    print("case 2 failed:", e)

# %%
# It does not even compute. The exported program does not get the correct shape.
#
# Rewritten Model
# +++++++++++++++
#
# ``max`` does not get captured, :func:`torch.sym_max` is no better,
# :func:`torch.max` only works on tensors. Nothing really works.
# We use a trick to introduce new shape the shape inference algorithm
# cannot know. This requires to hide the failing logic in a custom operator.


def make_undefined_dimension(i: int) -> torch.SymInt:
    """
    Uses for a custom op when a new dimension must be introduced to bypass
    some verification. The following function creates a dummy output
    with a dimension based on the content.

    .. code-block:: python

        def symbolic_shape(x, y):
            return torch.empty(
                x.shape[0],
                make_undefined_dimension(min(x.shape[1], y[0])),
            )
    """
    t = torch.ones((i * 2,))
    t[:i] = 0
    res = torch.nonzero(t).shape[0]
    return res


def copy_max_dimensions(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    shape = torch.max(torch.tensor(x.shape), torch.tensor(y.shape))
    z = torch.zeros(tuple(shape), dtype=x.dtype)
    z[0 : x.shape[0], 0 : x.shape[1]] = x[0 : x.shape[0], 0 : x.shape[1]]
    z[0 : y.shape[0], 0 : y.shape[1]] += y[0 : y.shape[0], 0 : y.shape[1]]
    return z


def symbolic_shape(x, y):
    return torch.empty(
        tuple(
            make_undefined_dimension(max(x.shape[i], y.shape[i])) for i in range(len(x.shape))
        ),
        dtype=x.dtype,
    )


def register(fct, fct_shape, namespace, fname):
    schema_str = torch.library.infer_schema(fct, mutates_args=())
    custom_def = torch.library.CustomOpDef(namespace, fname, schema_str, fct)
    custom_def.register_kernel("cpu")(fct)
    custom_def._abstract_fn = fct_shape


register(
    copy_max_dimensions, lambda x, y: symbolic_shape(x, y), "mylib", "copy_max_dimensions"
)

# %%
# Now everything is registered. Let's rewrite the model.


class RewrittenModel(torch.nn.Module):
    def forward(self, x, y, fact):
        z = torch.ops.mylib.copy_max_dimensions(x, y)
        return z * fact


# %%
# And check it works.

rewritten_model = RewrittenModel()
x = torch.arange(6).reshape((2, 3))
y = torch.arange(6).reshape((3, 2)) * 10
z = rewritten_model(x, y, fact)
print(f"x.shape={x.shape}, y.shape={y.shape}, z.shape={z.shape}")

# %%
# Export again
# ++++++++++++

ep = torch.export.export(
    rewritten_model,
    (x, y, fact),
    dynamic_shapes=({0: DYN, 1: DYN}, {0: DYN, 1: DYN}, {1: DYN}),
)
print(ep)

# %%
# We check it works.

model_ep = ep.module()
ez = model_ep(x, y, fact)
print("case 1:", z.shape, ez.shape)

x = torch.arange(4).reshape((2, 2))
y = torch.arange(9).reshape((3, 3))
try:
    ez = model_ep(x, y, fact)
    print("case 2:", rewritten_model(x, y, fact).shape, ez.shape)
except Exception as e:
    print("case 2 failed:", e)

# %%
# Final Check on very different dimension
# +++++++++++++++++++++++++++++++++++++++

x = torch.arange(6 * 8).reshape((6, 8))
y = torch.arange(10 * 4).reshape((10, 4)) * 10
fact = torch.arange(8).reshape((1, -1))

print("final case:", rewritten_model(x, y, fact).shape, model_ep(x, y, fact).shape)

# %%
# This is not perfect as we get an exported program but some logic
# is hidden in a custom operator.


doc.plot_legend("max(d1, d2)\nwith d1, d2 dimensions", "dynamic shapes", "green")
