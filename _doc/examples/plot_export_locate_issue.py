"""
.. _l-plot-export-locale-issue:

==================================================
Find and fix an export issue due to dynamic shapes
==================================================

LLMs must be exported with dynamic shapes and it is common that
a static dimension turns into a static ones. The error message from
:epkg:`pytorch` tells the user to define ``TORCH_LOGS="+dynamic"``
but it shows a very long list of messages where we need
to find the string ``range_refined_to_singleton`` and that
does not really indicates where it comes from. The example
shows how to tweak pytorch to get that information until
it gets better.

A model with an export issue
============================

The following model implies the first dimension of x is equal to 1
or equal to the number of element in the list ``ys``.
It is not really dynamic. It looks obvious here but
it is difficult to find deep inside a big model.
"""

import traceback
import torch
from onnx_diagnostic import doc
from onnx_diagnostic.torch_export_patches import torch_export_patches


class ModelWithIssue(torch.nn.Module):
    def forward(self, x: torch.Tensor, ys: list[torch.Tensor]):
        caty = torch.cat([y.unsqueeze(0) for y in ys], axis=0)
        z = x * caty
        return z


inputs = (torch.rand(2, 3, 1), [torch.rand(3, 4), torch.rand(3, 4)])
model = ModelWithIssue()
model(*inputs)


# %%
# Let's export.

DYN = torch.export.Dim.DYNAMIC
dyn_shapes = ({0: DYN, 1: DYN}, [{0: DYN, 1: DYN}, {0: DYN, 1: DYN}])
try:
    ep = torch.export.export(model, inputs, dynamic_shapes=dyn_shapes)
    print(ep)
except Exception as e:
    print("-- ERROR:")
    print(e)

# %%
# The error shows:
#
# .. code-block::
#
#       Constraints violated (L['args'][0][0].size()[0])!
#           For more information, run with TORCH_LOGS="+dynamic".
#       - Not all values of RelaxedUnspecConstraint(L['args'][0][0].size()[0])
#           are valid because L['args'][0][0].size()[0] was inferred to be a constant (2).
#
# Where does it happens? That's a tricky question we need to answer.
# The message is raised from
# `torch.fx.experimental.symbolic_shapes.ShapeEnv._set_replacement
# <https://github.com/pytorch/pytorch/blob/main/torch/fx/experimental/symbolic_shapes.py#L6239>`_.
# One way to find the exact location is to retrieve a stack trace
# by inserting an assert such as the following:
#
# .. code-block::
#
#   assert msg != "range_refined_to_singleton", (
#       f"A dynamic dimension becomes static! "
#       f"a={a!r}, tgt={tgt!r}, msg={msg!r}, tgt_bound={tgt_bound}"
#   )
#
# Stop when a dynamic dimension turns static
# ==========================================
#
# We use :func:`torch_export_patches
# <onnx_diagnostic.torch_export_patches.torch_export_patches>`
# to replace torch implementation by a new one raising the exception
# mentioned in previous section.

with torch_export_patches(stop_if_static=1, verbose=1):
    try:
        torch.export.export(model, inputs, dynamic_shapes=dyn_shapes)
    except (AssertionError, torch._dynamo.exc.TorchRuntimeError) as e:
        print("-- It failed as excepted.")
        print(f"-- final error is {e}")
        print("-- Stack Trace")
        print(traceback.format_exc())

# The stack trace is quite long but the first line referring to this example
# is the following one. It points out the line turing a dynamic dimension into
# static.
#
# .. code-block::
#
#   File "onnx-diagnostic/_doc/examples/plot_export_locate_issue.py", line 25, in forward
#       z = x * caty

# %%

doc.plot_legend(
    "dynamic dimension\nwas inferred\nto be a constant", "torch.export.export", "tomato"
)
