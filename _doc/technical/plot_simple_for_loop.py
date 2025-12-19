"""
.. _l-plot-simple-for-loop:

Export with loops
=================

This is a simple example of loop which cannot be efficiently rewritten
with ``scan``.
"""

import torch
from onnx_diagnostic import doc
from onnx_diagnostic.export.cf_simple_loop_for import simple_loop_for


class Model(torch.nn.Module):
    def __init__(self, crop_size):
        super().__init__()
        self.crop_size = crop_size

    def forward(self, W, splits):
        crop_size = self.crop_size
        starts = splits[:-1]
        ends = splits[1:]
        cropped = []
        for start, end in zip(starts, ends):
            extract = W[:, start:end]
            if extract.shape[1] < crop_size:
                cropped.append(extract)
            else:
                cropped.append(extract[:, :crop_size])
        return torch.cat(cropped, axis=1)


model = Model(4)
args = (torch.rand((2, 22)), torch.tensor([0, 5, 15, 20, 22], dtype=torch.int64))

expected = model(*args)
print(f"-- expected shape: {expected.shape}")


# %%
# Rewrite with higher order ops scan
# ++++++++++++++++++++++++++++++++++
#
# The loop cannot be exported as is. It needs to be rewritten.


class ModelWithScan(Model):
    def forward(self, W, splits):
        crop_size = self.crop_size
        starts = splits[:-1]
        ends = splits[1:]

        def body_scan(init, split, W):
            extract = W[:, split[0].item() : split[1].item()]
            cropped = extract[:, : torch.sym_min(extract.shape[1], crop_size)]
            carried = torch.cat([init, cropped], axis=1)
            return carried

        starts_ends = torch.cat([starts.unsqueeze(1), ends.unsqueeze(1)], axis=1)
        return torch.ops.higher_order.scan(
            body_scan, [torch.empty((W.shape[0], 0), dtype=W.dtype)], [starts_ends], [W]
        )


rewritten_model_with_scan = ModelWithScan(4)
(results,) = rewritten_model_with_scan(*args)

print(f"-- max discrepancies with scan: { torch.abs(expected - results).max()}")

# %%
# This approach has one flaw, the variable carried grows at every
# iteration and the cost of the copy is quadratic when the same operation
# in the first model is linear.
# We cannot simply return variable ``cropped`` because its shape
# is not always the same.
#
# Introduce of a new higher order ops: simple_loop_for
# ++++++++++++++++++++++++++++++++++++++++++++++++++++
#
# ``simple_loop_for`` was designed to support this specific case.
# It takes all the outputs coming from the body function and stores
# them in list. Then it contenates them according to ``concatenation_dims``.


class ModelWithLoop(Model):
    def forward(self, W, splits):
        crop_size = self.crop_size
        starts = splits[:-1]
        ends = splits[1:]

        def body_loop(i, splits, W):
            split = splits[i.item() : (i + 1).item()][0]  # [i.item()] fails
            extract = W[:, split[0].item() : split[1].item()]
            cropped = extract[:, : torch.sym_min(extract.shape[1], crop_size)]
            return (cropped,)

        starts_ends = torch.cat([starts.unsqueeze(1), ends.unsqueeze(1)], axis=1)
        n_iterations = torch.tensor(starts_ends.shape[0], dtype=torch.int64)
        return simple_loop_for(
            n_iterations, body_loop, (starts_ends, W), concatenation_dims=[1]
        )


rewritten_model_with_loop = ModelWithLoop(4)
results = rewritten_model_with_loop(*args)

print(f"-- max discrepancies with loop: { torch.abs(expected - results).max()}")


# %%
# torch.export.export?
# ++++++++++++++++++++

dynamic_shapes = (
    {0: torch.export.Dim.DYNAMIC, 1: torch.export.Dim.DYNAMIC},
    {0: torch.export.Dim.DYNAMIC},
)
try:
    ep = torch.export.export(rewritten_model_with_scan, args, dynamic_shapes=dynamic_shapes)
    print("----- exported program with scan:")
    print(ep)
except Exception as e:
    print(f"export failed due to {e}")

# %%
# And loops?


ep = torch.export.export(rewritten_model_with_loop, args, dynamic_shapes=dynamic_shapes)
print(ep)

# %%

doc.plot_legend("export a loop\nreturning\ndifferent shapes", "loops", "green")
