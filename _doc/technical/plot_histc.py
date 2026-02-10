"""
.. _l-plot-histc:

================================
Converting torch.histc into ONNX
================================

:func:`torch.histc` computes an histogram of a tensor,
it counts the number of elements falling into each bin.
There are many options do to this. If the number of bins
is not too high, we can use something based on braodcasting.
This method implies the creation of a matrix :math:`N \times B`
where *N* is the number of elements in a tensor and *B* the number
if bins. To avoid this, the best way is to use a tree.
Before doing that, let's first study :func:`torch.histc`.
See `issue 174668 <https://github.com/pytorch/pytorch/issues/174668>`_.

float32 and float16
===================
"""

import matplotlib.pyplot as plt
import torch


def create_input(dtype, hmin, hmax):
    inf = torch.tensor(torch.inf, dtype=torch.float16)
    buffer = torch.tensor([hmin], dtype=torch.float16)
    res = []
    while buffer[0] <= hmax:
        buffer = torch.nextafter(buffer, inf)
        res.append(buffer[0])
    return torch.tensor(res, dtype=dtype)


hbins, hmin, hmax = 20, -5, 5
dtype = torch.float16
tensor = create_input(dtype, hmin, hmax)
print(f"There are {tensor.shape} elements in [{hmin}, {hmax}] of type {torch.float16}).")

# %%
# histc

hist = torch.histc(tensor, hbins, hmin, hmax)
print(f"{hist=}")

# %%
# We can see there are more elements in the center.


def torch_histc_equivalent(tensor, bins, fmin, fmax, thresholds=None):
    # thresholds
    if thresholds is None:
        delta = (float(fmax) - float(fmin)) / float(bins)
        inf = torch.tensor(torch.inf, dtype=tensor.dtype)
        delta = torch.tensor(delta, dtype=tensor.dtype)
        min = torch.tensor(fmin, dtype=tensor.dtype)
        max = torch.tensor(fmax, dtype=tensor.dtype)
        bins = int(bins)
        thresholds = torch.zeros((bins + 1,), dtype=tensor.dtype)
        halfway = bins + 1 - (bins + 1) // 2
        for i in range(halfway):
            thresholds[i] = min + delta * i
        for i in range(halfway, bins + 1):
            thresholds[i] = max - delta * (bins - i)
        thresholds[-1] = torch.nextafter(thresholds[-1], inf)

    # computation
    value = thresholds.unsqueeze(1) < tensor.reshape((-1,)).unsqueeze(0)
    value = value.sum(dim=1).squeeze()
    res = value[:-1] - value[1:]
    res = res.to(torch.float16)
    return res


hist_equiv = torch_histc_equivalent(tensor, hbins, hmin, hmax)
print(f"{hist_equiv=}")
print(f"delta={(hist_equiv - hist).to(int)}")

# %%

diff = torch.abs(hist_equiv - hist).sum()
print(f"sum of differences {diff} with {dtype=}.")

# %%
# This is not really satisfactory.
# Let's check with float32.

hist32 = torch.histc(tensor.to(torch.float32), hbins, hmin, hmax)
hist32_equiv = torch_histc_equivalent(tensor.to(torch.float32), hbins, hmin, hmax)
diff32 = hist32_equiv - hist32
print(f"{diff32.abs().sum()} are misplaced: {diff32=}.")

# %%
# Is histc an increasing function?
# ++++++++++++++++++++++++++++++++

histc_index = torch.empty(tensor.shape, dtype=torch.float64)
buffer = torch.empty((1,), dtype=tensor.dtype)
for i in range(tensor.shape[0]):
    buffer[0] = tensor[i]
    histc_value = torch.histc(buffer, hbins, hmin, hmax)
    histc_index[i] = (
        histc_value.argmax() if histc_value.max().item() > 0 else histc_index.max()
    )


fig, ax = plt.subplots(1, 1)
ax.plot(list(range(tensor.shape[0])), histc_index.tolist(), "-", label="histc_index")
ax.legend()
fig.savefig("plot_histc_index.png")
ax

# %%
# It seems growing. Let's check.


diff = histc_index[1:] - histc_index[:-1]
print(f"min={diff.min()}, max={diff.max()}")

# %%
# It is so we can find threshold working with the implementation we made.
#
# Better thresholds
# =================


def tune_threshold_histc(
    dtype: torch.dtype, hbin: int, hmin: float, hmax: float
) -> torch.Tensor:
    possible_values = create_input(dtype, hmin, hmax)
    buffer = torch.empty((1,), dtype=tensor.dtype)
    previous_index = None
    thresholds = []
    for i in range(tensor.shape[0]):
        buffer[0] = tensor[i]
        histc_value = torch.histc(buffer, hbins, hmin, hmax)
        if histc_value.max().item() > 0:
            index = histc_value.argmax()
            if previous_index is None or index != previous_index:
                previous_index = index
                thresholds.append(possible_values[i])

    thresholds.append(
        torch.nextafter(torch.tensor(hmax, dtype=dtype), torch.tensor(torch.inf, dtype=dtype))
    )
    return torch.tensor(thresholds, dtype=tensor.dtype)


thresholds = tune_threshold_histc(torch.float16, hbins, hmin, hmax)
print(f"shape={thresholds.shape}: {thresholds=}")

# %%
# Let's check it is working.

hist_equiv = torch_histc_equivalent(tensor, hbins, hmin, hmax, thresholds=thresholds)
print(f"{hist_equiv=}")
print(f"delta={(hist_equiv - hist).to(int)}")
diff = torch.abs(hist_equiv - hist).sum()
print(f"sum of differences {diff} with {dtype=}.")

# %%
# That's not really working.
# Let's do another verification.
# We first start again by comparing the number of differences between
# histograms for the the whole tensor.

histc_value = torch.histc(tensor, hbins, hmin, hmax)
histc_equiv = torch_histc_equivalent(tensor, hbins, hmin, hmax, thresholds=thresholds)
diff = (histc_value - histc_equiv).abs()
print(f"with {tensor.shape[0]} elements, there {diff.sum()} differences.")

# %%
# We now take the elements with an even position.


histc_value = torch.histc(tensor[::2], hbins, hmin, hmax)
histc_equiv = torch_histc_equivalent(tensor[::2], hbins, hmin, hmax, thresholds=thresholds)
diff = (histc_value - histc_equiv).abs()
print(
    f"with {tensor[::2].shape[0]} elements at even position, there {diff.sum()} differences."
)


# %%
# We now take the elements with an odd position.

histc_value = torch.histc(tensor[1::2], hbins, hmin, hmax)
histc_equiv = torch_histc_equivalent(tensor[1::2], hbins, hmin, hmax, thresholds=thresholds)
diff = (histc_value - histc_equiv).abs()
print(
    f"with {tensor[1::2].shape[0]} elements at odd position, there {diff.sum()} differences."
)

# %%
# This does not add up. Let's proove now :func:`torch.histc` is really confusing.
# The following sum should be null but it is not.

diff = torch.histc(tensor, hbins, hmin, hmax) - (
    torch.histc(tensor[::2], hbins, hmin, hmax) + torch.histc(tensor[1::2], hbins, hmin, hmax)
)
print(f"torch.histc: {tensor.dtype=}, number of differences: {diff.abs().sum()}: {diff}")


# %%
# This does not add up. Our implementation is more reliable.

diff = torch_histc_equivalent(tensor, hbins, hmin, hmax) - (
    torch_histc_equivalent(tensor[::2], hbins, hmin, hmax)
    + torch_histc_equivalent(tensor[1::2], hbins, hmin, hmax)
)
print(
    f"torch_histc_equivalent: {tensor.dtype=}, "
    f"number of differences: {diff.abs().sum()}: {diff}"
)
