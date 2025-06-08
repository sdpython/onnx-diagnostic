"""
.. _l-plot-parallelized-reduction:

Reproducible Parallelized Reduction is difficult
================================================

A reduction is a frequent operation with neural networks. It appears in layer normalization,
softmax... Because of the float precision, the result of the computation
changes based on the order of the elements. The following examples show the variation
based on different hypothesis on the vector distribution.
We consider a vector :math:`X = (x_1, ..., x_n)`.
It computes the average:

.. math::

    mean(X) = \\frac{\\sum_{i=1}^n x_i}{n}

Or the normalization of the vector:

.. math::

    norm(X)_i = \\frac{ X_i  - \\mathbb{E}X}{ \\sqrt{ \\mathbb{V}X}}

With :math:`\\mathbb{E}X = mean(X)`,
:math:`\\mathbb{V}X = mean\\left(\\left(X - mean(X)\\right)^2\\right)`.
We draw 128 random permutations of X. The average or mean should not change.
And the normalized vector should have the same values. In the first case, we compute
the difference between the highest and the lowest values obtained for the average.
In the second case, we look for the maximum difference between the original normalized
vector and the permuted one, both sorted.

The computation code
++++++++++++++++++++
"""

import itertools
from tqdm import tqdm
import numpy as np
import pandas

DATA = []


def str_dtype(dtype):
    """Displays numpy dtype in a nicer way."""
    if dtype == np.float64:
        return "fp64"
    if dtype == np.float32:
        return "fp32"
    if dtype == np.float16:
        return "fp16"
    raise ValueError(f"Unexpected value {dtype}")


def layer_norm(a, eps=1e-6):
    """
    Normalized the vector a.
    The computation is done in float32 or float64.
    """
    ctype = np.float32 if a.dtype == np.float16 else a.dtype
    a32 = a.astype(ctype)
    m = a32.mean(axis=-1, keepdims=True)
    c = a32 - m
    va = np.sqrt((c * c).mean(axis=-1, keepdims=True))
    va += eps
    return (c / va).astype(a.dtype)


def compute(values, fct):
    """
    Compare the results of function ``fct`` on a sample.
    Loops over multiple sizes, dtypes. Tries 128 times.
    """

    def make_value(base, value):
        if value.size > 1:
            return np.abs(np.sort(base) - np.sort(value)).max()
        return value

    sizes = [2, 4, 8, 16, 512, 1024, 2048, 4096, 8192]
    dtypes = [np.float64, np.float32, np.float16]
    N = list(range(128))
    exps = list(itertools.product(sizes, dtypes, N))
    data = []
    ech = None
    for size, dtype, n in tqdm(exps):
        if n == 0:
            ech = values[:size].astype(dtype)
            base = fct(ech)
            assert base.dtype == ech.dtype
            obs = dict(
                n=n, size=size, dtype=str_dtype(ech.dtype), value=make_value(base, fct(ech))
            )
            data.append(obs)

        if n == 1:
            new_ech = np.sort(ech)
        elif n == 2:
            new_ech = np.sort(ech)[::-1]
        else:
            new_ech = np.random.permutation(ech)
        assert new_ech.dtype == ech.dtype
        assert new_ech.shape == ech.shape
        obs = dict(
            n=n + 1,
            size=size,
            dtype=str_dtype(new_ech.dtype),
            value=make_value(base, fct(new_ech)),
        )
        data.append(obs)

    df = pandas.DataFrame(data)
    agg = df.drop("n", axis=1).groupby(["dtype", "size"], as_index=False).agg(["min", "max"])
    agg["value", "delta"] = agg["value", "max"] - agg["value", "min"]
    piv = agg.pivot(index="size", columns="dtype", values=("value", "delta"))
    return piv


# %%
# Normal Law
# ++++++++++
#
# Let's see what it returns an on random sample following a normal law.
# First the average.

values = np.random.randn(4096)
mean = compute(values, lambda x: np.mean(x).astype(x.dtype))
mean["name"] = "normal"
print(mean)

# %%
# Then the layer normalization.

ln = compute(values, layer_norm)
ln["name"] = "normal"
DATA.append(ln.reset_index(drop=True).max(axis=0))
print(ln)

# %%
# Fixed values
# ++++++++++++
#
# We try a fixed vector with one very high value and all the others are small.

values[:] = -1e-4
values[::128] = 100
mean = compute(values, lambda x: np.mean(x).astype(x.dtype))
mean["name"] = "fixed"
print(mean)

# %%
# And the normalized vector.
ln = compute(values, layer_norm)
ln["name"] = "fixed"
DATA.append(ln.reset_index(drop=True).max(axis=0))
print(ln)

# %%
# Pareto Distribution
# +++++++++++++++++++
#
# A law with a long tail.

values = np.random.pareto(1, (4096,))
print(values)

mean = compute(values, lambda x: np.mean(x).astype(x.dtype))
mean["name"] = "normal"
print(mean)

# %%
# And the normalized vector.
ln = compute(values, layer_norm)
ln["name"] = "pareto"
DATA.append(ln.reset_index(drop=True).max(axis=0))
print(ln)

# %%
# Summary
# +++++++
#
# We consider the maximum difference obtained for any sample size.

df = pandas.DataFrame(DATA).set_index("name")
print(df)

# %%
# Visually.

ax = df.plot.bar(logy=True)
fig = ax.get_figure()
fig.savefig("plot_parallelized_reduction.png")

# %%
# In a deep neural network
# ++++++++++++++++++++++++
#
# Some of the vector have 500 values, 16x32x1024x1024. A layer normalization
# does 16x32x1024 ~ 2M reductions, over 20 layers.
# When a deep neural network is computed with a different code
# doing a different parallelization (GPU/CPU for example),
# the order of the reduction may change and therefore,
# some errors will appear and propagate.
