from typing import Optional
import numpy as np


def reset_torch_transformers(gallery_conf, fname):
    "Resets torch dynamo for :epkg:`sphinx-gallery`."
    import matplotlib.pyplot as plt
    import torch

    plt.style.use("ggplot")
    torch._dynamo.reset()


def plot_legend(
    text: str, text_bottom: str = "", color: str = "green", fontsize: int = 15
) -> "matplotlib.axes.Axes":  # noqa: F821
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(2, 2))
    ax = fig.add_subplot()
    ax.axis([0, 5, 0, 5])
    ax.text(2.5, 4, "END", fontsize=10, horizontalalignment="center")
    ax.text(
        2.5,
        2.5,
        text,
        fontsize=fontsize,
        bbox={"facecolor": color, "alpha": 0.5, "pad": 10},
        horizontalalignment="center",
        verticalalignment="center",
    )
    if text_bottom:
        ax.text(4.5, 0.5, text_bottom, fontsize=7, horizontalalignment="right")
    ax.grid(False)
    ax.set_axis_off()
    return ax


def rotate_align(ax, angle=15, align="right"):
    """Rotates x-label and align them to thr right. Returns ax."""
    for label in ax.get_xticklabels():
        label.set_rotation(angle)
        label.set_horizontalalignment(align)
    return ax


def save_fig(ax, name: str):
    """Applies ``tight_layout`` and saves the figures. Returns ax."""
    import matplotlib.pyplot as plt

    plt.tight_layout()
    fig = ax.get_figure()
    fig.savefig(name)
    return ax


def title(ax: "plt.axes", title: str) -> "plt.axes":  # noqa: F821
    "Adds a title to axes and returns them."
    ax.set_title(title)
    return ax


def plot_histogram(
    tensor: np.ndarray,
    ax: Optional["plt.axes"] = None,  # noqa: F821
    bins: int = 30,
    color: str = "orange",
    alpha: float = 0.7,
) -> "plt.axes":  # noqa: F821
    "Computes the distribution for a tensor."
    if ax is None:
        import matplotlib.pyplot as plt

        ax = plt.gca()
        ax.cla()
    ax.hist(tensor, bins=30, color="orange", alpha=0.7)
    ax.set_yscale("log")
    return ax
