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
