import os
import subprocess
import sys
import tempfile
from typing import List, Optional, Tuple, Union
import numpy as np
import onnx
from .helpers.dot_helper import to_dot


def get_latest_pypi_version(package_name="onnx-diagnostic") -> str:
    """Returns the latest published version."""

    import requests

    url = f"https://pypi.org/pypi/{package_name}/json"
    response = requests.get(url)

    assert response.status_code == 200, f"Unable to retrieve the version response={response}"
    data = response.json()
    version = data["info"]["version"]
    return version


def update_version_package(version: str, package_name="onnx-diagnostic") -> str:
    "Adds dev if the major version is different from the latest published one."
    released = get_latest_pypi_version(package_name)
    shorten_r = ".".join(released.split(".")[:2])
    shorten_v = ".".join(version.split(".")[:2])
    return version if shorten_r == shorten_v else f"{shorten_v}.dev"


def reset_torch_transformers(gallery_conf, fname):
    "Resets torch dynamo for :epkg:`sphinx-gallery`."
    import matplotlib.pyplot as plt
    import torch

    plt.style.use("ggplot")
    torch._dynamo.reset()


def plot_legend(
    text: str, text_bottom: str = "", color: str = "green", fontsize: int = 15
) -> "matplotlib.axes.Axes":  # noqa: F821
    """
    Plots a graph with only text (for :epkg:`sphinx-gallery`).

    :param text: legend
    :param text_bottom: text at the bottom
    :param color: color
    :param fontsize: font size
    :return: axis
    """
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


def save_fig(ax, name: str, **kwargs) -> "matplotlib.axis.Axis":  # noqa: F821
    """Applies ``tight_layout`` and saves the figures. Returns ax."""
    fig = ax.get_figure()
    fig.savefig(name, **kwargs)
    return ax


def title(ax: "plt.axes", title: str) -> "matplotlib.axis.Axis":  # noqa: F821
    "Adds a title to axes and returns them."
    ax.set_title(title)
    return ax


def plot_histogram(
    tensor: np.ndarray,
    ax: Optional["plt.axes"] = None,  # noqa: F821
    bins: int = 30,
    color: str = "orange",
    alpha: float = 0.7,
) -> "matplotlib.axis.Axis":  # noqa: F821
    "Computes the distribution for a tensor."
    if ax is None:
        import matplotlib.pyplot as plt

        ax = plt.gca()
        ax.cla()
    ax.hist(tensor, bins=30, color="orange", alpha=0.7)
    ax.set_yscale("log")
    return ax


def _find_in_PATH(prog: str) -> Optional[str]:
    """
    Looks into every path mentioned in ``%PATH%`` a specific file,
    it raises an exception if not found.

    :param prog: program to look for
    :return: path
    """
    sep = ";" if sys.platform.startswith("win") else ":"
    path = os.environ["PATH"]
    for p in path.split(sep):
        f = os.path.join(p, prog)
        if os.path.exists(f):
            return p
    return None


def _find_graphviz_dot(exc: bool = True) -> str:
    """
    Determines the path to graphviz (on Windows),
    the function tests the existence of versions 34 to 45
    assuming it was installed in a standard folder:
    ``C:\\Program Files\\MiKTeX 2.9\\miktex\\bin\\x64``.

    :param exc: raise exception of be silent
    :return: path to dot
    :raises FileNotFoundError: if graphviz not found
    """
    if sys.platform.startswith("win"):
        version = list(range(34, 60))
        version.extend([f"{v}.1" for v in version])
        for v in version:
            graphviz_dot = f"C:\\Program Files (x86)\\Graphviz2.{v}\\bin\\dot.exe"
            if os.path.exists(graphviz_dot):
                return graphviz_dot
        extra = ["build/update_modules/Graphviz/bin"]
        for ext in extra:
            graphviz_dot = os.path.join(ext, "dot.exe")
            if os.path.exists(graphviz_dot):
                return graphviz_dot
        p = _find_in_PATH("dot.exe")
        if p is None:
            if exc:
                raise FileNotFoundError(
                    f"Unable to find graphviz, look into paths such as {graphviz_dot}."
                )
            return None
        return os.path.join(p, "dot.exe")
    # linux
    return "dot"


def _run_subprocess(args: List[str], cwd: Optional[str] = None):
    assert not isinstance(args, str), "args should be a sequence of strings, not a string."

    p = subprocess.Popen(
        args,
        cwd=cwd,
        shell=False,
        env=os.environ,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    raise_exception = False
    output = ""
    while True:
        output = p.stdout.readline().decode(errors="ignore")
        if output == "" and p.poll() is not None:
            break
        if output:
            if (
                "fatal error" in output
                or "CMake Error" in output
                or "gmake: ***" in output
                or "): error C" in output
                or ": error: " in output
            ):
                raise_exception = True
    p.poll()
    error = p.stderr.readline().decode(errors="ignore")
    p.stdout.close()
    if error and raise_exception:
        raise RuntimeError(
            f"An error was found in the output. The build is stopped."
            f"\n{output}\n---\n{error}"
        )
    return output + "\n" + error


def _run_graphviz(filename: str, image: str, engine: str = "dot") -> str:
    """
    Run :epkg:`Graphviz`.

    :param filename: filename which contains the graph definition
    :param image: output image
    :param engine: *dot* or *neato*
    :return: output of graphviz
    """
    ext = os.path.splitext(image)[-1]
    assert ext in {
        ".png",
        ".bmp",
        ".fig",
        ".gif",
        ".ico",
        ".jpg",
        ".jpeg",
        ".pdf",
        ".ps",
        ".svg",
        ".vrml",
        ".tif",
        ".tiff",
        ".wbmp",
    }, f"Unexpected extension {ext!r} for {image!r}."
    if sys.platform.startswith("win"):
        bin_ = os.path.dirname(_find_graphviz_dot())
        # if bin not in os.environ["PATH"]:
        #    os.environ["PATH"] = os.environ["PATH"] + ";" + bin
        exe = os.path.join(bin_, engine)
    else:
        exe = engine
    if os.path.exists(image):
        os.remove(image)
    cmd = [exe, f"-T{ext[1:]}", filename, "-o", image]
    output = _run_subprocess(cmd)
    assert os.path.exists(image), (
        f"Unable to find {image!r}, command line is "
        f"{' '.join(cmd)!r}, Graphviz failed due to\n{output}"
    )
    return output


def draw_graph_graphviz(
    dot: Union[str, onnx.ModelProto], image: str, engine: str = "dot"
) -> str:
    """
    Draws a graph using :epkg:`Graphviz`.

    :param dot: dot graph or ModelProto
    :param image: output image, None, just returns the output
    :param engine: *dot* or *neato*
    :return: :epkg:`Graphviz` output or
        the dot text if *image* is None

    The function creates a temporary file to store the dot file if *image* is not None.
    """
    if isinstance(dot, onnx.ModelProto):
        sdot = to_dot(dot)
    else:
        if "{" not in dot:
            assert dot.endswith(".onnx"), f"Unexpected file extension for {dot!r}"
            proto = onnx.load(dot)
            sdot = to_dot(proto)
        else:
            sdot = dot
        assert "{" in sdot, f"This string is not a dot string\n{sdot}"
    with tempfile.NamedTemporaryFile(delete=False) as fp:
        fp.write(sdot.encode("utf-8"))
        fp.close()

        filename = fp.name
        assert os.path.exists(
            filename
        ), f"File {filename!r} cannot be created to store the graph."
        out = _run_graphviz(filename, image, engine=engine)
        assert os.path.exists(
            image
        ), f"Graphviz failed with no reason, {image!r} not found, output is {out}."
        os.remove(filename)
        return out


def plot_dot(
    dot: Union[str, onnx.ModelProto],
    ax: Optional["matplotlib.axis.Axis"] = None,  # noqa: F821
    engine: str = "dot",
    figsize: Optional[Tuple[int, int]] = None,
) -> "matplotlib.axis.Axis":  # noqa: F821
    """
    Draws a dot graph into a matplotlib graph.

    :param dot: dot graph or ModelProto
    :param image: output image, None, just returns the output
    :param engine: *dot* or *neato*
    :param figsize: figsize of ax is None
    :return: :epkg:`Graphviz` output or, the dot text if *image* is None

    .. plot::

        import matplotlib.pyplot as plt
        import onnx.parser
        from onnx_diagnostic.doc import plot_dot

        model = onnx.parser.parse_model(
            '''
            <ir_version: 8, opset_import: [ "": 18]>
            agraph (float[N] x) => (float[N] z) {
                two = Constant <value_float=2.0> ()
                four = Add(two, two)
                z = Mul(four, four)
            }
        ''')

        ax = plot_dot(model)
        ax.set_title("Dummy graph")
        plt.show()
    """
    if ax is None:
        import matplotlib.pyplot as plt

        _, ax = plt.subplots(1, 1, figsize=figsize)
        clean = True
    else:
        clean = False

    from PIL import Image

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fp:
        fp.close()

        draw_graph_graphviz(dot, fp.name, engine=engine)
        img = np.asarray(Image.open(fp.name))
        os.remove(fp.name)

        ax.imshow(img)

    if clean:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_axis_off()
        ax.get_figure().tight_layout()
    return ax
