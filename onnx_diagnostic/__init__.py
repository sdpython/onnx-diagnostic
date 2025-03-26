"""
Investigates onnx models.
Functions, classes to dig into a model when this one is right, slow, wrong...
"""

__version__ = "0.3.0"
__author__ = "Xavier Dupré"


def reset_torch_transformers(gallery_conf, fname):
    "Resets torch dynamo for :epkg:`sphinx-gallery`."
    import matplotlib.pyplot as plt
    import torch

    plt.style.use("ggplot")
    torch._dynamo.reset()
