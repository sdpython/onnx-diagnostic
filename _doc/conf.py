import os
import sys
import packaging.version as pv
from sphinx_runpython.github_link import make_linkcode_resolve
from sphinx_runpython.conf_helper import has_dvipng, has_dvisvgm
import torch
from onnx_diagnostic import __version__
from onnx_diagnostic.doc import update_version_package

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.githubpages",
    "sphinx.ext.ifconfig",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
    "sphinx_gallery.gen_gallery",
    "sphinx_issues",
    "matplotlib.sphinxext.plot_directive",
    "sphinx_runpython.epkg",
    "sphinx_runpython.gdot",
    "sphinx_runpython.runpython",
]

if has_dvisvgm():
    extensions.append("sphinx.ext.imgmath")
    imgmath_image_format = "svg"
elif has_dvipng():
    extensions.append("sphinx.ext.pngmath")
    imgmath_image_format = "png"
else:
    extensions.append("sphinx.ext.mathjax")

templates_path = ["_templates"]
html_logo = "_static/logo.png"
source_suffix = ".rst"
master_doc = "index"
project = "onnx-diagnostic"
copyright = "2025"
author = "Xavier Dupré"
version = update_version_package(__version__)
release = version
language = "en"
exclude_patterns = []
pygments_style = "sphinx"
todo_include_todos = True

html_theme = "furo"
html_theme_path = ["_static"]
html_theme_options = {}
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_sourcelink_suffix = ""

issues_github_path = "sdpython/onnx-diagnostic"

# suppress_warnings = [
#     "tutorial.exported_onnx",
#     "tutorial.exported_onnx_dynamic",
#     "tutorial.exported_program",
#     "tutorial.exported_program_dynamic",
# ]

# The following is used by sphinx.ext.linkcode to provide links to github
_linkcode_resolve = make_linkcode_resolve(
    "onnx_diagnostic",
    (
        "https://github.com/sdpython/onnx-diagnostic/"
        "blob/{revision}/{package}/"
        "{path}#L{lineno}"
    ),
)


def linkcode_resolve(domain, info):
    return _linkcode_resolve(domain, info)


latex_elements = {
    "papersize": "a4",
    "pointsize": "10pt",
    "title": project,
}

intersphinx_mapping = {
    "_".join(["experimental", "experiment"]): (
        "https://sdpython.github.io/doc/experimental-experiment/dev/",
        None,
    ),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "onnx": ("https://onnx.ai/onnx/", None),
    "onnxruntime": ("https://onnxruntime.ai/docs/api/python/", None),
    "onnxscript": ("https://microsoft.github.io/onnxscript/", None),
    "onnx_array_api": ("https://sdpython.github.io/doc/onnx-array-api/dev/", None),
    "onnx_diagnostic": ("https://sdpython.github.io/doc/onnx-diagnostic/dev/", None),
    "onnx_extended": ("https://sdpython.github.io/doc/onnx-extended/dev/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "python": (f"https://docs.python.org/{sys.version_info.major}", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "skl2onnx": ("https://onnx.ai/sklearn-onnx/", None),
    "torch": ("https://pytorch.org/docs/main/", None),
}

# Check intersphinx reference targets exist
nitpicky = True
# See also scikit-learn/scikit-learn#26761
nitpick_ignore = [
    ("py:class", "ast.Node"),
    ("py:class", "dtype"),
    ("py:class", "False"),
    ("py:class", "True"),
    ("py:class", "Argument"),
    ("py:class", "default=sklearn.utils.metadata_routing.UNCHANGED"),
    ("py:class", "ModelProto"),
    ("py:class", "Model"),
    ("py:class", "Module"),
    ("py:class", "np.ndarray"),
    ("py:class", "onnx_ir.Tuple"),
    ("py:class", "pandas.core.groupby.generic.DataFrameGroupBy"),
    ("py:class", "pipeline.Pipeline"),
    ("py:class", "torch.fx.passes.operator_support.OperatorSupport"),
    ("py:class", "torch.fx.proxy.TracerBase"),
    ("py:class", "torch.FloatTensor"),
    ("py:class", "torch.LongTensor"),
    ("py:class", "torch.utils._pytree.Context"),
    ("py:class", "torch.utils._pytree.KeyEntry"),
    ("py:class", "torch.utils._pytree.TreeSpec"),
    ("py:class", "transformers.BartForConditionalGeneration"),
    ("py:class", "transformers.LlamaConfig"),
    ("py:class", "transformers.cache_utils.Cache"),
    ("py:class", "transformers.cache_utils.DynamicCache"),
    ("py:class", "transformers.cache_utils.EncoderDecoderCache"),
    ("py:class", "transformers.cache_utils.MambaCache"),
    ("py:class", "transformers.cache_utils.SlidingWindowCache"),
    ("py:class", "transformers.cache_utils.StaticCache"),
    ("py:class", "transformers.configuration_utils.PretrainedConfig"),
    ("py:class", "transformers.modeling_outputs.BaseModelOutput"),
    ("py:class", "transformers.models.phi3.modeling_phi3.Phi3RotaryEmbedding"),
    ("py:func", "torch.export._draft_export.draft_export"),
    ("py:func", "torch._export.tools.report_exportability"),
    ("py:func", "torch.utils._pytree.register_pytree_node"),
    ("py:meth", "huggingface_hub.HfApi.list_models"),
    ("py:meth", "transformers.AutoConfig.from_pretrained"),
    ("py:meth", "transformers.GenerationMixin.generate"),
    ("py:meth", "transformers.models.bart.modeling_bart.BartEncoderLayer.forward"),
    ("py:meth", "unittests.TestCase.subTest"),
]

nitpick_ignore_regex = [
    ("py:func", ".*numpy[.].*"),
    ("py:func", ".*scipy[.].*"),
    # ("py:func", ".*torch.ops.higher_order.*"),
    ("py:class", ".*numpy._typing[.].*"),
    ("py:class", ".*onnxruntime[.].*"),
    ("py:meth", ".*onnxruntime[.].*"),
]


sphinx_gallery_conf = {
    # path to your examples scripts
    "examples_dirs": [
        os.path.join(os.path.dirname(__file__), "examples"),
        os.path.join(os.path.dirname(__file__), "recipes"),
        os.path.join(os.path.dirname(__file__), "technical"),
    ],
    # path where to save gallery generated examples
    "gallery_dirs": [
        "auto_examples",
        "auto_recipes",
        "auto_technical",
    ],
    # no parallelization to avoid conflict with environment variables
    "parallel": 1,
    # sorting
    "within_subsection_order": "ExampleTitleSortKey",
    # errors
    "abort_on_example_error": True,
    # recommendation
    "recommender": {"enable": True, "n_examples": 3, "min_df": 3, "max_df": 0.9},
    # ignore capture for matplotib axes
    "ignore_repr_types": "matplotlib\\.(text|axes)",
    # robubstness
    "reset_modules_order": "both",
    "reset_modules": ("matplotlib", "onnx_diagnostic.doc.reset_torch_transformers"),
}

if int(os.environ.get("UNITTEST_GOING", "0")):
    sphinx_gallery_conf["ignore_pattern"] = (
        ".*((tiny_llm)|(dort)|(draft_mode)|(hub_codellama.py)).*"
    )
elif pv.Version(torch.__version__) < pv.Version("2.8"):
    sphinx_gallery_conf["ignore_pattern"] = ".*((_oe_)|(dort)|(draft_mode)).*"


epkg_dictionary = {
    "aten functions": "https://pytorch.org/cppdocs/api/namespace_at.html#functions",
    "azure pipeline": "https://azure.microsoft.com/en-us/products/devops/pipelines",
    "Custom Backends": "https://pytorch.org/docs/stable/torch.compiler_custom_backends.html",
    "diffusers": "https://github.com/huggingface/diffusers",
    "DOT": "https://graphviz.org/doc/info/lang.html",
    "executorch": "https://pytorch.org/executorch/stable/intro-overview.html",
    "ExecuTorch": "https://pytorch.org/executorch/stable/intro-overview.html",
    "ExecuTorch Runtime Python API Reference": "https://pytorch.org/executorch/stable/runtime-python-api-reference.html",
    "ExecuTorch Tutorial": "https://pytorch.org/executorch/stable/tutorials/export-to-executorch-tutorial.html",
    "experimental-experiment": "https://sdpython.github.io/doc/experimental-experiment/dev/",
    "JIT": "https://en.wikipedia.org/wiki/Just-in-time_compilation",
    "FunctionProto": "https://onnx.ai/onnx/api/classes.html#functionproto",
    "graph break": "https://pytorch.org/docs/stable/torch.compiler_faq.html#graph-breaks",
    "GraphModule": "https://pytorch.org/docs/stable/fx.html#torch.fx.GraphModule",
    "HuggingFace": "https://huggingface.co/docs/hub/en/index",
    "huggingface_hub": "https://github.com/huggingface/huggingface_hub",
    "Linux": "https://www.linux.org/",
    "ml_dtypes": "https://github.com/jax-ml/ml_dtypes",
    "ModelBuilder": "https://onnxruntime.ai/docs/genai/howto/build-model.html",
    "monai": "https://monai.io/",
    "numpy": "https://numpy.org/",
    "onnx": "https://onnx.ai/onnx/",
    "onnx-ir": "https://github.com/onnx/ir-py",
    "onnx.helper": "https://onnx.ai/onnx/api/helper.html",
    "ONNX": "https://onnx.ai/",
    "ONNX Operators": "https://onnx.ai/onnx/operators/",
    "onnxrt backend": "https://pytorch.org/docs/stable/onnx_dynamo_onnxruntime_backend.html",
    "onnxruntime": "https://onnxruntime.ai/",
    "onnxruntime-training": "https://onnxruntime.ai/docs/get-started/training-on-device.html",
    "onnxruntime kernels": "https://onnxruntime.ai/docs/reference/operators/OperatorKernels.html",
    "onnx-array-api": "https://sdpython.github.io/doc/onnx-array-api/dev/",
    "onnx-diagnostic": "https://sdpython.github.io/doc/onnx-diagnostic/dev/",
    "onnx-extended": "https://sdpython.github.io/doc/onnx-extended/dev/",
    "onnx-script": "https://github.com/microsoft/onnxscript",
    "onnxscript": "https://github.com/microsoft/onnxscript",
    "onnxscript Tutorial": "https://microsoft.github.io/onnxscript/tutorial/index.html",
    "optree": "https://github.com/metaopt/optree",
    "Pattern-based Rewrite Using Rules With onnxscript": "https://microsoft.github.io/onnxscript/tutorial/rewriter/rewrite_patterns.html",
    "opsets": "https://onnx.ai/onnx/intro/concepts.html#what-is-an-opset-version",
    "pyinstrument": "https://pyinstrument.readthedocs.io/en/latest/",
    "psutil": "https://psutil.readthedocs.io/en/latest/",
    "python": "https://www.python.org/",
    "pytorch": "https://pytorch.org/",
    "run_with_ortvaluevector": "https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/onnxruntime_inference_collection.py#L339",
    "scikit-learn": "https://scikit-learn.org/stable/",
    "scipy": "https://scipy.org/",
    "sklearn-onnx": "https://onnx.ai/sklearn-onnx/",
    "sphinx-gallery": "https://github.com/sphinx-gallery/sphinx-gallery",
    "Supported Operators and Data Types": "https://github.com/microsoft/onnxruntime/blob/main/docs/OperatorKernels.md",
    "sympy": "https://www.sympy.org/en/index.html",
    "timm": "https://github.com/huggingface/pytorch-image-models",
    "torch": "https://pytorch.org/docs/stable/torch.html",
    "torchbench": "https://github.com/pytorch/benchmark",
    "torch.compile": "https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html",
    "torch.compiler": "https://pytorch.org/docs/stable/torch.compiler.html",
    "torch.export.export": "https://pytorch.org/docs/stable/export.html#torch.export.export",
    "torch.onnx": "https://pytorch.org/docs/stable/onnx.html",
    "transformers": "https://huggingface.co/docs/transformers/en/index",
    "vocos": "https://github.com/gemelo-ai/vocos",
    "Windows": "https://www.microsoft.com/windows",
}

# models
epkg_dictionary.update(
    {
        "arnir0/Tiny-LLM": "https://huggingface.co/arnir0/Tiny-LLM",
        "microsoft/phi-2": "https://huggingface.co/microsoft/phi-2",
        "microsoft/Phi-3.5-mini-instruct": "https://huggingface.co/microsoft/Phi-3.5-mini-instruct",
        "microsoft/Phi-3.5-vision-instruct": "https://huggingface.co/microsoft/Phi-3.5-vision-instruct",
    }
)
