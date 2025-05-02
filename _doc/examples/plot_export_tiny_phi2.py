"""
.. _l-plot-export_tiny_phi2:

======================
Export microsoft/phi-2
======================

This function exports an smaller untrained model with the same architecture.
It is faster than the pretrained model.
When this works, the untrained model can be replaced by the trained one.

:epkg:`microsoft/phi-2` is not a big model but still quite big
when it comes to write unittests. Function
:func:`onnx_diagnostic.torch_models.hghub.get_untrained_model_with_inputs`
can be used to create a reduced untrained version of a model coming from
:epkg:`HuggingFace`. It downloads the configuration from the website
but creates a dummy model with 1 or 2 hidden layers in order to reduce
the size and get a fast execution. The goal is usually to test
the export or to compare performance. The relevance does not matter.

Create the dummy model
======================
"""

import copy
import pprint
import warnings
import torch
import onnxruntime
from onnx_diagnostic import doc
from onnx_diagnostic.helpers import max_diff, string_diff, string_type
from onnx_diagnostic.helpers.cache_helper import is_cache_dynamic_registered
from onnx_diagnostic.helpers.rt_helper import make_feeds
from onnx_diagnostic.torch_export_patches import torch_export_patches
from onnx_diagnostic.torch_export_patches.patch_inputs import use_dyn_not_str
from onnx_diagnostic.torch_models.hghub import (
    get_untrained_model_with_inputs,
)

warnings.simplefilter("ignore")

# another tiny id: arnir0/Tiny-LLM
data = get_untrained_model_with_inputs("microsoft/phi-2")
untrained_model, inputs, dynamic_shapes, config, size, n_weights = (
    data["model"],
    data["inputs"],
    data["dynamic_shapes"],
    data["configuration"],
    data["size"],
    data["n_weights"],
)

print(f"model {size / 2**20:1.1f} Mb with {n_weights // 1000} thousands of parameters.")
# %%
# The original model has 2.7 billion parameters. It was divided by more than 10.
# However, it can still be used with
# ``get_untrained_model_with_inputs("microsoft/phi-2", same_as_pretrained=True)``.
# Let's see the configuration.
print(config)


# %%
# Inputs:

print(string_type(inputs, with_shape=True))

# %%
# With min/max values.
print(string_type(inputs, with_shape=True, with_min_max=True))

# %%
# And the dynamic shapes
pprint.pprint(dynamic_shapes)

# %%
# We execute the model to produce expected outputs.
expected = untrained_model(**copy.deepcopy(inputs))
print(f"expected: {string_type(expected, with_shape=True, with_min_max=True)}")


# %%
# Export to fx.Graph
# ==================
#
# :func:`torch.export.export` is the first step before converting
# a model into ONNX. The inputs are duplicated (with ``copy.deepcopy``)
# because the model may modify them inline (a cache for example).
# Shapes may not match on the second call with the modified inputs.


with torch_export_patches(patch_transformers=True):

    # Two unnecessary steps but useful in case of an error
    # We check the cache is registered.
    assert is_cache_dynamic_registered()

    # We check there is no discrepancies when the cache is applied.
    d = max_diff(expected, untrained_model(**copy.deepcopy(inputs)))
    assert (
        d["abs"] < 1e-5
    ), f"The model with patches produces different outputs: {string_diff(d)}"

    # Then we export: the only import line in this section.
    ep = torch.export.export(
        untrained_model,
        (),
        kwargs=copy.deepcopy(inputs),
        dynamic_shapes=use_dyn_not_str(dynamic_shapes),
        strict=False,  # mandatory for torch==2.6
    )

    # We check the exported program produces the same results as well.
    # This step is again unnecessary.
    d = max_diff(expected, ep.module()(**copy.deepcopy(inputs)))
    assert d["abs"] < 1e-5, f"The exported model different outputs: {string_diff(d)}"

# %%
# Export to ONNX
# ==============
#
# The export works. We can export to ONNX now
# :func:`torch.onnx.export`.
# Patches are still needed because the export
# applies :meth:`torch.export.ExportedProgram.run_decompositions`
# may export local pieces of the model again.

with torch_export_patches(patch_transformers=True):
    epo = torch.onnx.export(
        ep, (), kwargs=copy.deepcopy(inputs), dynamic_shapes=dynamic_shapes, dynamo=True
    )

# %%
# We can save it.
epo.save("plot_export_tiny_phi2.onnx", external_data=True)

# Or directly get the :class:`onnx.ModelProto`.
onx = epo.model_proto


# %%
# Discrepancies
# +++++++++++++
#
# The we check the conversion to ONNX.
# Let's make sure the ONNX model produces the same outputs.
# It takes flatten inputs.

feeds = make_feeds(onx, copy.deepcopy(inputs), use_numpy=True, copy=True)

print(f"torch inputs: {string_type(inputs)}")
print(f"onxrt inputs: {string_type(feeds)}")

# %%
# We then create a :class:`onnxruntime.InferenceSession`.

sess = onnxruntime.InferenceSession(
    onx.SerializeToString(), providers=["CPUExecutionProvider"]
)

# %%
# Let's run.
got = sess.run(None, feeds)

# %%
# And finally the discrepancies.

diff = max_diff(expected, got, flatten=True)
print(f"onnx discrepancies: {string_diff(diff)}")

# %%
# It looks good.

# %%
doc.plot_legend("export\nuntrained smaller\nmicrosoft/phi-2", "torch.onnx.export", "orange")

# %%
# Possible Issues
# ===============
#
# Unknown task
# ++++++++++++
#
# Function :func:`onnx_diagnostic.torch_models.hghub.get_untrained_model_with_inputs`
# is unabl to guess a task associated to the model.
# A different set of dummy inputs is defined for every task.
# The user needs to explicitly give that information to the function.
# Tasks are the same as the one defined by
# `HuggingFace/models <https://huggingface.co/models>`_.
#
# Inputs are incorrect
# ++++++++++++++++++++
#
# Example :ref:`l-plot-tiny-llm-export` explains
# how to retrieve that information. If you cannot guess the dynamic
# shapes - a cache can be tricky sometimes, follow example
# :ref:`l-plot-export-with-args-kwargs`.
#
# DynamicCache or any other cache cannot be exported
# ++++++++++++++++++++++++++++++++++++++++++++++++++
#
# That's the role of :func:`onnx_diagnostic.torch_export_patches.torch_export_patches`.
# It registers the necessary information into pytorch to make the export
# work with these. Its need should slowly disappear until :epkg:`transformers`
# includes the serialization functions.
#
# Control Flow
# ++++++++++++
#
# Every mixture of models goes through a control flow (a test).
# It also happens when a cache is truncated. The code of the model
# needs to be changed. See example :ref:`l-plot-export-cond`.
# Loops are not supported yet.
#
# Issue with dynamic shapes
# +++++++++++++++++++++++++
#
# Example :ref:`l-plot-dynamic-shapes-python-int` gives one reason
# this process may fail but that's not the only one.
# Example :ref:`l-plot-export-locale-issue` gives an way to locate
# the cause but that does not cover all the possible causes.
# Raising an issue on github would be the recommended option
# until it is fixed.
