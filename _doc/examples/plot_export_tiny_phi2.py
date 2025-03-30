"""
.. _l-plot-export_tiny_phi2:

Untrained microsoft/phi-2
=========================

:epkg:`microsoft/phi-2` is not a big models but still quite big
when it comes to write unittest. Function
:func:`onnx_diagnostic.torch_models.hghub.get_untrained_model_with_inputs`
can be used to create a reduced untrained version of a model coming from
:epkg:`HuggingFace`. It downloads the configuration from the website
but creates a dummy model with 1 or 2 hidden layers in order to reduce
the size and get a fast execution. The goal is usually to test
the export or to compare performance. The relevance does not matter.

Create the dummy model
++++++++++++++++++++++
"""

import copy
import pprint
import warnings
import torch
import onnxruntime
from onnx_diagnostic import doc
from onnx_diagnostic.helpers import max_diff, string_diff, string_type
from onnx_diagnostic.helpers.cache_helper import is_cache_dynamic_registered
from onnx_diagnostic.helpers.ort_session import make_feeds
from onnx_diagnostic.torch_export_patches import bypass_export_some_errors
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

print(f"model {size / 2**10:1.3f} Kb with {n_weights} parameters.")
# %%
# The original model has 2.7 billion parameters. It was divided by more than 10.
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
# Export
# ++++++


with bypass_export_some_errors(patch_transformers=True) as modificator:

    # Unnecessary steps but useful in case of an error
    # We check the cache is registered.
    assert is_cache_dynamic_registered()

    # We check there is no discrepancies when the cache is applied.
    d = max_diff(expected, untrained_model(**copy.deepcopy(inputs)))
    assert (
        d["abs"] < 1e-5
    ), f"The model with patches produces different outputs: {string_diff(d)}"

    # Then we export.
    ep = torch.export.export(
        untrained_model,
        (),
        kwargs=modificator(copy.deepcopy(inputs)),
        dynamic_shapes=dynamic_shapes,
        strict=False,  # mandatory for torch==2.6
    )

    # We check the exported program produces the same results as well.
    d = max_diff(expected, ep.module()(**copy.deepcopy(inputs)))
    assert d["abs"] < 1e-5, f"The exported model different outputs: {string_diff(d)}"

# %%
# Export to ONNX
# ++++++++++++++
#
# The export works. We can export to ONNX now.
# Patches are still needed because the export
# applies :meth:`torch.export.ExportedProgram.run_decompositions`
# may export local pieces of the model again.

with bypass_export_some_errors(patch_transformers=True):
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

feeds = make_feeds(onx, copy.deepcopy(inputs), use_numpy=True)

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
doc.plot_legend("untrained smaller\nmicrosoft/phi-2", "torch.onnx.export", "green")
