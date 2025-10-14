"""
.. _l-plot-export-hub-codellama:

Test the export on untrained models
===================================

Checking the exporter on a whole model takes time as it is
usually big but we can create a smaller version with
the same architecture. Then fix export issues on such a
small model is faster.

codellama/CodeLlama-7b-Python-hf
++++++++++++++++++++++++++++++++

Let's grab some information about this model.
This reuses :epkg:`huggingface_hub` API.
"""

import copy
import pprint
import torch
from onnx_diagnostic import doc
from onnx_diagnostic.ext_test_case import unit_test_going
from onnx_diagnostic.helpers import string_type
from onnx_diagnostic.torch_models.hghub import (
    get_untrained_model_with_inputs,
)
from onnx_diagnostic.torch_models.hghub.hub_api import (
    get_model_info,
    get_pretrained_config,
    task_from_id,
)
from onnx_diagnostic.torch_export_patches import torch_export_patches
from onnx_diagnostic.torch_export_patches.patch_inputs import use_dyn_not_str

model_id = (
    "HuggingFaceM4/tiny-random-idefics"
    if unit_test_going()
    else "codellama/CodeLlama-7b-Python-hf"
)
print(f"model_id={model_id!r}")
print("info", get_model_info(model_id))

# %%
# The configuration.

print("config", get_pretrained_config(model_id))

# %%
# The task determines the set of inputs which needs
# to be created for this input.

print("task", task_from_id(model_id))

# %%
# Untrained model
# +++++++++++++++
#
# The function :func:`get_untrained_model_with_inputs
# <onnx_diagnostic.torch_models.hghub.get_untrained_model_with_inputs>`.
# It loads the pretrained configuration, extracts the task associated
# to the model and them creates random inputs and dynamic shapes
# for :func:`torch.export.export`.

data = get_untrained_model_with_inputs(model_id, verbose=1)
print("model size:", data["size"])
print("number of weights:", data["n_weights"])
print("fields:", set(data))

# %%
# Inputs
print("inputs:", string_type(data["inputs"], with_shape=True))

# %%
# Dynamic Shapes
print("dynamic shapes:", pprint.pformat(data["dynamic_shapes"]))

# %%
# Let's check the model runs. We still needs to
# copy the inputs before using the models, the cache
# is usually modified inplace.
# Expected outputs can be used later to compute
# discrepancies.

inputs_copy = copy.deepcopy(data["inputs"])
model = data["model"]
expected_outputs = model(**inputs_copy)

print("outputs:", string_type(expected_outputs, with_shape=True))

# %%
# It works.
#
# Export
# ++++++
#
# The model uses :class:`transformers.cache_utils.DynamicCache`.
# It still requires patches to be exportable (control flow).
# See :func:`onnx_diagnostic.torch_export_patches.torch_export_patches`

with torch_export_patches(patch_torch=False, patch_transformers=True) as f:
    ep = torch.export.export(
        model,
        (),
        kwargs=f(data["inputs"]),
        dynamic_shapes=use_dyn_not_str(data["dynamic_shapes"]),
        strict=False,
    )
    print(ep)


# %%

doc.plot_legend(
    "untrained\ncodellama/\nCodeLlama-7b-Python-hf", "torch.export.export", "tomato"
)
