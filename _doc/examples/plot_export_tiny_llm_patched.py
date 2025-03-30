"""
.. _l-plot-tiny-llm-export-patched:

Export Tiny-LLM with patches
============================

Many models from :epkg:`transformers` cannot be converted because
the implementation uses cache classes. Let's see how to get around that.
We focus on the model :epkg:`arnir0/Tiny-LLM`.
To avoid downloading any weights, we write a function creating a
random model based on the same architecture.
This continues example :ref:`l-plot-tiny-llm-export`.

Errors
++++++

They depend on transformers version.

``transformers>=4.40,<4.50`` cannot serialize DynamicCache and cannot
map dynamic shapes to instances of DynamicCache. The following errors
would appear:

::

  torch._dynamo.exc.UserError: Cannot associate shape
      [[{0: <class '....batch'>, 2: <class '....cache_length'>}],
       [{0: <class '....batch'>, 2: <class '....cache_length'>}]]
      specified at `dynamic_shapes['past_key_values']`
      to non-tensor type <class 'transformers.cache_utils.DynamicCache'>
      at `inputs['past_key_values']` (expected None)
  For more information about this error,
  see: https://pytorch.org/docs/main/generated/exportdb/index.html#dynamic-shapes-validation

With ``transformers==4.50``, it shows the following:

::

  torch._dynamo.exc.UserError: Constraints violated (batch)!
  For more information, run with TORCH_LOGS="+dynamic".
      - Not all values of batch = L['args'][1]['input_ids'].size()[0]
          in the specified range batch <= 1024 are valid
          because batch was inferred to be a constant (2).
      - Not all values of batch = L['args'][1]['attention_mask'].size()[0]
          in the specified range batch <= 1024 are valid
          because batch was inferred to be a constant (2).
      - Not all values of batch = L['args'][1]['past_key_values']['key_cache'][0].size()[0]
          in the specified range batch <= 1024 are valid
          because batch was inferred to be a constant (2).
      - Not all values of batch = L['args'][1]['past_key_values']['value_cache'][0].size()[0]
          in the specified range batch <= 1024 are valid
          because batch was inferred to be a constant (2).
   Suggested fixes:
       batch = 2

However, this package implements a patch mechanism
with replaces the part causing these issues.

.. note:: restart after an export failure

    If the export fails, it is better to start executing again,
    or restart the kernel if you are in the notebook.
    The export may leave :epkg:`torch` in one unstable state.
"""

import copy
import pprint
import torch
import transformers
from onnx_diagnostic import doc
from onnx_diagnostic.helpers.cache_helper import is_cache_dynamic_registered
from onnx_diagnostic.helpers import string_type
from onnx_diagnostic.torch_export_patches import bypass_export_some_errors
from onnx_diagnostic.torch_models.llms import get_tiny_llm


experiment = get_tiny_llm()
untrained_model, inputs, dynamic_shapes = (
    experiment["model"],
    experiment["inputs"],
    experiment["dynamic_shapes"],
)

cloned_inputs = copy.deepcopy(inputs)

# %%
# Let's show this inputs, this was inferred in
# example :ref:`l-plot-tiny-llm-export`.

print(string_type(inputs, with_shape=True))

# %%
# And the dynamic shapes
pprint.pprint(dynamic_shapes)

# %%
# Before exporting, we check :class:`transformers.cache_utils.DynamicCache`
# can serialized and deserialized otherwise :func:`torch.export.export`
# fails.

print("-- DynamicCache registered: ", is_cache_dynamic_registered())

# %%
# If they are not registered, function
# func:`onnx_diagnostic.torch_export_patches.bypass_export_some_errors`
# should take care of it. Then we export.

with bypass_export_some_errors(patch_transformers=True, verbose=10) as modificator:
    assert is_cache_dynamic_registered()  # it must be true here
    ep = torch.export.export(
        untrained_model,
        (),
        kwargs=modificator(cloned_inputs),
        dynamic_shapes=dynamic_shapes,
        strict=False,  # mandatory for torch==2.6
    )
    print("It worked:")
    print(ep)

# %%
# With the original model
# +++++++++++++++++++++++

MODEL_NAME = "arnir0/Tiny-LLM"
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_NAME)

cloned_inputs = copy.deepcopy(inputs)

with bypass_export_some_errors(patch_transformers=True, verbose=10) as modificator:
    ep = torch.export.export(
        model,
        (),
        kwargs=modificator(cloned_inputs),
        dynamic_shapes=dynamic_shapes,
        strict=False,  # mandatory for torch==2.6
    )
    print("It worked:")
    print(ep)

# %%
doc.plot_legend("Tiny-LLM patched", "torch.export.export", "green")
