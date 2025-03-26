"""
.. _l-plot-tiny-llm-export:

Steel method forward to guess the dynamic shapes (with Tiny-LLM)
================================================================

Inputs are always dynamic with LLMs that is why dynamic shapes
needs to be specified when a LLM is exported with:func:`torch.export.export`.
Most of the examples on :epkg:`HuggingFace` use method
:meth:`transformers.GenerationMixin.generate` but we only want to
export the model and its method ``forward``.

That example shows to guess the inputs of this method even though the model
is executed through meth ``generate``.

We focus on the model
`Tiny-LLM <https://huggingface.co/arnir0/Tiny-LLM>`_.
To avoid downloading any weights, we write a function creating a
random model based on the same architecture.

Steel the forward method
++++++++++++++++++++++++

The first step is to guess the dummy inputs.
Let's use the true model for that.
We use the dummy example from the model page.
"""

import copy
import pprint
import torch
import transformers
from onnx_diagnostic.helpers import string_type
from onnx_diagnostic.torch_models.llms import get_tiny_llm


MODEL_NAME = "arnir0/Tiny-LLM"
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# %%
# We rewrite the forward method to print the cache dimension.


def _forward_(*args, _f=None, **kwargs):
    assert _f is not None
    if not torch.compiler.is_exporting():
        print("<-", string_type((args, kwargs), with_shape=True, with_min_max=True))
    res = _f(*args, **kwargs)
    if not torch.compiler.is_exporting():
        print("->", string_type((args, kwargs), with_shape=True, with_min_max=True))
    return res


keep_model_forward = model.forward
model.forward = lambda *args, _f=keep_model_forward, **kwargs: _forward_(
    *args, _f=_f, **kwargs
)

# %%
# Let's run the model.
prompt = "Continue: it rains..."
inputs = tokenizer.encode(prompt, return_tensors="pt")

outputs = model.generate(
    inputs, max_length=50, temperature=1, top_k=50, top_p=0.95, do_sample=True
)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("-- prompt", prompt)
print("-- answer", generated_text)

# %%
# Let's restore the forward as it was.
model.forward = keep_model_forward

# %%
# Untrained model
# +++++++++++++++
#
# This part can skipped if you are only interested in exporting
# the original model. It is useful to create a unit test to ensure
# a specific architecture can be exported despite the many changes
# brought to :epkg:`torch` or :epkg:`transformers`.
#
# Let's create an untrained model using the config file provided
# `config.json <https://huggingface.co/arnir0/Tiny-LLM/blob/main/config.json>`_
# to create an untrained model:
# :func:`onnx_diagnostic.torch_models.llms.get_tiny_llm`.
# Then let's use it.

experiment = get_tiny_llm()
untrained_model, inputs, dynamic_shapes = (
    experiment["model"],
    experiment["inputs"],
    experiment["dynamic_shapes"],
)

# %%
# Before we run it, we make a copy of the inputs as the cache
# get modified by the execution. Then it is no longer valid
# associated with the previous input_ids and mask.
cloned_inputs = copy.deepcopy(inputs)


# %% Let's run it.
print("input type before", string_type(inputs, with_shape=True))

expected_output = untrained_model(**inputs)

print("input type after-", string_type(inputs, with_shape=True))

# %%
# The outputs

print("result type", string_type(expected_output, with_shape=True))

# %%
# It works.
#
# ExportedProgram
# +++++++++++++++

try:
    ep = torch.export.export(
        untrained_model, (), kwargs=cloned_inputs, dynamic_shapes=dynamic_shapes
    )
    print("It worked:")
    print(ep)
except Exception as e:
    # To work, it needs at least PRs:
    # * https://github.com/huggingface/transformers/pull/36311
    # * https://github.com/huggingface/transformers/pull/36652
    print("It failed:", e)


# %%
# Back to the original model
# ++++++++++++++++++++++++++
#
# Let's use the same dummy inputs but we use the downloaded model.
# Dummy inputs and dynamic shapes are created by function
# :func:`onnx_diagnostic.torch_models.llms.get_tiny_llm`.

data = get_tiny_llm()
inputs, dynamic_shapes = data["inputs"], data["dynamic_shapes"]

# %%
# Let's print the inputs.

print(string_type(inputs, with_shape=True))

# %% Let's print the dynamic shapes
pprint.pprint(dynamic_shapes)

# %%
# And Let's finally export.

try:
    ep = torch.export.export(model, (), kwargs=cloned_inputs, dynamic_shapes=dynamic_shapes)
    print("It worked:")
    print(ep)
except Exception as e:
    # To work, it needs at least PRs:
    # * https://github.com/huggingface/transformers/pull/36311
    # * https://github.com/huggingface/transformers/pull/36652
    print("It failed:", e)

# %%
# If you have any error, then look at example
# :ref:`l-plot-tiny-llm-export-patched`.
