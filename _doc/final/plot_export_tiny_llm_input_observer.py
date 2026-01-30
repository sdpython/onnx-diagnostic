"""
.. _l-plot-tiny-llm-export-input-observer:

Export a LLM with InputObserver (with Tiny-LLM)
===============================================

The main issue when exporting a LLM is the example on HuggingFace is
based on method generate but we only need to export the forward method.
Example :ref:`l-plot-tiny-llm-export` gives details on how to guess
dummy inputs and dynamic shapes to do so.
Let's see how to simplify that.

Dummy Example
+++++++++++++

Let's use the example provided on
`arnir0/Tiny-LLM <https://huggingface.co/arnir0/Tiny-LLM>`_.
"""

import pandas
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from onnx_diagnostic import doc
from onnx_diagnostic.helpers import string_type
from onnx_diagnostic.torch_export_patches import (
    register_additional_serialization_functions,
    torch_export_patches,
)
from onnx_diagnostic.export.api import to_onnx
from onnx_diagnostic.investigate.input_observer import InputObserver

MODEL_NAME = "arnir0/Tiny-LLM"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)


def generate_text(
    prompt, model, tokenizer, max_length=50, temperature=1, top_k=50, top_p=0.95
):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=True,
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


# Define your prompt
prompt = "Continue: it rains, what should I do?"
generated_text = generate_text(prompt, model, tokenizer)
print("-----------------")
print(generated_text)
print("-----------------")

# %%
# Replace forward method
# ++++++++++++++++++++++
#
# We first cpature inputs and outputs with an :class`InputObserver
# <onnx_diagnostic.investigate.input_observer>`.
# We also need to registers additional patchs for :epkg:`transformers`.
# Then :epkg:`pytorch` knows how to flatten/unflatten inputs.


observer = InputObserver()
with register_additional_serialization_functions(patch_transformers=True), observer(model):
    generate_text(prompt, model, tokenizer)

print(f"number of stored inputs: {len(observer.info)}")

# %%
# Exports
# +++++++
#
# The `InputObserver` has now enough data to infer arguments and dynamic shapes.
# We need more than serialization but also patches to export the model.
# Inferred dynamic shapes looks like:
print(observer.infer_dynamic_shapes(set_batch_dimension_for=True))

# %%
# and inferred arguments:
print(string_type(observer.infer_arguments(), with_shape=True))

# %%
# Let's export.

filename = "plot_export_tiny_llm_input_observer.custom.onnx"
with torch_export_patches(patch_transformers=True):
    to_onnx(
        model,
        (),
        kwargs=observer.infer_arguments(),
        dynamic_shapes=observer.infer_dynamic_shapes(set_batch_dimension_for=True),
        filename=filename,
        exporter="custom",
    )

# %%
# Check disrepancies
# ++++++++++++++++++
#
# The model is exported into ONNX. We use again the stored inputs and outputs
# to verify the model prooduces the same outputs.

data = observer.check_discrepancies(filename, progress_bar=True)
print(pandas.DataFrame(data))


# %%
# Minimal script to export a LLM
# ++++++++++++++++++++++++++++++
#
# The following lines are a condensed copy with less comments.

# from HuggingFace
print("----------------")
MODEL_NAME = "arnir0/Tiny-LLM"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# from HuggingFace again
prompt = "Continue: it rains, what should I do?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_length=100,
    temperature=1,
    top_k=50,
    top_p=0.95,
    do_sample=True,
)

observer = InputObserver()
with register_additional_serialization_functions(patch_transformers=True), observer:
    generate_text(prompt, model, tokenizer)

filename = "plot_export_tiny_llm_input_observer.onnx"
with torch_export_patches(patch_transformers=True):
    torch.onnx.export(
        model,
        (),
        filename,
        kwargs=observer.infer_arguments(),
        dynamic_shapes=observer.infer_dynamic_shapes(set_batch_dimension_for=True),
    )

data = observer.check_discrepancies(filename, progress_bar=True)
print(pandas.DataFrame(data))


# %%
doc.save_fig(doc.plot_dot(filename), f"{filename}.png", dpi=400)
