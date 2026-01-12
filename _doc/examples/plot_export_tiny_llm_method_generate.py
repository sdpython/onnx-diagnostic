"""
.. _l-plot-tiny-llm-export-method-generate:

Export a LLM through method generate (with Tiny-LLM)
====================================================

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
from transformers import AutoModelForCausalLM, AutoTokenizer
from onnx_diagnostic import doc
from onnx_diagnostic.export.api import method_to_onnx


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
prompt = "Continue: it rains..."
generated_text = generate_text(prompt, model, tokenizer)
print("-----------------")
print(generated_text)
print("-----------------")

# %%
# Replace forward method
# ++++++++++++++++++++++
#
# We now modify the model to export the model by replacing the forward method.
# We still call method ``generate`` but this one will call a different function
# created by :func:`onnx_diagnostic.export.api.method_to_onnx`.
# This one captured the inputs of the forward method, 2 calls are needed or
# at least, 3 are recommended for LLMs as the first call does not contain any cache.
# If the default settings do not work, ``skip_kwargs_names`` and ``dynamic_shapes``
# can be changed to remove some undesired inputs or add more dynamic dimensions.

filename = "plot_export_tiny_llm_method_generate.onnx"
forward_replacement = method_to_onnx(
    model,
    method_name="forward",  # default value
    exporter="custom",  # onnx-dynamo to use the official exporter
    filename=filename,  # onnx file to create
    patch_kwargs=dict(patch_transformers=True),  # patches before eporting
    # to see the progress, it is recommended on the first try to see
    # how to set ``skip_kwargs_names`` and ``dynamic_shapes`` if it is needed
    verbose=1,
    # triggers the ONNX conversion after 3 calls to forward method,
    # the onnx version is triggered with the last one,
    # the others are used to infer the dynamic shapes if they are not
    # specified below
    convert_after_n_calls=3,
    # The input used in the example has a batch size equal to 1, all
    # inputs going through method forward will have the same batch size.
    # To force the dynamism of this dimension, we need to indicate
    # which inputs have a batch size.
    dynamic_batch_for={"input_ids", "attention_mask", "past_key_values"},
)

# %%
# dynamic shapes can be inferred from at least two calls to the forward method,
# 3 is better for LLMs (first call is prefill, cache is missing),
# you can see the inference results with ``verbose=1``.
# If the value is not the expected one (to change the names for example),
# They can be overwritten.
#
# .. code-block:: python
#
#   dynamic_shapes={
#       "cache_position": {0: "sequence_length"},
#       "past_key_values": [
#           {0: "batch_size", 2: "past_sequence_length"},
#           {0: "batch_size", 2: "past_sequence_length"},
#       ],
#       "input_ids": {0: "batch_size", 1: "sequence_length"},
#       "attention_mask": {0: "batch_size", 1: "total_sequence_length"},
#   }
#
# Finally, we need to replace the forward method.
# As ``forward_replacement`` is a module of type
# :class:`onnx_diagnostic.export.api.WrapperToExportMethodToOnnx`,
# a lambda function must be used to avoid this one to be
# included as a submodule (and create an infinite loop).

print(f"type(forward_replacement)={type(forward_replacement)}")
model.forward = lambda *args, **kwargs: forward_replacement(*args, **kwargs)


# %%
# Let's call generate again. The conversion is triggered after
# ``convert_after_n_calls=3`` calls to the method forward,
# which exactly what the method generate is doing.
generated_text = generate_text(prompt, model, tokenizer)
print(generated_text)

# %%
# We finally need to check the discrepancies.
# The exports produced an onnx file and dumped the input and output
# of the torch model. We now run the onnx model to check
# it produces the same results.
# It is done after because the model may not hold twice in memory
# (torch and onnxruntime).
# verbose=2 shows more information about expected outputs.
data = forward_replacement.check_discrepancies(verbose=1)
df = pandas.DataFrame(data)
print(df)

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

# to export into onnx
forward_replacement = method_to_onnx(
    model,
    method_name="forward",
    exporter="custom",
    filename="plot_export_tiny_llm_method_generate.onnx",
    patch_kwargs=dict(patch_transformers=True),
    verbose=0,
    convert_after_n_calls=3,
    dynamic_batch_for={"input_ids", "attention_mask", "past_key_values"},
)
model.forward = lambda *args, **kwargs: forward_replacement(*args, **kwargs)

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
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("prompt answer:", generated_text)

# to check discrepancies
data = forward_replacement.check_discrepancies()
df = pandas.DataFrame(data)
print(df)


# %%
doc.save_fig(doc.plot_dot(filename), f"{filename}.png", dpi=400)
