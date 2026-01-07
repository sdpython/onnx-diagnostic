"""
.. _l-plot-tiny-llm-export-method-generate:

Export a model through method generate (with Tiny-LLM)
======================================================

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

from transformers import AutoModelForCausalLM, AutoTokenizer
from onnx_diagnostic import doc
from onnx_diagnostic.export.api import method_to_onnx


MODEL_NAME = "arnir0/Tiny-LLM"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)


def generate_text(
    prompt, model, tokenizer, max_length=50, temperature=1, top_k=50, top_p=0.95
):
    inputs = tokenizer.encode(prompt, return_tensors="pt")

    outputs = model.generate(
        inputs,
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
    # skips the following inputs even though they are captured,
    # these ones are filled with default values we don't want in
    # the onnx model
    skip_kwargs_names={"kwargs", "use_cache", "return_dict", "inputs_embeds"},
    # dynamic shapes can be inferred from at least two calls to the forward method,
    # 3 is better for LLMs, you can see the inference results with ``verbose=1``,
    # this parameter is used to overwrite the inferred values,
    # this is usually needed because the inferred dynamic shapes contains
    # less dynamic dimension than requested.
    dynamic_shapes={
        "cache_position": {0: "total_sequence_length"},
        "past_key_values": [
            {0: "batch_size", 2: "past_sequence_length"},
            {0: "batch_size", 2: "past_sequence_length"},
        ],
        "input_ids": {0: "batch_size", 1: "sequence_length"},
    },
)

# %%
# The lambda function cannot be skipped as
# forward_replacement is a module.

print(f"type(forward_replacement)={type(forward_replacement)}")
model.forward = lambda *args, **kwargs: forward_replacement(*args, **kwargs)


# %%
# Let's call generate again.
generated_text = generate_text(prompt, model, tokenizer)
print(generated_text)


# %%

doc.plot_legend("Tiny-LLM\nforward inputs\nthrough generate", "onnx export", "tomato")
