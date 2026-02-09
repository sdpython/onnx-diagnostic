"""
.. _l-plot-tiny-llm-attention-export-input-observer:

Export attention from arnir0/Tiny-LLM with InputObserver
========================================================

This shows how to only export attention from model
`arnir0/Tiny-LLM <https://huggingface.co/arnir0/Tiny-LLM>`_.
It uses what was shown in example
:ref:`l-plot-tiny-llm-export-input-observer`.

Let's create a random model
+++++++++++++++++++++++++++
"""

import pandas
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from onnx_diagnostic import doc
from onnx_diagnostic.export.api import to_onnx
from onnx_diagnostic.helpers import string_type
from onnx_diagnostic.torch_export_patches import (
    register_additional_serialization_functions,
    torch_export_patches,
)
from onnx_diagnostic.investigate.input_observer import InputObserver

device = "cuda"
model_id = "arnir0/Tiny-LLM"
print(f"get tokenizer {model_id!r}")
tokenizer = AutoTokenizer.from_pretrained(model_id)
print(f"get config {model_id!r}")
config = AutoConfig.from_pretrained(model_id)
print(f"create model from config for {model_id!r}")
model = AutoModelForCausalLM.from_config(config)
print(f"the model is created with {len(list(model.named_modules()))} subdmodules.")
model = model.to(device).to(torch.float16)

# %%
# We need to only export class LlamaAttention
# +++++++++++++++++++++++++++++++++++++++++++


export_module = None
for _name, sub in model.named_modules():
    if sub.__class__.__name__ == "LlamaAttention":
        export_module = sub

assert export_module is not None, (
    f"Unable to find a submodule from class LlamaAttention in "
    f"{set(sub.__class__.__name__ for _, sub in model.named_modules())}"
)

# %%
# Let's run the model and capture the inputs and outputs of the attention part.


def generate_text(
    prompt,
    model,
    tokenizer,
    max_length=50,
    temperature=0.01,
    top_k=50,
    top_p=0.95,
    do_sample=True,
):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=do_sample,
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


prompt = "Continue: it rains, what should I do?"
observer = InputObserver()
with (
    register_additional_serialization_functions(patch_transformers=True),
    observer(export_module),
):
    generate_text(prompt, model, tokenizer)


# %%
# Export
# ++++++
#
# First, what was inferred.

kwargs = observer.infer_arguments()
dynamic_shapes = observer.infer_dynamic_shapes()
print(f"kwargs={string_type(kwargs, with_shape=True, with_device=True)}")
print(f"dynamic_shapes={dynamic_shapes}")

# %%
# Next, the export.


filename = "plot_export_tiny_llm_attention_input_observer.onnx"
with torch_export_patches(patch_transformers=True):
    to_onnx(
        export_module,
        args=(),
        kwargs=kwargs,
        filename=filename,
        dynamic_shapes=dynamic_shapes,
        exporter="custom",
        verbose=1,
    )

# %%
# Let's measure the discrepancies.
data = observer.check_discrepancies(filename, progress_bar=True, atol=1e-2, include_io=True)
df = pandas.DataFrame(data)
df.to_excel("plot_export_tiny_llm_attention_input_observer.xlsx")
print(df)

# %%
# Let's show the errors.
for row in data:
    if not row["SUCCESS"] and "error" in row:
        print(row["error"])


# %%
doc.save_fig(doc.plot_dot(filename), f"{filename}.png", dpi=400)
