"""
.. _l-plot-export-model-builder:

Export with ModelBuilder
========================

"""

import os
import pandas
from transformers import AutoModelForCausalLM, AutoTokenizer
from onnx_diagnostic import doc
from onnx_diagnostic.investigate.input_observer import InputObserver
from onnx_diagnostic.helpers.rt_helper import onnx_generate
from onnx_diagnostic.torch_export_patches import (
    register_additional_serialization_functions,
    torch_export_patches,
)
from onnx_diagnostic.export.api import to_onnx


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
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

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


# %%
# Creating the model
print("-- creating...")
MODEL_NAME = "arnir0/Tiny-LLM"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)


# %%
# Capturing inputs/outputs to infer dynamic shapes and arguments
print("-- capturing...")
prompt = "Continue: it rains, what should I do?"
observer = InputObserver()
with register_additional_serialization_functions(patch_transformers=True), observer(model):
    generate_text(prompt, model, tokenizer)


# %%
# Exporting.
print("-- exporting...")
observer.remove_inputs(["cache_position", "logits_to_keep", "position_ids"])
ds = observer.infer_dynamic_shapes(set_batch_dimension_for=True)
kwargs = observer.infer_arguments()

cache_dir = "dump_modelbuilder"
os.makedirs(cache_dir, exist_ok=True)
filename = os.path.join(cache_dir, "plot_export_with_modelbuilder.onnx")
with torch_export_patches(patch_transformers=True):
    to_onnx(
        model,
        filename=filename,
        kwargs=kwargs,
        dynamic_shapes=ds,
        exporter="modelbuilder",
    )

data = observer.check_discrepancies(filename, progress_bar=True)
print(pandas.DataFrame(data))

# %%
# ONNX Prompt
# +++++++++++
print("-- ONNX prompts...")
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

onnx_tokens = onnx_generate(
    filename,
    input_ids=input_ids,
    attention_mask=attention_mask,
    eos_token_id=model.config.eos_token_id,
    max_new_tokens=50,
)
onnx_generated_text = tokenizer.decode(onnx_tokens, skip_special_tokens=True)

print("-----------------")
print("\n".join(onnx_generated_text))
print("-----------------")

# %%
doc.save_fig(doc.plot_dot(filename), f"{filename}.png", dpi=400)
