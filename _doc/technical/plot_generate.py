"""
.. _l-plot-generate:

=================================
From a LLM to processing a prompt
=================================

Method ``generate`` generates the model answer for a given prompt.
Let's implement our own to understand better how it works and
then apply it to an ONNX model.

Example with Phi 1.5
====================

epkg:`microsoft/Phi-1.5` is a small LLM. The example given
"""

import os
import time
import sys
import pandas
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from onnx_diagnostic.ext_test_case import unit_test_going
from onnx_diagnostic.helpers import string_type
from onnx_diagnostic.helpers.torch_helper import to_any, get_weight_type
from onnx_diagnostic.helpers.rt_helper import onnx_generate
from onnx_diagnostic.torch_export_patches import torch_export_patches
from onnx_diagnostic.torch_models.hghub import get_untrained_model_with_inputs
from onnx_diagnostic.torch_models.hghub.hub_api import get_pretrained_config, task_from_id
from onnx_diagnostic.tasks import random_input_kwargs
from onnx_diagnostic.export.api import to_onnx


device = "cuda" if torch.cuda.is_available() else "cpu"
data = []

print("-- load the model...")
if unit_test_going():
    # unit_test_going() returns True if UNITTEST_GOING is 1
    # The example switches to a faster scenario.
    model_id = "arnir0/Tiny-LLM"
    data_export = get_untrained_model_with_inputs(model_id)
    model = data_export["model"]
    export_inputs = data_export["inputs"]
    export_shapes = data_export["dynamic_shapes"]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
else:
    model_id = "microsoft/phi-1_5"
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    config = get_pretrained_config(model_id)
    task = task = task_from_id(model_id)
    kwargs, fct = random_input_kwargs(config, task)
    res = fct(model, config, add_second_input=False, **kwargs)
    export_inputs = res["inputs"]
    export_shapes = res["dynamic_shapes"]
model = model.to(device)
print("-- done.")

print("-- tokenize the prompt...")
inputs = tokenizer(
    '''def print_prime(n):
   """
   Print all primes between 1 and n
   """''',
    return_tensors="pt",
    return_attention_mask=False,
).to(device)
print("-- done.")

print("-- compute the answer...")
begin = time.perf_counter()
outputs = model.generate(**inputs, max_new_tokens=100)
duration = time.perf_counter() - begin
print(f"-- done in {duration}")
data.append(dict(name="generate", duration=duration))
print("output shape:", string_type(outputs, with_shape=True, with_min_max=True))
print("-- decode the answer...")
text = tokenizer.batch_decode(outputs)[0]
print("-- done.")
print(text)


# %%
# eos_token_id?
# =============
#
# This token means the end of the answer.

print("eos_token_id=", tokenizer.eos_token_id)

# %%
# Custom method generate
# ======================
#
# Let's implement a simple function replicating when method
# ``generate`` does.


def simple_generate_with_cache(
    model, input_ids: torch.Tensor, eos_token_id: int, max_new_tokens: int = 100
):
    # First call: prefill
    outputs = model(input_ids, use_cache=True)

    # Next calls: decode
    for _ in tqdm(list(range(max_new_tokens))):
        next_token_logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values

        # The most probable next token is chosen.
        next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        # But we could select it using a multinomial law
        # <<< probs = torch.softmax(next_token_logits / temperature, dim=-1)
        # <<< top_probs, top_indices = torch.topk(probs, top_k)
        # <<< next_token_id = top_indices[torch.multinomial(top_probs, 1)]

        if next_token_id.item() == eos_token_id:
            break
        input_ids = torch.cat([input_ids, next_token_id], dim=-1)

        # Feed only the new token, but with the cache
        outputs = model(next_token_id, use_cache=True, past_key_values=past_key_values)

    return input_ids


print("-- compute the answer with custom generate...")
begin = time.perf_counter()
outputs = simple_generate_with_cache(
    model, inputs.input_ids, eos_token_id=tokenizer.eos_token_id, max_new_tokens=100
)
duration = time.perf_counter() - begin
print(f"-- done in {duration}")
data.append(dict(name="custom", duration=duration))

print("-- done.")
print("output shape:", string_type(outputs, with_shape=True, with_min_max=True))
print("-- decode the answer...")
text = tokenizer.batch_decode(outputs)[0]
print("-- done.")
print(text)

# %%
# Method generate for onnx models
# ===============================
#
# We first need to export the model into ONNX.
#
# ONNX Conversion
# +++++++++++++++

if "position_ids" in export_inputs:
    del export_inputs["position_ids"]
    del export_shapes["position_ids"]
dtype = get_weight_type(model)
print("-- model dtype:", dtype)
export_inputs["past_key_values"] = to_any(export_inputs["past_key_values"], dtype)
exporter = "onnx-dynamo" if "dynamo" in sys.argv else "custom"
model_name = f"model_{model_id.replace('/', '-')}.{exporter}.onnx"
if not os.path.exists(model_name):
    # This step is slow so let's skip it if it was already done.
    print("-- conversion to ONNX.")
    begin = time.perf_counter()
    with torch_export_patches(patch_transformers=True):
        to_onnx(
            model,
            (),
            kwargs=to_any(export_inputs, device),
            dynamic_shapes=export_shapes,
            filename=model_name,
            verbose=1,
            exporter=exporter,
        )
    duration = time.perf_counter() - begin
    print(f"-- done in {duration}")

# %%
# onnx_generate
# +++++++++++++
#
# Then we can call method generate for two tokens.
# This function is part of :mod:`onnx_diagnostic` but follows the implementation
# seen earlier for a torch model.
# Let's ask first the function to return the session to avoid creating on the second call.

_res, session = onnx_generate(
    model_name, inputs.input_ids, 2, max_new_tokens=2, return_session=True
)

# And now the full answer.
print("-- compute the answer with custom generate...")
begin = time.perf_counter()
outputs = onnx_generate(
    session, inputs.input_ids, eos_token_id=tokenizer.eos_token_id, max_new_tokens=100
)
duration = time.perf_counter() - begin
print(f"-- done in {duration}")
data.append(dict(name="onnx", duration=duration))

print("-- done.")
print("output shape:", string_type(outputs, with_shape=True, with_min_max=True))
print("-- decode the answer...")
text = tokenizer.batch_decode(outputs)[0]
print("-- done.")
print(text)


# %%
# Plots
# =====
df = pandas.DataFrame(data).set_index("name")
print(df)

# %%
ax = df.plot(kind="bar", title="Time (s) comparison to generate a prompt.", rot=45)
ax.figure.tight_layout()
ax.figure.savefig("plot_generate.png")
