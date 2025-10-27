"""
.. _l-plot-generate:

=================================
From a LLM to processing a prompt
=================================

Method ``generate`` generates the model answer fro a given prompt.
Let's implement our own to understand better how it works.

Example with Phi 1.5
====================

epkg:`microsoft/Phi-1.5` is a small LLM. The example given
"""

import time
import pandas
from tqdm import tqdm
from onnx_diagnostic.ext_test_case import unit_test_going
from onnx_diagnostic.helpers import string_type
from onnx_diagnostic.torch_models.hghub import get_untrained_model_with_inputs
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available else "cpu"
data = []

print("-- load the model...")
# unit_test_going() returns True if UNITTEST_GOING is 1
if unit_test_going():
    model_id = "arnir0/Tiny-LLM"
    model = get_untrained_model_with_inputs(model_id)["model"]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
else:
    model_id = "microsoft/phi-1_5"
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
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
outputs = model.generate(**inputs, max_length=100)
duration = time.perf_counter() - begin
print(f"-- done in {duration}")
data.append(dict(name="generate", duration=duration))
print("output shape:", string_type(outputs, with_shape=True))
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


def simple_generate_with_cache(
    model, input_ids: torch.Tensor, eos_token_id: int, max_new_tokens: int = 100
):
    answer = []
    # First call.
    outputs = model(input_ids, use_cache=True)
    next_token_logits = outputs.logits[:, -1, :]
    past_key_values = outputs.past_key_values

    # Next calls.
    for _ in tqdm(list(range(max_new_tokens))):
        # The most probable next token is chosen.
        next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        # But we could select it using a multinomial law
        # <<< probs = torch.softmax(next_token_logits / temperature, dim=-1)
        # <<< top_probs, top_indices = torch.topk(probs, top_k)
        # <<< next_token_id = top_indices[torch.multinomial(top_probs, 1)]

        # Let's add the predicted token to the answer.
        answer.append(next_token_id)

        # Feed only the new token, but with the cache
        outputs = model(next_token_id, use_cache=True, past_key_values=past_key_values)
        next_token_logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values

        input_ids = torch.cat([input_ids, next_token_id], dim=-1)

        if next_token_id.item() == eos_token_id:
            break

    return torch.cat(answer, dim=1)


print("-- compute the answer with custom generate...")
begin = time.perf_counter()
outputs = simple_generate_with_cache(
    model, inputs.input_ids, eos_token_id=tokenizer.eos_token_id, max_new_tokens=100
)
duration = time.perf_counter() - begin
print(f"-- done in {duration}")
data.append(dict(name="custom", duration=duration))

print("-- done.")
print("output shape:", string_type(outputs, with_shape=True))
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
