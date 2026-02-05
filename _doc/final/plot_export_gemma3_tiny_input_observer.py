"""
.. _l-plot-gemma3-tiny-export-input-observer:

Export Gemma3 tiny random with InputObserver
============================================

This reuses the recipe introduced by example :ref:`l-plot-tiny-llm-export-input-observer`
for model `tiny-random/gemma-3 <https://huggingface.co/tiny-random/gemma-3>`_.
"""

import pandas
import torch
from onnx_diagnostic import doc
from onnx_diagnostic.helpers import string_type
from onnx_diagnostic.export.api import to_onnx
from onnx_diagnostic.torch_export_patches import (
    register_additional_serialization_functions,
    torch_export_patches,
)
from onnx_diagnostic.investigate.input_observer import InputObserver


from transformers import pipeline

model_id = "tiny-random/gemma-3"
pipe = pipeline(
    "image-text-to-text",
    model=model_id,
    device="cpu",
    trust_remote_code=True,
    max_new_tokens=3,
    dtype=torch.float16,
)
messages = [
    {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG",
            },
            {"type": "text", "text": "What animal is on the candy?"},
        ],
    },
]


# %%
# The model to observe.
print("model type:", type(pipe.model))

# %%
# Captures inputs and outputs for the model.
observer = InputObserver(
    missing=dict(pixel_values=torch.empty((0, 3, 896, 896), dtype=torch.float16))
)
with (
    register_additional_serialization_functions(patch_transformers=True),
    observer(pipe.model),
):
    pipe(text=messages, max_new_tokens=4)


print(f"{observer.num_obs()} observations stored for encoder.")

# %%
# Exports the model.
kwargs = observer.infer_arguments()
dynamic_shapes = observer.infer_dynamic_shapes(set_batch_dimension_for=True)
print(f"encoder kwargs={string_type(kwargs, with_shape=True)}")
print(f"encoder dynamic_shapes={dynamic_shapes}")
for candidate in observer.info.inputs:
    print(
        "   ",
        candidate,
        candidate.str_obs(),
        string_type(candidate.aligned_flat_list, with_shape=True),
    )


filename = "plot_export_gemma3_tiny_input_observer.onnx"
with torch_export_patches(patch_transformers=True, patch_torch=True, stop_if_static=2):
    to_onnx(
        pipe.model,
        args=(),
        filename=filename,
        kwargs=kwargs,
        dynamic_shapes=dynamic_shapes,
        exporter="custom",
    )

# %%
# Let's measure the discrepancies.
data = observer.check_discrepancies(filename, progress_bar=True, atol=1e-2, include_io=True)
df = pandas.DataFrame(data)
df.to_excel("plot_export_gemma3_tiny_input_observer.xlsx")
print(df)

# %%
# Let's show the errors.
for row in data:
    if not row["SUCCESS"] and "error" in row:
        print(row["error"])


# %%
doc.save_fig(doc.plot_dot(filename), f"{filename}.png", dpi=400)
