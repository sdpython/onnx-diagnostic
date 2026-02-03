"""
.. _l-plot-whisper-tiny-export-input-observer:

Export whisper-tiny with InputObserver
======================================

This reuses the recipe introduced by example :ref:`l-plot-tiny-llm-export-input-observer`
for model `openai/whisper-tiny <https://huggingface.co/openai/whisper-tiny>`_.

The model
+++++++++
"""

import pandas
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset
from onnx_diagnostic import doc
from onnx_diagnostic.helpers import string_type
from onnx_diagnostic.export.api import to_onnx
from onnx_diagnostic.torch_export_patches import (
    register_additional_serialization_functions,
    torch_export_patches,
)
from onnx_diagnostic.investigate.input_observer import InputObserver

# load model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
model.config.forced_decoder_ids = None

# load dummy dataset and read audio files
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
samples = [ds[0]["audio"], ds[2]["audio"]]
for s in samples:
    print(s["array"].shape, s["array"].min(), s["array"].max(), s["sampling_rate"])
input_features = [
    processor(
        sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt"
    ).input_features
    for sample in samples
]

# %%
# Captures inputs and outputs for the encoder, decoder.
observer_encoder, observer_decoder = InputObserver(), InputObserver()
with register_additional_serialization_functions(patch_transformers=True):
    for features in input_features:
        with (
            observer_encoder(model.model.encoder, store_n_calls=4),
            observer_decoder(model.model.decoder, store_n_calls=4),
        ):
            predicted_ids = model.generate(features)


print(f"{observer_encoder.num_obs()} observations stored for encoder.")
print(f"{observer_decoder.num_obs()} observations stored for decoder.")

# %%
# Export the encoder
# ++++++++++++++++++
kwargs = observer_encoder.infer_arguments()
dynamic_shapes = observer_encoder.infer_dynamic_shapes(set_batch_dimension_for=True)
print(f"encoder kwargs={string_type(kwargs, with_shape=True)}")
print(f"encoder dynamic_shapes={dynamic_shapes}")
for candidate in observer_encoder.info.inputs:
    print(
        "   ",
        candidate,
        candidate.str_obs(),
        string_type(candidate.aligned_flat_list, with_shape=True),
    )


filename_encoder = "plot_export_whisper_tiny_input_observer_encoder.onnx"
with torch_export_patches(patch_transformers=True):
    to_onnx(
        model.model.encoder,
        args=(),
        filename=filename_encoder,
        kwargs=kwargs,
        dynamic_shapes=dynamic_shapes,
        exporter="custom",
    )

# %%
# Let's measure the discrepancies.
data = observer_encoder.check_discrepancies(filename_encoder, progress_bar=True)
print(pandas.DataFrame(data))

# %%
# Export the decoder
# ++++++++++++++++++

kwargs = observer_decoder.infer_arguments()
dynamic_shapes = observer_decoder.infer_dynamic_shapes(set_batch_dimension_for=True)
print(f"decoder kwargs={string_type(kwargs, with_shape=True)}")
print(f"decoder dynamic_shapes={dynamic_shapes}")

filename_decoder = "plot_export_whisper_tiny_input_observer_decoder.onnx"
with torch_export_patches(patch_transformers=True):
    to_onnx(
        model.model.decoder,
        args=(),
        filename=filename_decoder,
        kwargs=observer_decoder.infer_arguments(),
        dynamic_shapes=observer_decoder.infer_dynamic_shapes(set_batch_dimension_for=True),
        exporter="custom",
    )

# %%
# Let's measure the discrepancies.
data = observer_decoder.check_discrepancies(filename_decoder, progress_bar=True, atol=1e-3)
print(pandas.DataFrame(data))


# %%
doc.save_fig(doc.plot_dot(filename_decoder), f"{filename_decoder}.png", dpi=400)
