"""
.. _l-plot-tiny-llm-export-dim01-onnx-custom:

Export with dynamic dimensions in ``{0,1}`` into ONNX (custom)
==============================================================

This duplicates the example :ref:`l-plot-tiny-llm-export-dim01` but for
:func:`experimental_experiment.torch_interpreter.to_onnx`.
It checks what inputs can be used to export and with which inputs it can work.

Available input sets
++++++++++++++++++++

"""

import itertools
from tqdm import tqdm
import numpy as np
import pandas
import onnxruntime
from onnx_diagnostic import doc
from onnx_diagnostic.helpers import max_diff, string_type, flatten_object
from onnx_diagnostic.helpers.torch_helper import torch_deepcopy
from onnx_diagnostic.helpers.rt_helper import make_feeds
from onnx_diagnostic.torch_models.hghub.model_inputs import get_untrained_model_with_inputs
from onnx_diagnostic.torch_export_patches import (
    torch_export_patches,
    register_additional_serialization_functions,
)
from experimental_experiment.torch_interpreter import to_onnx, ExportOptions


data = get_untrained_model_with_inputs("arnir0/Tiny-LLM", add_second_input=True)
model, dynamic_shapes = data["model"], data["dynamic_shapes"]

# %%
# The trained model can be obtained with:
#
# .. code-block:: python
#
#   MODEL_NAME = "arnir0/Tiny-LLM"
#   tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
#   model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_NAME)

input_sets = {k: v for k, v in data.items() if k.startswith("inputs")}

for k, v in input_sets.items():
    print(f"{k:20}: {string_type(v, with_shape=True)}")

# %%
# The dynamic shapes are:

print(f"dynamic_shapes: {string_type(dynamic_shapes)}")

# %%
# Let's check they all work and compute the expected values.
# We use deepcopy because caches are usually modified inplace.

expected = {}
for k, v in input_sets.items():
    expected[k] = model(**torch_deepcopy(v))
    print(f"{k:20}: {string_type(expected[k], with_shape=True)}")

# %%
# Export with options
# +++++++++++++++++++
#
# We try to export with the following options:
# - cache registration: register cache serialization with
#   :func:`onnx_diagnostic.torch_export_patches.register_additional_serialization_functions`
# - oblivious: an option to remove some the exception raises by the exporter
# - rt: see ``prefer_deferred_runtime_asserts_over_guards`` in :func:`torch.export.export`
# - cache_patch: patches the model before exporting with
#   :func:`onnx_diagnostic.torch_export_patches.torch_export_patches`
#
# Some function first.


def export_model(
    model, dynamic_shapes, inputs, cache=False, oblivious=False, rt=False, cache_patch=False
):
    if cache and not cache_patch:
        with register_additional_serialization_functions(patch_transformers=True):
            return export_model(model, dynamic_shapes, inputs, oblivious=oblivious, rt=rt)
    if cache_patch:
        with torch_export_patches(
            patch_torch=cache_patch in ("all", "torch", True, 1),
            patch_transformers=cache_patch in ("all", "transformers", True, 1),
        ):
            return export_model(model, dynamic_shapes, inputs, oblivious=oblivious, rt=rt)
    return to_onnx(
        model,
        (),
        kwargs=inputs,
        dynamic_shapes=dynamic_shapes,
        export_options=ExportOptions(
            prefer_deferred_runtime_asserts_over_guards=rt,
            backed_size_oblivious=oblivious,
        ),
    )


def try_export_model(
    model, dynamic_shapes, inputs, cache=False, oblivious=False, rt=False, cache_patch=False
):
    try:
        return export_model(
            model,
            dynamic_shapes,
            inputs,
            cache=cache,
            oblivious=oblivious,
            rt=rt,
            cache_patch=cache_patch,
        )
    except Exception as e:
        return e


def validation(onx, input_sets, expected, catch_exception=True):
    sess = onnxruntime.InferenceSession(
        onx.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    for k, v in input_sets.items():
        feeds = make_feeds(sess, torch_deepcopy(v), use_numpy=True)
        try:
            got = sess.run(None, feeds)
        except Exception as e:
            if not catch_exception:
                raise
            yield k, e
            continue
        yield k, max_diff(flatten_object(expected[k], drop_keys=True), got)


# %%
# Verification an example known to be working is.

ep = export_model(
    model,
    dynamic_shapes,
    torch_deepcopy(input_sets["inputs"]),
    cache_patch=True,
)
res = list(validation(ep, dict(inputs=input_sets["inputs"]), expected, catch_exception=False))
assert res[0][1]["abs"] < 1e-5, f"Unexpected issue with res={res}"


# %%
# The main loop
# +++++++++++++

results = []

possibilities = [
    [0, 1],
    [0, "all", "torch", "transformers"],
    [0, 1, "auto", "half"],
    [0, 1],
    list(input_sets),
]
with tqdm(list(itertools.product(*possibilities))) as pbar:
    for cache, cache_patch, oblivious, rt, inputs in pbar:
        if cache_patch and not cache:
            # patches include caches.
            continue
        kwargs = dict(cache=cache, cache_patch=cache_patch, oblivious=oblivious, rt=rt)
        legend = "-".join(
            (k if isinstance(v, int) else f"{k}:{v}") for k, v in kwargs.items() if v
        )
        legend = f"{legend}/{inputs}"
        pbar.set_description(f"{legend} EXPORT")

        # export
        ep = try_export_model(
            model, dynamic_shapes, torch_deepcopy(input_sets[inputs]), **kwargs
        )
        if isinstance(ep, Exception):
            obs = {
                **kwargs,
                "export_with": inputs,
                "EXPORT": 0,
                "ERR-EXPORT": str(ep).split("\n")[0],
            }
            results.append(obs)
            continue

        pbar.set_description(f"{legend} VALIDATE")
        common = {**kwargs, "export_with": inputs, "EXPORT": 1}
        for inp, res in validation(ep, input_sets, expected):
            if isinstance(res, Exception):
                obs = {
                    **common,
                    "run_with": inp,
                    "ERR-RUN": str(res).split("\n")[0],
                    "WORKS": 0,
                }
            else:
                obs = {
                    **common,
                    "run_with": inp,
                    "WORKS": int(~np.isnan(res["abs"]) and res["abs"] < 1e-3),
                }
            results.append(obs)

# %%
# Let's save the results.

df = pandas.DataFrame(results)
df.to_excel("plot_export_tiny_llm_dim01_onnx.xlsx")
df

# %% The export failures.

no_export = df[df.EXPORT == 0]
no_export.to_excel("plot_export_tiny_llm_dim01_onnx_custom.no_export.xlsx")
no_export

# %%
# The validation failures.

invalid = df[(df.EXPORT == 1) & (df.WORKS == 0)].pivot(
    index=["cache", "cache_patch", "oblivious", "rt", "export_with"],
    columns=["run_with"],
    values=["WORKS", "ERR-RUN"],
)
invalid.to_excel("plot_export_tiny_llm_dim01_onnx_custom.invalid.xlsx")
invalid

# %% Successes.

success = df[(df.EXPORT == 1) & (df.WORKS == 1)].pivot(
    index=["cache", "cache_patch", "oblivious", "rt", "export_with"],
    columns=["run_with"],
    values=["WORKS"],
)
success.to_excel("plot_export_tiny_llm_dim01_onnx_custom.success.xlsx")
success


# %%
# If you have any error, then look at example
# :ref:`l-plot-tiny-llm-export-patched`.

doc.plot_legend("Tiny-LLM\nexport with\ndimension in {0,1}", "to_onnx", "tomato")
