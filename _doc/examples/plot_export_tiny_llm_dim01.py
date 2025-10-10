"""
.. _l-plot-tiny-llm-export-dim01:

Export with dynamic dimensions in ``{0,1}``
===========================================

The first version of :func:`torch.export.export` did not support
any tensor with a dimension equal to 0, 1 if the dimension was expected
to be dynamic. The latest versions offers more options. Let's check it works.
The experiments consists in exporting the model with different sets of inputs
and checking the exported models works with all set of inputs.

Available input sets
++++++++++++++++++++

"""

import itertools
from tqdm import tqdm
import numpy as np
import pandas
import torch
from onnx_diagnostic import doc
from onnx_diagnostic.helpers import max_diff, string_type
from onnx_diagnostic.helpers.torch_helper import torch_deepcopy
from onnx_diagnostic.torch_models.hghub.model_inputs import get_untrained_model_with_inputs
from onnx_diagnostic.torch_export_patches.patch_inputs import use_dyn_not_str
from onnx_diagnostic.torch_export_patches import (
    torch_export_patches,
    register_additional_serialization_functions,
)


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

# %% The exporter does not support strings.

dynamic_shapes = use_dyn_not_str(dynamic_shapes)
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
#
# - cache registration: register cache serialization with
#   :func:`onnx_diagnostic.torch_export_patches.register_additional_serialization_functions`
#
# - oblivious: an option to remove some the exception raises by the exporter
#
# - rt: see ``prefer_deferred_runtime_asserts_over_guards`` in :func:`torch.export.export`
#
# - cache_patch: patches the model before exporting with
#   :func:`onnx_diagnostic.torch_export_patches.torch_export_patches`
#
# Some function first.


def export_model(
    model,
    dynamic_shapes,
    inputs,
    cache=False,
    oblivious=False,
    rt=False,
    cache_patch=False,
    strict=False,
):
    if cache and not cache_patch:
        with register_additional_serialization_functions(patch_transformers=True):
            return export_model(
                model, dynamic_shapes, inputs, oblivious=oblivious, rt=rt, strict=strict
            )
    if cache_patch:
        with torch_export_patches(
            patch_torch=cache_patch in ("all", "torch", True, 1),
            patch_transformers=cache_patch in ("all", "transformers", True, 1),
        ):
            return export_model(
                model, dynamic_shapes, inputs, oblivious=oblivious, rt=rt, strict=strict
            )
    if oblivious:
        with torch.fx.experimental._config.patch(backed_size_oblivious=True):
            return export_model(model, dynamic_shapes, inputs, rt=rt, strict=strict)
    return torch.export.export(
        model,
        (),
        inputs,
        dynamic_shapes=dynamic_shapes,
        strict=strict,
        prefer_deferred_runtime_asserts_over_guards=rt,
    )


def try_export_model(
    model,
    dynamic_shapes,
    inputs,
    cache=False,
    oblivious=False,
    rt=False,
    cache_patch=False,
    strict=False,
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
            strict=strict,
        )
    except Exception as e:
        return e


def validation(ep, input_sets, expected):
    mod = ep.module()
    for k, v in input_sets.items():
        try:
            got = mod(**torch_deepcopy(v))
        except Exception as e:
            yield k, e
            continue
        yield k, max_diff(expected[k], got, verbose=0)


# %%
# The main loop
# +++++++++++++

results = []

possibilities = [*[[0, 1] for _ in range(5)], list(input_sets)]
possibilities[1] = [0, "all", "torch", "transformers"]
with tqdm(list(itertools.product(*possibilities))) as pbar:
    for cache, cache_patch, strict, oblivious, rt, inputs in pbar:
        if cache_patch and not cache:
            # patches include caches.
            continue
        kwargs = dict(
            cache=cache, cache_patch=cache_patch, strict=strict, oblivious=oblivious, rt=rt
        )
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
df.to_excel("plot_export_tiny_llm_dim01.xlsx")
df

# %% The export failures.

no_export = df[df.EXPORT == 0]
no_export.to_excel("plot_export_tiny_llm_dim01.no_export.xlsx")
no_export

# %%
# The validation failures.

invalid = df[(df.EXPORT == 1) & (df.WORKS == 0)].pivot(
    index=["cache", "cache_patch", "strict", "oblivious", "rt", "export_with"],
    columns=["run_with"],
    values=["WORKS", "ERR-RUN"],
)
invalid.to_excel("plot_export_tiny_llm_dim01.invalid.xlsx")
invalid

# %% Successes.

success = df[(df.EXPORT == 1) & (df.WORKS == 1)].pivot(
    index=["cache", "cache_patch", "strict", "oblivious", "rt", "export_with"],
    columns=["run_with"],
    values=["WORKS"],
)
success.to_excel("plot_export_tiny_llm_dim01.success.xlsx")
success


# %%
# If you have any error, then look at example
# :ref:`l-plot-tiny-llm-export-patched`.

doc.plot_legend("Tiny-LLM\nexport with\ndimension in {0,1}", "torch.export.export", "tomato")
