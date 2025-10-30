"""
JSON returns list when the original dynamic shapes are list or tuple
====================================================================

Dynamic shapes given to :func:`torch.export.export` must follow the
same semantic. What if we confuse tuple and list when defining the dynamic shapes,
how to restore the expected type assuming we know the inputs?
Not often useful but maybe we will learn more about
:epkg:`optree`.

Dynamic Shapes After JSON
+++++++++++++++++++++++++

JSON format does not make the difference between a list and a tuple.
So after serializing to json and restoring, both of them become lists.
"""

import json
import pprint
import torch
from onnx_diagnostic import doc
from onnx_diagnostic.helpers import string_type
from onnx_diagnostic.helpers.cache_helper import make_dynamic_cache
from onnx_diagnostic.export.shape_helper import all_dynamic_shapes_from_inputs
from onnx_diagnostic.torch_export_patches import register_additional_serialization_functions

bsize, nheads, slen, dim = 2, 1, 30, 96

inputs = dict(
    input_mask_position=(
        torch.randint(15, size=(2, 3), dtype=torch.int64),
        torch.randint(1, size=(2, 33), dtype=torch.int64),
        torch.arange(3, dtype=torch.int64),
    ),
    past_key_values=make_dynamic_cache(
        [(torch.randn(bsize, nheads, slen, dim), torch.randn(bsize, nheads, slen, dim))]
    ),
)

print(string_type(inputs, with_shape=True))

# %%
# Function :func:`onnx_diagnostic.export.shape_helper.all_dynamic_shapes_from_inputs`
# produces the corresponding dynamic shapes assuming they are all dynamic.
# ``register_additional_serialization_functions(patch_transformers=True)`` registers
# function letting pytorch know how to serialize, deserialize the class DynamicCache.

with register_additional_serialization_functions(patch_transformers=True):
    ds = all_dynamic_shapes_from_inputs(inputs)
pprint.pprint(ds)

# %%
# Converted into JSON.

json_str = json.dumps(ds, indent=2, ensure_ascii=False)
print(json_str)

# %%
# Restoration.
ds2 = json.loads(json_str)
pprint.pprint(ds2)

# %%
# tuple are replaced by list.

# The trick to restore tuple when expected
# ++++++++++++++++++++++++++++++++++++++++


def flatten_unflatten_like_dynamic_shapes(obj):
    if isinstance(obj, torch.Tensor):
        return obj
    flat, spec = torch.utils._pytree.tree_flatten(obj)
    start = 0
    end = 0
    subtrees = []
    for subspec in spec.children_specs:
        end += subspec.num_leaves
        value = subspec.unflatten(flat[start:end])
        value = flatten_unflatten_like_dynamic_shapes(value)
        subtrees.append(value)
        start = end
    if spec.type is dict:
        # This a dictionary.
        return dict(zip(spec.context, subtrees))
    if spec.type is tuple:
        return tuple(subtrees)
    if spec.type is list:
        return list(subtrees)
    if spec.context:
        # This is a custom class with attributes.
        # It is returned as a list.
        return list(subtrees)
    raise ValueError(
        f"Unable to interpret spec type {spec.type} "
        f"(type is {type(spec.type)}, context is {spec.context}), "
        f"obj type is {type(obj)}."
    )


def _align(inputs, ds):
    if isinstance(inputs, torch.Tensor):
        return ds
    if isinstance(inputs, tuple):
        return tuple(_align(o, d) for o, d in zip(inputs, ds))
    if isinstance(inputs, list):
        return [_align(o, d) for o, d in zip(inputs, ds)]
    if isinstance(inputs, dict):
        return {k: _align(inputs[k], d) for k, d in ds.items()}
    raise TypeError(f"Unexpected types inputs is {type(inputs)}, ds is {type(ds)}")


def fix_dynamic_shapes(inputs, dynamic_shapes):
    flat_unflat_inputs = flatten_unflatten_like_dynamic_shapes(inputs)
    return _align(flat_unflat_inputs, dynamic_shapes)


with register_additional_serialization_functions(patch_transformers=True):
    fixed_ds = fix_dynamic_shapes(inputs, ds2)
pprint.pprint(fixed_ds)

# %%
# The code changed tuple into list as expected.
assert isinstance(ds2["input_mask_position"], list)
assert isinstance(fixed_ds["input_mask_position"], tuple)


# %%

doc.plot_legend("dynamic shapes\nto json\nfrom json", "torch.export.export", "green")
