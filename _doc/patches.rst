.. _l-patches-explained:

=================
Patches Explained
=================

Function :func:`onnx_diagnostic.torch_export_patches.torch_export_patches`
implements four kinds of patches to make it easier to export a model, usually
coming from :epkg:`transformers`.
All patches takes place in :mod:`onnx_diagnostic.torch_export_patches`.

.. toctree::

    status/index

Four Kinds of Patches
=====================

.. code-block:: python

    with torch_export_patches(...) as f:
        ep = torch.export.export(model, args, kwargs=kwargs, dynamic_shapes=dynamic_shapes)

1. **torch fixes**:
   it disables some exceptions or improves some functions related to dynamic shapes
   until :epkg:`torch` addresses the issues
   (see `mostly exporter issues
   <https://github.com/pytorch/pytorch/issues?q=is%3Aissue%20state%3Aopen%20author%3Axadupre>`_)
2. **transformers rewriting**:
   some methods are replaced with a version :func:`torch.export.export` can understand,
   some rewriting may migrate to :epkg:`transformers`, others are applied only
   at export time because it would make the implementation less efficient
3. **cache serialization**: :func:`torch.export.export` needs to know how to
   serialize custom classes such as :class:`transformers.cache_utils.DynamicCache`
4. **control flow rewriting**: control flow (if, for) cannot be exported as is,
   there is still some work to be done to automatically process them,
   this package offers some automated rewriting, but it is far from being perfect.

All of them are triggered by :func:`onnx_diagnostic.torch_export_patches.torch_export_patches`.

.. code-block:: bash

    python -m onnx_diagnostic validate \
        -m hf-tiny-model-private/tiny-random-PLBartForConditionalGeneration \
        --run -v 1 --export onnx-dynamo -o dump_test --dtype float16 --device cuda


All patches can be disabled with ``with torch_export_patches(patch=False)``.

torch fixes
===========

Implemented in :mod:`onnx_diagnostic.torch_export_patches.patches.patch_torch` and triggered with
``with torch_export_patches(patch_sympy=True, patch_torch=True, catch_constraints=True, stop_if_static=1...)``.

It fixes some issues found while exporting model. Some of them might not be needed anymore.
It improves shape broadcasting or inserts an exception every time a dynamic dimension
becomes static (``stop_if_static=1``).

transformers rewriting
======================

Implemented in :mod:`onnx_diagnostic.torch_export_patches.patches.patch_transformers` and triggered with
``with torch_export_patches(patch_transformers=True)``.

Every patched class is prefixed with ``patched_``. It contains two class attributes.
``_PATCHES_`` contains the list of methods to replace.
``_PATCHED_CLASS_`` is the class patched by this one.

.. code-block:: python

    class patched_AttentionMaskConverter:
        """
        Patches
        ``transformers.modeling_attn_mask_utils.AttentionMaskConverter._make_causal_mask``.
        """

        # This method was fixed in 4.51 at least.
        _PATCHES_ = ["_make_causal_mask"] if not has_transformers("4.48.3") else []
        _PATCHED_CLASS_ = AttentionMaskConverter

The packages automatically parses this file to extract the patched methods.
More can be added by populating the argument ``custom_patches``:
``with torch_export_patches(patch_transformers=True, custom_patches=[...])``.
Here is the list of available patches:

.. runpython::
    :showcode:

    import onnx_diagnostic.torch_export_patches.patches.patch_transformers as p

    for name, cls in p.__dict__.items():
        if name.startswith("patched_") and hasattr(cls, "_PATCHES_"):
            print(
                f"{cls._PATCHED_CLASS_.__name__}: "
                f"{', '.join([_ for _ in cls._PATCHES_ if _ is not None])}"
            )

Cache serialization
===================

Implemented in :mod:`onnx_diagnostic.torch_export_patches.onnx_export_serialization`.
Any custom classes manipulated by a model needs to be registered through 
``torch.utils._pytree.register_pytree_node`` or with
:func:`onnx_diagnostic.torch_export_patches.onnx_export_serialization.register_class_serialization`
and triggered by ``with torch_export_patches(patch_transformers=True)``.
This function does one class, 
:func:`onnx_diagnostic.torch_export_patches.onnx_export_serialization.register_cache_serialization`
does all known classes.
It can be undone with
:func:`onnx_diagnostic.torch_export_patches.onnx_export_serialization.unregister_class_serialization`
or :func:`onnx_diagnostic.torch_export_patches.onnx_export_serialization.unregister_cache_serialization`.
Here is the list of supported caches:

.. runpython::
    :showcode:

    import onnx_diagnostic.torch_export_patches.onnx_export_serialization as p

    print(
        "\n".join(sorted(t.__name__ for t in p.serialization_functions(
            patch_transformers=True, patch_diffusers=True)))
    )

.. _l-control-flow-rewriting:

Control flow rewriting
======================

This is an attempt to automatically rewrite control flow using :mod:`ast`.
It is implemented in :mod:`onnx_diagnostic.torch_export_patches.patch_module` and
triggered ``with torch_export_patches(rewrite=<instance of torch.nn.Module>)``.
Option ``dump_rewriting=<folder>`` tells the function to dump all applied
rewritings.

The following example contains the rewriting of method
:meth:`transformers.models.bart.modeling_bart.BartEncoderLayer.forward`.
The list of known rewriting to apply are returned by function
:func:`onnx_diagnostic.torch_export_patches.patch_module_helper.code_needing_rewriting`
and applied by function :func:`onnx_diagnostic.torch_export_patches.patch_module.transform_method`.

While parsing the code, it is missing type information but this is known by
:func:`torch.export.export`. Due to that, the automation usually needs manual tuning
to filter out some tests (argument ``filter_node``) or pre/post processing
(arguments ``pre_rewriter``,  ``post_rewriter``) of function 
:func:`onnx_diagnostic.torch_export_patches.patch_module.transform_method`.

The main entry point is the context
:func:`onnx_diagnostic.torch_export_patches.torch_export_rewrite`
which rewrites and undoes the rewriting.
For example, the model :class:`transformers.BartForConditionalGeneration`
requires the following value for parameter ``rewrite``:

.. runpython::
    :showcode:

    import pprint
    from onnx_diagnostic.torch_export_patches.patch_module_helper import (
        code_needing_rewriting,
    )

    pprint.pprint(code_needing_rewriting("BartForConditionalGeneration"))    

This method has two tests. Only the first one needs to be rewritten.
The second one manipulates tuple and the automated rewritten does not handle
that because it cannot detect types. That explains why the parameter
``filter_node`` is filled. Then, the first test includes a condition relying on ``or``
which must be replaced by ``|``. That explains the parameter ``pre_rewriter``.
We finally get:

.. code-block:: diff

    --- original
    +++ rewritten
    @@ -26,7 +26,6 @@
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
    -
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(
    @@ -37,15 +36,22 @@
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
    
    -    if hidden_states.dtype == torch.float16 and (
    -        torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
    -    ):
    +    def branch_cond_then_1(hidden_states):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
    +        return hidden_states.clone()
    
    +    def branch_cond_else_1(hidden_states):
    +        return hidden_states.clone()
    +
    +    hidden_states = torch.cond(
    +        hidden_states.dtype == torch.float16
    +        and torch.isinf(hidden_states).any() | torch.isnan(hidden_states).any(),
    +        branch_cond_then_1,
    +        branch_cond_else_1,
    +        [hidden_states],
    +    )
        outputs = (hidden_states,)
    -
        if output_attentions:
    -        outputs += (attn_weights,)
    -
    +        outputs = outputs + (attn_weights,)
        return outputs

The locations where it has to be done:

.. runpython::
    :showcode:

    import pprint
    from onnx_diagnostic.torch_export_patches.patch_module_helper import (
        known_transformers_rewritings_clamp_float16,
    )

    pprint.pprint(known_transformers_rewritings_clamp_float16())

.. runpython::
    :showcode:

    import pprint
    from onnx_diagnostic.torch_export_patches.patch_module_helper import (
        _rewrite_forward_clamp_float16,
    )

    pprint.pprint(_rewrite_forward_clamp_float16())