.. _l-patch-diff:

============
Patches Diff
============

Patches are not always needed to export a LLM.
Most of the time, only serialization function are needed to export
a LLM with cache (``DynamicCache``, ...).
Function :func:`register_additional_serialization_functions
<onnx_diagnostic.torch_export_patches.register_additional_serialization_functions>`
is enough in many cases.

.. code-block:: python

    import torch
    from onnx_diagnostic.torch_export_patches import register_additional_serialization_functions

    with register_additional_serialization_functions(patch_transformers=True):
        ep = torch.export.export(...)

Function :func:`torch_export_patches
<onnx_diagnostic.torch_export_patches.torch_export_patches>`
helps fixing some issues for many models.

.. code-block:: python

    import torch
    from onnx_diagnostic.torch_export_patches import torch_export_patches

    with torch_export_patches(patch_transformers=True):
        ep = torch.export.export(...)

Class :class:`PatchDetails <onnx_diagnostic.torch_export_patches.patch_details.PatchDetails>`
gives an example on how to retrieve the list of involded patches for a specific model.
Those patches belong to the following list which depends on :epkg:`transformers`` and
:epkg:`pytorch` versions.

.. runpython::
    :showcode:

    import torch
    import transformers

    print(torch.__version__, transformers.__version__)

Those two versions leads to the following list of patches.

.. runpython::
    :showcode:
    :rst:

    from onnx_diagnostic.torch_export_patches.patch_details import PatchDetails
    from onnx_diagnostic.torch_export_patches import torch_export_patches

    details = PatchDetails()
    with torch_export_patches(
        patch_transformers=True,
        patch_torch=True,
        patch_diffusers=True,
        patch_details=details,
    ):
        pass
    for patch in details.patched:
        print(f"* {patch.family} - {getattr(patch.function_to_patch, '__name__', patch.function_to_patch)}")
    print()
    print()
    for patch in details.patched:
        if patch.function_to_patch == patch.patch:
            continue
        rst = patch.format_diff(format="rst")
        print()
        print()
        print(rst)
        print()
        print()
