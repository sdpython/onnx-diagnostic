.. _l-patch-coverage:

=======================
Coverage of the Patches
=======================

Serialized Classes
==================

The following code shows the list of serialized classes in transformers.

.. runpython::
    :showcode:

    import onnx_diagnostic.torch_export_patches.onnx_export_serialization as p

    print('\n'.join(sorted(p.serialization_functions())))

Patched Classes
===============

The following script shows the list of methods patched
for transformers.

.. runpython::
    :showcode:

    import onnx_diagnostic.torch_export_patches.patches.patch_transformers as p

    for name, cls in p.__dict__.items():
        if name.startswith("patched_") and hasattr(cls, "_PATCHES_"):
            print(f"{cls._PATCHED_CLASS_.__name__}: {', '.join(cls._PATCHES_)}")

Half Automated Rewrites for Control Flows
=========================================

The following script shows the list of methods automatically rewritten
due to control flows. The same code is duplicated in many model classes.
The number of fixes if much less than the number of classes to fix.

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