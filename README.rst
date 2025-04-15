
.. image:: https://github.com/sdpython/onnx-diagnostic/raw/main/_doc/_static/logo.png
    :width: 120

onnx-diagnostic: investigate onnx models
========================================

.. image:: https://github.com/sdpython/onnx-diagnostic/actions/workflows/documentation.yml/badge.svg
    :target: https://github.com/sdpython/onnx-diagnostic/actions/workflows/documentation.yml

.. image:: https://badge.fury.io/py/onnx-diagnostic.svg
    :target: http://badge.fury.io/py/onnx-diagnostic

.. image:: https://img.shields.io/badge/license-MIT-blue.svg
    :alt: MIT License
    :target: https://opensource.org/license/MIT/

.. image:: https://img.shields.io/github/repo-size/sdpython/onnx-diagnostic
    :target: https://github.com/sdpython/onnx-diagnostic/
    :alt: size

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black

.. image:: https://codecov.io/gh/sdpython/onnx-diagnostic/branch/main/graph/badge.svg?token=Wb9ZGDta8J 
    :target: https://codecov.io/gh/sdpython/onnx-diagnostic

The main feature is about `patches <https://github.com/sdpython/onnx-diagnostic/tree/main/onnx_diagnostic/torch_export_patches>`_:
it helps exporting **pytorch models into ONNX**, mostly designed for LLMs using dynamic caches.

.. code-block:: python

  with bypass_export_some_errors(patch_transformers=True) as f:
      ep = torch.export.export(model, args, kwargs=kwargs, dynamic_shapes=dynamic_shapes)
      # ...

It also implements tools to investigate, validate exported models (ExportedProgramm, ONNXProgram, ...).
See `documentation of onnx-diagnostic <https://sdpython.github.io/doc/onnx-diagnostic/dev/>`_ and
`bypass_export_some_errors <https://sdpython.github.io/doc/onnx-diagnostic/dev/api/torch_export_patches/index.html#onnx_diagnostic.torch_export_patches.bypass_export_some_errors>`_.

Getting started
+++++++++++++++

::

    git clone https://github.com/sdpython/onnx-diagnostic.git
    cd onnx-diagnostic
    pip install -e .

or

::

    pip install onnx-diagnostic

Enlightening Examples
+++++++++++++++++++++

**Torch Export**

* `Use DYNAMIC or AUTO when exporting if dynamic shapes has constraints
  <https://sdpython.github.io/doc/onnx-diagnostic/dev/auto_examples/plot_export_with_dynamic_shapes_auto.html>`_
* `Find and fix an export issue due to dynamic shapes
  <https://sdpython.github.io/doc/onnx-diagnostic/dev/auto_examples/plot_export_locate_issue.html>`_
* `Export with DynamicCache and dynamic shapes
  <https://sdpython.github.io/doc/onnx-diagnostic/dev/auto_examples/plot_export_with_dynamic_cache.html>`_
* `Steel method forward to guess the dynamic shapes (with Tiny-LLM)
  <https://sdpython.github.io/doc/onnx-diagnostic/dev/auto_examples/plot_export_tiny_llm.html>`_
* `Export Tiny-LLM with patches
  <https://sdpython.github.io/doc/onnx-diagnostic/dev/auto_examples/plot_export_tiny_llm_patched.html>`_

**Investigate ONNX models**

* `Find where a model is failing by running submodels
  <https://sdpython.github.io/doc/onnx-diagnostic/dev/auto_examples/plot_failing_model_extract.html>`_
* `Intermediate results with (ONNX) ReferenceEvaluator
  <https://sdpython.github.io/doc/onnx-diagnostic/dev/auto_examples/plot_failing_reference_evaluator.html>`_
* `Intermediate results with onnxruntime
  <https://sdpython.github.io/doc/onnx-diagnostic/dev/auto_examples/plot_failing_onnxruntime_evaluator.html>`_

Snapshot of usefuls tools
+++++++++++++++++++++++++

**string_type**

.. code-block:: python

    import torch
    from onnx_diagnostic.helpers import string_type

    inputs = (
        torch.rand((3, 4), dtype=torch.float16),
        [
            torch.rand((5, 6), dtype=torch.float16),
            torch.rand((5, 6, 7), dtype=torch.float16),
        ]
    )

    # with shapes
    print(string_type(inputs, with_shape=True))

::

    >>> (T10s3x4,#2[T10s5x6,T10s5x6x7])

**onnx_dtype_name**

.. code-block:: python

        import onnx
        from onnx_diagnostic.helpers.onnx_helper import onnx_dtype_name

        itype = onnx.TensorProto.BFLOAT16
        print(onnx_dtype_name(itype))
        print(onnx_dtype_name(7))

::

    >>> BFLOAT16
    >>> INT64

**max_diff**

Returns the maximum discrancies across nested containers containing tensors.
