
.. image:: https://github.com/sdpython/onnx-diagnostic/raw/main/_doc/_static/logo.png
    :width: 120

onnx-diagnostic: investigate onnx models
========================================

.. image:: https://github.com/sdpython/onnx-diagnostic/actions/workflows/documentation.yml/badge.svg
    :target: https://github.com/sdpython/onnx-diagnostic/actions/workflows/documentation.yml

.. image:: https://badge.fury.io/py/onnx-diagnostic.svg
    :target: http://badge.fury.io/py/onnx-diagnostic

.. image:: http://img.shields.io/github/issues/sdpython/onnx-diagnostic.png
    :alt: GitHub Issues
    :target: https://github.com/sdpython/onnx-diagnostic/issues

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

Getting started
+++++++++++++++

::

    git clone https://github.com/sdpython/onnx-diagnostic.git
    cd onnx-diagnostic
    pip install -e .

or

::

    pip install onnx-diagnostic

**Enlightening Examples**

* `Use DYNAMIC or AUTO when dynamic shapes has constraints
  <https://sdpython.github.io/doc/onnx-diagnostic/dev/auto_examples/plot_export_with_dynamic_shapes_auto.html>`_
* `Steel method forward to guess the dynamic shapes
  <https://sdpython.github.io/doc/onnx-diagnostic/dev/auto_examples/plot_export_tiny_llm.html>`_
* `Running ReferenceEvaluator on a failing model
  <https://sdpython.github.io/doc/onnx-diagnostic/dev/auto_examples/plot_failing_reference_evaluator.html>`_
* `Find where a model is failing by running submodels
  <https://sdpython.github.io/doc/onnx-diagnostic/dev/auto_examples/plot_failing_model_extract.html>`_

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
        from onnx_diagnostic.helpers import onnx_dtype_name

        itype = onnx.TensorProto.BFLOAT16
        print(onnx_dtype_name(itype))
        print(onnx_dtype_name(7))

::

    >>> BFLOAT16
    >>> INT64

**max_diff**

Returns the maximum discrancies accross nested containers containing tensors.

Documentation
+++++++++++++

See `onnx-diagnostic <https://sdpython.github.io/doc/onnx-diagnostic/dev/>`_.
