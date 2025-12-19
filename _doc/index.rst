========================================
onnx-diagnostic: investigate onnx models
========================================

.. image:: https://github.com/sdpython/onnx-diagnostic/actions/workflows/documentation.yml/badge.svg
    :target: https://github.com/sdpython/onnx-diagnostic/actions/workflows/documentation.yml

.. image:: https://badge.fury.io/py/onnx-diagnostic.svg
    :target: http://badge.fury.io/py/onnx-diagnostic

.. image:: https://img.shields.io/badge/license-MIT-blue.svg
    :alt: MIT License
    :target: https://opensource.org/license/MIT/

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black

.. image:: https://codecov.io/gh/sdpython/onnx-diagnostic/graph/badge.svg?token=91T5ZVIP96 
    :target: https://codecov.io/gh/sdpython/onnx-diagnostic

The main feature is about `patches <https://github.com/sdpython/onnx-diagnostic/tree/main/onnx_diagnostic/torch_export_patches>`_:
it helps exporting **pytorch models into ONNX**, mostly designed for LLMs using dynamic caches.
Sources available at `github/onnx-diagnostic <https://github.com/sdpython/onnx-diagnostic/>`_.
Patches can be enabled as follows with function
:func:`onnx_diagnostic.torch_export_patches.torch_export_patches`:

.. code-block:: python

  from onnx_diagnostic.torch_export_patches import torch_export_patches

  with torch_export_patches(patch_transformers=True) as f:
      ep = torch.export.export(model, args, kwargs=kwargs, dynamic_shapes=dynamic_shapes)
      # ...

Dynamic shapes are difficult to guess for caches, function
:func:`onnx_diagnostic.export.shape_helper.all_dynamic_shapes_from_inputs`
returns a structure defining all dimensions as dynamic.
You need then to remove those which are not dynamic in your model.

.. code-block:: python

  from onnx_diagnostic.export.shape_helper import all_dynamic_shapes_from_inputs

  dynamic_shapes = all_dynamic_shapes_from_inputs(cache)

It also implements tools to investigate, validate exported models (ExportedProgramm, ONNXProgram, ...).
:func:`onnx_diagnostic.torch_export_patches.torch_export_patches`.

.. toctree::
    :maxdepth: 1
    :caption: Contents

    patches
    api/index
    cmds/index
    auto_examples/index
    auto_recipes/index
    auto_technical/index

.. toctree::
    :maxdepth: 1
    :caption: More

    CHANGELOGS
    license

Getting started
===============

::

    git clone https://github.com/sdpython/onnx-diagnostic.git
    cd onnx-diagnostic
    pip install -e .

or

::

    pip install onnx-diagnostic

Enlightening Examples
=====================

**Where to start to export a model**

* :ref:`l-plot-export_tiny_phi2`

**Exporter Recipes**

* :ref:`l-plot-export-dim1`
* :ref:`l-plot-export-cond`
* :ref:`l-plot-export-with-dynamic`
* :ref:`l-plot-export-with-dynamic-shape`
* :ref:`l-plot-dynamic-shapes-python-int`

**Torch Export Models**

* :ref:`l-plot-nonzero`
* :ref:`l-plot-export-locale-issue`
* :ref:`l-plot-tiny-llm-export`
* :ref:`l-plot-tiny-llm-export-patched`
* :ref:`l-plot-export-hub-codellama`

**Investigate ONNX models**

* :ref:`l-plot-failing-reference-evaluator`
* :ref:`l-plot-failing-onnxruntime-evaluator`
* :ref:`l-plot-failing-model-extract`
* :ref:`l-plot-intermediate-results`

Some Usefuls Tools
==================

torch_export_patches
++++++++++++++++++++

See :func:`onnx_diagnostic.torch_export_patches.torch_export_patches`.

.. code-block:: python

  with torch_export_patches(patch_transformers=True) as f:
      ep = torch.export.export(model, args, kwargs=kwargs, dynamic_shapes=dynamic_shapes)
      # ...

torch_export_rewrite
++++++++++++++++++++

See :func:`onnx_diagnostic.torch_export_patches.torch_export_rewrite`.

.. code-block:: python

  with torch_export_rewrite(rewrite=[Model.forward]) as f:
      ep = torch.export.export(model, args, kwargs=kwargs, dynamic_shapes=dynamic_shapes)
      # ...

all_dynamic_shapes_from_inputs
++++++++++++++++++++++++++++++

See :func:`onnx_diagnostic.export.shape_helper.all_dynamic_shapes_from_inputs`.

.. code-block:: python

  from onnx_diagnostic.export.shape_helper import all_dynamic_shapes_from_inputs

  dynamic_shapes = all_dynamic_shapes_from_inputs(cache)

string_type
+++++++++++

See :func:`onnx_diagnostic.helpers.string_type`.

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

onnx_dtype_name
+++++++++++++++

See :func:`onnx_diagnostic.helpers.onnx_helper.onnx_dtype_name`.

.. code-block:: python

        import onnx
        from onnx_diagnostic.helpers.onnx_helper import onnx_dtype_name

        itype = onnx.TensorProto.BFLOAT16
        print(onnx_dtype_name(itype))
        print(onnx_dtype_name(7))

::

    >>> BFLOAT16
    >>> INT64

max_diff
++++++++

See :func:`onnx_diagnostic.helpers.max_diff`.

.. code-block:: python

    import torch
    from onnx_diagnostic.helpers import max_diff

    print(
        max_diff(
            (torch.Tensor([1, 2]), (torch.Tensor([1, 2]),)),
            (torch.Tensor([1, 2]), (torch.Tensor([1, 2]),)),
        )
    )

::

    >>> {"abs": 0.0, "rel": 0.0, "sum": 0.0, "n": 4.0, "dnan": 0.0}s

guess_dynamic_shapes
++++++++++++++++++++

See :meth:`onnx_diagnostic.export.ModelInputs.guess_dynamic_shapes`.

.. code-block:: python

    inputs = [
        (torch.randn((5, 6)), torch.randn((1, 6))),
        (torch.randn((7, 8)), torch.randn((1, 8))),
    ]
    ds = ModelInputs(model, inputs).guess_dynamic_shapes(auto="dim")
    print(ds)

::

    >>> (({0: 'dim_0I0', 1: 'dim_0I1'}, {1: 'dim_1I1'}), {})

use_dyn_not_str
+++++++++++++++

See :meth:`onnx_diagnostic.torch_export_patches.patch_inputs.use_dyn_not_str`.

The function replaces dynamic dimensions defined as strings by
``torch.export.Dim.DYNAMIC``.

Older versions
==============

* `0.8.8 <../v0.8.8/index.html>`_
* `0.8.7 <../v0.8.7/index.html>`_
* `0.7.16 <../v0.7.16/index.html>`_
* `0.6.3 <../v0.6.3/index.html>`_
* `0.5.0 <../v0.5.0/index.html>`_
* `0.4.4 <../v0.4.4/index.html>`_

The documentation was updated on:

.. runpython::
    
    import datetime
    print(datetime.datetime.now())

With the following versions:

.. runpython::

    import numpy    
    import ml_dtypes
    import sklearn
    import onnx
    import onnx_ir
    import onnxruntime
    import onnxscript
    import torch
    import transformers
    import timm

    for m in [
        numpy,
        ml_dtypes,
        sklearn,
        onnx,
        onnx_ir,
        onnxruntime,
        onnxscript,
        torch,
        transformers,
        timm,
    ]:
        print(f"{m.__name__}: {getattr(m, '__version__', 'dev')}")

    from onnx_diagnostic.ext_test_case import has_onnxruntime_training
    print(f"has_onnxruntime_training: {has_onnxruntime_training()}")

Size of the package:

.. runpython::

    import os
    import pprint
    import pandas
    from onnx_diagnostic import __file__
    from onnx_diagnostic.ext_test_case import statistics_on_folder

    df = pandas.DataFrame(statistics_on_folder(os.path.dirname(__file__), aggregation=1))
    gr = df[["dir", "ext", "lines", "chars"]].groupby(["ext", "dir"]).sum()
    print(gr)
