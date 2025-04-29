
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

.. image:: https://codecov.io/gh/sdpython/onnx-diagnostic/branch/main/graph/badge.svg?token=Wb9ZGDta8J 
    :target: https://codecov.io/gh/sdpython/onnx-diagnostic

The main feature is about `patches <https://github.com/sdpython/onnx-diagnostic/tree/main/onnx_diagnostic/torch_export_patches>`_:
it helps exporting **pytorch models into ONNX**, mostly designed for LLMs using dynamic caches.
Sources available at `github/onnx-diagnostic <https://github.com/sdpython/onnx-diagnostic/>`_.

.. code-block:: python

  with torch_export_patches(patch_transformers=True) as f:
      ep = torch.export.export(model, args, kwargs=kwargs, dynamic_shapes=dynamic_shapes)
      # ...

It also implements tools to investigate, validate exported models (ExportedProgramm, ONNXProgram, ...).
:func:`onnx_diagnostic.torch_export_patches.torch_export_patches`.

.. toctree::
    :maxdepth: 1
    :caption: Contents

    api/index
    cmds/index
    auto_examples/index
    auto_recipes/index

.. toctree::
    :maxdepth: 1
    :caption: More

    CHANGELOGS
    license

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

* :ref:`l-plot-export-cond`
* :ref:`l-plot-sxport-with-auto`
* :ref:`l-plot-export-with-dynamic-shape`
* :ref:`l-plot-export-locale-issue`
* :ref:`l-plot-tiny-llm-export`
* :ref:`l-plot-tiny-llm-export-patched`
* :ref:`l-plot-export-hub-codellama`

**Investigate ONNX models**

* :ref:`l-plot-failing-reference-evaluator`
* :ref:`l-plot-failing-onnxruntime-evaluator`
* :ref:`l-plot-failing-model-extract`

**Some Usefuls Tools**

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

.. code-block:: python

        import onnx
        from onnx_diagnostic.helpers import onnx_dtype_name

        itype = onnx.TensorProto.BFLOAT16
        print(onnx_dtype_name(itype))
        print(onnx_dtype_name(7))

::

    >>> BFLOAT16
    >>> INT64

:func:`onnx_diagnostic.helpers.max_diff`, ...

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

Older versions
++++++++++++++

* `0.4.4 <../v0.4.4/index.html>`_
* `0.4.3 <../v0.4.3/index.html>`_
* `0.4.2 <../v0.4.2/index.html>`_
* `0.4.1 <../v0.4.1/index.html>`_
* `0.4.0 <../v0.4.0/index.html>`_
* `0.3.0 <../v0.3.0/index.html>`_
* `0.2.2 <../v0.2.2/index.html>`_
* `0.2.1 <../v0.2.1/index.html>`_
* `0.2.0 <../v0.2.0/index.html>`_
* `0.1.0 <../v0.1.0/index.html>`_
