
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

**onnx-diagnostic** is mostly to experiment ideas.

Source are `sdpython/onnx-diagnostic
<https://github.com/sdpython/onnx-diagnostic>`_.

.. toctree::
    :maxdepth: 1
    :caption: Contents

    api/index
    auto_examples/index

.. toctree::
    :maxdepth: 1
    :caption: More

    CHANGELOGS
    license

**Enlightening Examples**

* :ref:`l-plot-tiny-llm-export`

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

**Older versions**

* `0.1.0 <../v0.1.0/index.html>`_
