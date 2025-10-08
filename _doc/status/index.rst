===============
Exporter Status
===============

Following sections tries to capture what patches are put in place to export,
what works and what does not with :func:`torch.export.export`.

.. toctree::
    :maxdepth: 1

    exported_program_dynamic
    exporter_dynamic
    patches_coverage

Examples checking about dynamic dimensions:

* :ref:`l-plot-tiny-llm-export-dim01`
* :ref:`l-plot-tiny-llm-export-dim01-onnx`
* :ref:`l-plot-tiny-llm-export-dim01-onnx-custom`

Some PRs in :epkg:`transformers` to keep in mind when it comes to export
a model using a cache or a custom class:

* `Completely rewrite the masking logic for all attentions <https://github.com/huggingface/transformers/pull/37866>`_
* `Fix bugs in DynamicCache <https://github.com/huggingface/transformers/pull/37880>`_
* `Fixes DynamicCache export issues due to control flow and inplace modifications <https://github.com/huggingface/transformers/pull/36652>`_
* `Support tracable dynamicKVcache <https://github.com/huggingface/transformers/pull/36311>`_
