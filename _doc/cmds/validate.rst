.. _l-cmd-validate:

-m onnx_diagnostic validate ... validate a model id
===================================================

The command line is a wrapper around function
:func:`onnx_diagnostic.torch_models.validate.validate_model`.

Description
+++++++++++

The command lines validate a model id
available on :epkg:`HuggingFace` but not only.
It creates dummy inputs, runs the models on them,
exports the model, measures the discrepancies...

.. runpython::

    from onnx_diagnostic._command_lines_parser import get_parser_validate

    get_parser_validate().print_help()

Get the list of supported tasks
+++++++++++++++++++++++++++++++

The task are the same defined by :epkg:`HuggingFace`.
The tool only supports a subset of them.

.. code-block::

    python -m onnx_diagnostic validate

.. runpython::

    from onnx_diagnostic._command_lines_parser import main

    main(["validate"])


Get the default inputs for a specific task
++++++++++++++++++++++++++++++++++++++++++

This returns the dummy inputs for a specific task.
There may be too many inputs. Only those the forward method
defines are kept.

.. code-block::

    python -m onnx_diagnostic validate -t text-generation

.. runpython::

    from onnx_diagnostic._command_lines_parser import main

    main("validate -t text-generation".split())

Validate dummy inputs for a model
+++++++++++++++++++++++++++++++++

The dummy inputs may not work for this model and this task.
The following command line checks that. It is no use to export
if this fails.

.. code-block::

    python -m onnx_diagnostic validate -m arnir0/Tiny-LLM --run -v 1

.. runpython::

    from onnx_diagnostic._command_lines_parser import main

    main("validate -m arnir0/Tiny-LLM --run -v 1".split())

Validate and export a model
+++++++++++++++++++++++++++

Exports a model given the task. Checks for discrepancies as well.
The latency given are just for one run. It tells how long the benchmark
runs but it is far from the latency measure we can get by running multiple times
the same model.


.. code-block::

    python -m onnx_diagnostic validate -m arnir0/Tiny-LLM --run -v 1 --export export-nostrict -o dump_models --patch

.. runpython::

    from onnx_diagnostic._command_lines_parser import main

    main("validate -m arnir0/Tiny-LLM --run -v 1 --export export-nostrict -o dump_models --patch".split())

Validate ONNX discrepancies
+++++++++++++++++++++++++++

Let's export with ONNX this time and checks for discrepancies.

.. code-block::

    python -m onnx_diagnostic validate -m arnir0/Tiny-LLM --run -v 1 --export onnx-dynamo -o dump_models --patch --opt ir

.. runpython::
    :process:

    from onnx_diagnostic._command_lines_parser import main

    main("validate -m arnir0/Tiny-LLM --run -v 1 --export onnx-dynamo -o dump_models --patch --opt ir".split())

Run onnxruntime fusions
+++++++++++++++++++++++

This option runs `transformers optimizations <https://onnxruntime.ai/docs/performance/transformers-optimization.html>`_ 
implemented in :epkg:`onnxruntime`. The list of supported ``model_type`` can be found in the documentation
of function :func:`onnx_diagnostic.torch_models.validate.run_ort_fusion`.

.. code-block::

    python -m onnx_diagnostic validate -m arnir0/Tiny-LLM --run -v 1 --export onnx-dynamo -o dump_models --patch --opt ir --ortfusiontype ALL

.. runpython::
    :process:

    from onnx_diagnostic._command_lines_parser import main

    main("validate -m arnir0/Tiny-LLM --run -v 1 --export onnx-dynamo -o dump_models --patch --opt ir --ortfusiontype ALL".split())

SDPA or Eager implementation or Use a StaticCache
+++++++++++++++++++++++++++++++++++++++++++++++++

Add ``--mop cache_implementation=static --iop cls_cache=StaticCache`` to use a StaticCache instead of a DynamicCache (default).
Add ``--mop attn_implementation=eager`` to explicitly select eager implementation for attention.

.. code-block:: bash

    python -m onnx_diagnostic validate \
                -m google/gemma-2b \
                --run \
                -v 1 \
                --export custom \
                -o dump_test \
                --dtype float16 \
                --device cpu \
                --patch \
                --no-quiet \
                --opt default \
                --rewrite \
                --mop attn_implementation=eager \
                --mop cache_implementation=static \
                --iop cls_cache=StaticCache

Frequent examples used to test
++++++++++++++++++++++++++++++

.. code-block:: bash

    python -m onnx_diagnostic validate -m arnir0/Tiny-LLM --run -v 1 --device cuda --dtype float16 -o dump_models --patch --opt default+onnxruntime --export custom

About the exporter 'custom'
+++++++++++++++++++++++++++

It used to investigate issues or scenarios. It is usually very strict
and fails every time it falls in one unexpected situation.
It call :func:`experimental_experiment.torch_interpreter.to_onnx`.
Some useful environment variables to set before running the command line.

* ``DROPPATTERN=<pattern1,patterns2,...>``: do not apply those patterns when optimizing a model
* ``DUMPPATTERNS=<folder>``: dumps all matched and applied nodes when a pattern is applied
* ``PATTERN=<pattern1,pattern2,...>``: increase verbosity for specific patterns to understand why one pattern was not applied
