-m onnx_diagnostic validate ... validate a model id
===================================================

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

    from onnx_diagnostic._command_lines_parser import main

    main("validate -m arnir0/Tiny-LLM --run -v 1 --export onnx-dynamo -o dump_models --patch --opt ir".split())

