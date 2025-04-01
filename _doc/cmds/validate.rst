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

.. code-block::

    python -m onnx_diagnostic validate

.. runpython::

    from onnx_diagnostic._command_lines_parser import main

    main(["validate"])


Get the default inputs for a specific task
++++++++++++++++++++++++++++++++++++++++++

.. code-block::

    python -m onnx_diagnostic validate -t text-generation

.. runpython::

    from onnx_diagnostic._command_lines_parser import main

    main("validate -t text-generation".split())

Validate a model
++++++++++++++++

.. code-block::

    python -m onnx_diagnostic validate -m arnir0/Tiny-LLM --run -v 1

.. runpython::

    from onnx_diagnostic._command_lines_parser import main

    main("validate -m arnir0/Tiny-LLM --run -v 1".split())
