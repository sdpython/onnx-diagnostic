-m onnx_diagnostic partition ... move layer nodes in local functions
====================================================================

The command line leverages the metadata added by the exporter.
Every node is tagged with information indicating which part of the model
it comes from. In particular the eky `namespace`:

::

    transformers.models.llama.modeling_llama.LlamaForCausalLM/model:
    transformers.models.llama.modeling_llama.LlamaModel/model.layers.0:
    transformers.models.llama.modeling_llama.LlamaDecoderLayer/model.layers.0.self_attn:
    transformers.models.llama.modeling_llama.LlamaAttention/unsqueeze_15:
    aten.unsqueeze.default

Description
+++++++++++

See :func:`onnx_diagnostic.helpers.onnx_helper.make_model_with_local_functions`.

.. runpython::

    from onnx_diagnostic._command_lines_parser import get_parser_partition

    get_parser_partition().print_help()

Example
+++++++

.. code-block:: bash

    python -m onnx_diagnostic partition arnir0_Tiny-LLM-onnx-dynamo-ir-f16-cuda-op18.onnx partition.onnx -r ".*[.]layers[.][0-9]+$" -v 1

This produces the following output:

::

    -- load 'arnir0_Tiny-LLM-onnx-dynamo-ir-f16-cuda-op18.onnx'
    -- partition
    [make_model_with_local_functions] matched 1 partitions
    [make_model_with_local_functions] move 89 nodes in partition 'transformers_models_llama_modeling_llama_LlamaModel/model_layers_0'
    -- save into 'partition.onnx'
    -- done

The partitioned model includes the following node:

.. image:: _img_partition.png
