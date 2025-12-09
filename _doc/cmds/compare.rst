-m onnx_diagnostic compare ... compares two models
==================================================

Description
+++++++++++

The command lines compares two models assuming they represent
the same models and most parts of both are the same.
Different options were used to export or an optimization
was different. This highlights the differences.

.. runpython::

    from onnx_diagnostic._command_lines_parser import get_parser_compare

    get_parser_compare().print_help()

Example
+++++++

.. code-block:: bash

    python -m onnx_diagnostic compare <mode1.onnx> <mode1.onnx>

.. code-block:: text

    -- loading 'two_nodes.onnx'
    -- loading 'two_nodes.onnx'
    -- starts comparison
    -- done with distance 0
    0000 INITIA FLOAT    ?                                  encoder.encoders.0.layer_norm_att.w | INITIA FLOAT    ?                                  encoder.encoders.0.layer_norm_att.w
    0001 INITIA FLOAT    ?                                  encoder.encoders.0.layer_norm_att.b | INITIA FLOAT    ?                                  encoder.encoders.0.layer_norm_att.b
    0002 INPUT  FLOAT    s0x(((s1 - 1)//8))                 linear                              | INPUT  FLOAT    s0x(((s1 - 1)//8))                 linear                             
    0003 INPUT  FLOAT    s0x(((s1 - 1)//8))                 mul_178                             | INPUT  FLOAT    s0x(((s1 - 1)//8))                 mul_178                            
    0004 NODE   FLOAT    s0x(((s1 - 1)//8)) Add             add_256                             | NODE   FLOAT    s0x(((s1 - 1)//8)) Add             add_256                            
    0005 NODE   FLOAT    s0x(((s1 - 1)//8)) LayerNormalizat layer_norm_1                        | NODE   FLOAT    s0x(((s1 - 1)//8)) LayerNormalizat layer_norm_1                       
    0006 OUTPUT FLOAT    s0x(((s1 - 1)//8))                 layer_norm_1                        | OUTPUT FLOAT    s0x(((s1 - 1)//8))                 layer_norm_1                       
    0007 OUTPUT FLOAT    s0x(((s1 - 1)//8))                 add_256                             | OUTPUT FLOAT    s0x(((s1 - 1)//8))                 add_256                            
