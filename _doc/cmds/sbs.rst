-m onnx_diagnostic sbs ... runs a side-by-side torch/onnx
=========================================================

Description
+++++++++++

It compares the intermediate results between an exported program saved with
:func:`torch.export.save` and an exported model on saved inputs
with :func:`torch.save`. It assumes intermediate results share the same
names.

.. runpython::

    from onnx_diagnostic._command_lines_parser import get_parser_sbs

    get_parser_sbs().print_help()

CPU, CUDA
+++++++++

Inputs are saved :func:`torch.save`. The execution will run on CUDA
if the device of the inputs is CUDA, same goes on CPU.

Example
+++++++

.. code-block::

    python -m onnx_diagnostic sbs \
        -i qwen_2_5_vl_instruct_visual.inputs.pt \
        --ep test_imagetext2text_qwen_2_5_vl_instruct_visual.cuda.float16.custom.graph.ep.pt2 \
        -m test_imagetext2text_qwen_2_5_vl_instruct_visual.cuda.float16.custom.onnx \
        -o results.dynamo.float16.xlsx \
        -v 1 --atol=0.1 --rtol=1 \
        --replay-names conv3d,rsqrt,to_4,mul_48,linear,linear_2,linear_84,linear_89,mul_172,linear_156,linear_159 \
        -2 --reset conv3d

A snippet of the table it produces:

::

    ep_name         onnx_name       ep_target               onnx_op_type            onnx_id_output   ep_shape_type      onnx_shape_type    err_abs 
    transpose_18    transpose_18    aten.transpose.int      Transpose                           0    GT10s16x1292x80    GT10s16x1292x80    0.0083 
    unsqueeze_50    unsqueeze_50    aten.unsqueeze.default  Unsqueeze                           0    GT10s1x16x1292x80  GT10s1x16x1292x80  0.0083 
    eq_20           eq_20           aten.eq.Scalar          Equal                               0    GT9s1292x1292      GT9s1292x1292      0   
    unsqueeze_56    unsqueeze_56    aten.unsqueeze.default  Unsqueeze                           0    GT9s1x1x1292x1292  GT9s1x1x1292x1292  0   
    slice_29        slice_29        aten.slice.Tensor       Slice                               0    GT9s1x1x1292x1292  GT9s1x1x1292x1292  0   
    transpose_19    transpose_19    aten.transpose.int      Transpose                           0    GT10s1x1292x16x80  GT10s1x1292x16x80  0.0071 
    reshape_20      reshape_20      aten.reshape.default    Reshape                             0    GT10s1292x1280     GT10s1292x1280     0.0071 
    linear_21       linear_21       aten.linear.default     Gemm                                0    GT10s1292x1280     GT10s1292x1280     0.0015 
    mul_54          mul_54          aten.mul.Tensor         SkipSimplifiedLayerNormalization    0    GT10s1292x1280     GT10s1292x1280     0.0098 
    add_32          add_32          aten.add.Tensor         SkipSimplifiedLayerNormalization    3    GT10s1292x1280     GT10s1292x1280     0.0313 
    linear_22       linear_22       aten.linear.default     Gemm                                0    GT10s1292x3420     GT10s1292x3420     0.0078 
    silu_4          silu_4          aten.silu.default       QuickGelu                           0    GT10s1292x3420     GT10s1292x3420     0.0059 

The available column are described by
:class:`RunAlignedRecord <onnx_diagnostic.torch_onnx.sbs_dataclasses.RunAlignedRecord>`.
It is possible to dump pieces of the model to study some particular input
with :class:`ReplayConfiguration <onnx_diagnostic.torch_onnx.sbs_dataclasses.ReplayConfiguration>`.
