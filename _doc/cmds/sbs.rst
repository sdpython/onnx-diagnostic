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
