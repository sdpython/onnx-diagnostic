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

This example is based on python but it produces the same output
than the command line.

.. runpython::
    :showcode:

    import torch
    from onnx_diagnostic.export.api import to_onnx
    from onnx_diagnostic.torch_onnx.compare import ObsComparePair, ObsCompare


    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 16, 5)
            self.fc1 = torch.nn.Linear(144, 64)
            self.fc2 = torch.nn.Linear(64, 128)
            self.fc3 = torch.nn.Linear(128, 10)

        def forward(self, x):
            x = torch.nn.functional.max_pool2d(torch.nn.functional.relu(self.conv1(x)), (4, 4))
            # x = F.max_pool2d(F.relu(self.conv2(x)), 2)
            x = torch.flatten(x, 1)
            x = torch.nn.functional.relu(self.fc1(x))
            x = torch.nn.functional.relu(self.fc2(x))
            y = self.fc3(x)
            return y


    model = Model()
    x = torch.randn((2, 3, 16, 17), dtype=torch.float32)
    dynamic_shapes = ({0: "batch", 3: "dim"},)
    onnx_optimized = to_onnx(
        model, (x,), dynamic_shapes=dynamic_shapes, exporter="custom", optimize=True
    ).model_proto
    onnx_not_optimized = to_onnx(
        model, (x,), dynamic_shapes=dynamic_shapes, exporter="custom", optimize=False
    ).model_proto
    seq1 = ObsCompare.obs_sequence_from_model(onnx_not_optimized)
    seq2 = ObsCompare.obs_sequence_from_model(onnx_optimized)
    _dist, _path, pair_cmp = ObsComparePair.distance_sequence(seq1, seq2)
    text = ObsComparePair.to_str(pair_cmp)
    print(text)
