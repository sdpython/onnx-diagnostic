import unittest
import torch
from onnx_diagnostic.ext_test_case import ExtTestCase, hide_stdout
from onnx_diagnostic.export.api import to_onnx


class TestValidate(ExtTestCase):
    @hide_stdout()
    def test_to_onnx(self):
        class Model(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        x = torch.randn((5, 6))
        y = torch.randn((1, 6))
        ds = ({0: "a", 1: "b"}, {1: "b"})
        to_onnx(
            Model(),
            (x, y),
            dynamic_shapes=ds,
            exporter="custom",
            filename=self.get_dump_file("custom.onnx"),
        )
        to_onnx(
            Model(),
            (x, y),
            dynamic_shapes=ds,
            exporter="onnx-dynamo",
            filename=self.get_dump_file("onnx-dynamo.onnx"),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
