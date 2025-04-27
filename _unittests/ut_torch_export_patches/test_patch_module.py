import ast
import unittest
import torch
from onnx_diagnostic.ext_test_case import ExtTestCase
from onnx_diagnostic.torch_export_patches.patch_module import transform_method


class TestPatchModule(ExtTestCase):
    def test_rewrite_forward(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                if x.sum() > 0:
                    return x + y
                else:
                    return torch.abs(x) + y

        x, y = torch.rand((3, 4)), torch.rand((3, 4))
        Model()(x, y)
        tree, me = transform_method(Model.forward)

        print("-------------")
        print(ast.dump(tree.body[0], indent=4))
        print("-------------")
        code = ast.unparse(tree)
        print(code)
        print("-------------")


if __name__ == "__main__":
    unittest.main(verbosity=2)
