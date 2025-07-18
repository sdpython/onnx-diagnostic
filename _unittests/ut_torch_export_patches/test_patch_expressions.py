import unittest
import torch
from onnx_diagnostic.ext_test_case import ExtTestCase
from onnx_diagnostic.torch_export_patches.patch_expressions import (
    _iterate_patched_expressions,
    register_patched_expressions,
    patched_selector,
    patched_float_arange,
)
from onnx_diagnostic.helpers.torch_helper import fake_torchdynamo_exporting


class TestOnnxExportErrors(ExtTestCase):
    @classmethod
    def setUp(cls):
        register_patched_expressions()

    def test_patched_expressions(self):
        res = list(_iterate_patched_expressions())
        names = {_[0] for _ in res}
        self.assertIn("float_arange", names)

    def test_float_arange(self):
        _T = torch.tensor
        res = torch.arange(4, 6, 0.234)
        got = torch.arange(4, 6, 0.234, dtype=torch.float32, device=torch.device("cpu"))
        self.assertEqualArray(res, got)
        got = torch.ops.patched.float_arange(_T(4.0), _T(6.0), _T(0.234))
        self.assertEqualArray(res, got, atol=1e-5)
        got = patched_selector(
            (lambda a, b, c: torch.arange(a.item(), b.item(), c.item())),
            torch.ops.patched.float_arange,
        )(_T(4.0), _T(6.0), _T(0.234))
        self.assertEqualArray(res, got, atol=1e-5)
        got = patched_float_arange(_T(4.0), _T(6.0), _T(0.234))
        self.assertEqualArray(res, got, atol=1e-5)
        with fake_torchdynamo_exporting():
            got = patched_selector(None, torch.ops.patched.float_arange)(
                _T(4.0), _T(6.0), _T(0.234)
            )
            self.assertEqualArray(res, got, atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
