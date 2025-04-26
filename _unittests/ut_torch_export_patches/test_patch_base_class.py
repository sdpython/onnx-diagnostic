import unittest
import torch
from onnx_diagnostic.ext_test_case import ExtTestCase, hide_stdout
from onnx_diagnostic.torch_export_patches import torch_export_patches


class TestPatchBaseClass(ExtTestCase):
    def test_check_that_trick_can_work_in_python(self):
        class zero:
            def ret(self, a):
                return a - 100

            def ok(self):
                return self.ret(3)

        class A(zero):
            def ret(self, a):
                return a + 1

        class B:
            def ret(self, a):
                return a + 10

        z = zero()
        self.assertEqual(z.ret(4), -96)
        self.assertEqual(z.ok(), -97)
        a = A()
        self.assertEqual(a.ret(4), 5)
        self.assertEqual(a.ok(), 4)
        b = B()
        self.assertEqual(b.ret(4), 14)
        self.assertFalse(hasattr(b, "ok"))
        self.assertFalse(hasattr(B, "ok"))

        self.assertEqual(A.__bases__, (zero,))
        A.__bases__ = (zero, B)
        self.assertEqual(a.ret(4), 5)
        self.assertEqual(a.ok(), 4)
        aa = A()
        self.assertEqual(aa.ret(4), 5)
        self.assertEqual(aa.ok(), 4)

        A.__bases__ = (B, zero)
        self.assertEqual(a.ret(4), 5)
        self.assertEqual(a.ok(), 4)
        aa = A()
        self.assertEqual(aa.ret(4), 5)
        self.assertEqual(aa.ok(), 4)

        A.__bases__ = (zero,)
        A.ret = B.ret
        self.assertEqual(aa.ret(4), 14)
        self.assertEqual(aa.ok(), 13)
        self.assertEqual(a.ret(4), 14)
        self.assertEqual(a.ok(), 13)

    @hide_stdout()
    def test_custom_patches(self):
        class Model(torch.nn.Module):
            def m1(self, x):
                return x * x

            def forward(self, x):
                return self.m1(x)

        class patched_Model:
            _PATCHED_CLASS_ = Model
            _PATCHES_ = ["m1"]

            def m1(self, x):
                return x**3

        model = Model()
        x = torch.arange(4)
        self.assertEqualArray(x * x, model(x))
        with torch_export_patches(custom_patches=[patched_Model], verbose=10):
            self.assertEqualArray(x**3, model(x))


if __name__ == "__main__":
    unittest.main(verbosity=2)
