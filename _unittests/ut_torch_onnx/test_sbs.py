import unittest
from onnx_diagnostic.ext_test_case import (
    ExtTestCase,
    hide_stdout,
    ignore_warnings,
    ignore_errors,
)
from onnx_diagnostic.reference import ExtendedReferenceEvaluator
from onnx_diagnostic.torch_export_patches.patch_inputs import use_dyn_not_str
from onnx_diagnostic.torch_onnx.sbs import run_aligned

try:
    from experimental_experiment.torch_interpreter import to_onnx
except ImportError:
    to_onnx = None


class TestSideBySide(ExtTestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        cls.torch = torch

    @hide_stdout()
    @unittest.skipIf(to_onnx is None, "to_onnx not installed")
    @ignore_errors(OSError)  # connectivity issues
    @ignore_warnings((UserWarning,))
    def test_ep_onnx_sync_exp(self):
        class Model(self.torch.nn.Module):
            def forward(self, x):
                ry = x.abs()
                rz = ry.exp()
                rw = rz + 1
                ru = rw.log() + rw
                return ru

        x = self.torch.randn((5, 4))
        Model()(x)
        ep = self.torch.export.export(
            Model(), (x,), dynamic_shapes=({0: self.torch.export.Dim("batch")},)
        )
        onx = to_onnx(ep)
        results = list(
            run_aligned(
                ep,
                onx,
                args=(x,),
                run_cls=ExtendedReferenceEvaluator,
                atol=1e-5,
                rtol=1e-5,
                verbose=1,
            ),
        )
        self.assertEqual(len(results), 5)

    @hide_stdout()
    @ignore_warnings((DeprecationWarning, FutureWarning, UserWarning))
    def test_ep_onnx_sync_a(self):
        class Model(self.torch.nn.Module):
            def forward(self, x):
                ry = x.abs()
                rz = ry.exp()
                rw = rz + 1
                ru = rw.log() + rw
                return ru

        x = self.torch.randn((5, 4))
        Model()(x)
        ep = self.torch.export.export(
            Model(), (x,), dynamic_shapes=({0: self.torch.export.Dim("batch")},)
        )
        epo = self.torch.onnx.export(
            ep, (x,), dynamic_shapes=({0: self.torch.export.Dim("batch")},)
        )
        onx = epo.model_proto
        results = list(
            run_aligned(
                ep,
                onx,
                args=(x,),
                run_cls=ExtendedReferenceEvaluator,
                atol=1e-5,
                rtol=1e-5,
                verbose=1,
            ),
        )
        self.assertEqual(len(results), 4)

    @hide_stdout()
    @ignore_warnings((DeprecationWarning, FutureWarning, UserWarning))
    def test_sbs_dict(self):
        class Model(self.torch.nn.Module):
            def forward(self, x):
                ry = x.abs()
                rz = ry.exp()
                rw = rz + 1
                ru = rw.log() + rw
                return ru

        inputs = dict(x=self.torch.randn((5, 4)))
        ds = dict(x={0: "batch"})
        Model()(**inputs)
        ep = self.torch.export.export(
            Model(), (), kwargs=inputs, dynamic_shapes=use_dyn_not_str(ds)
        )
        epo = self.torch.onnx.export(Model(), (), kwargs=inputs, dynamic_shapes=ds)
        onx = epo.model_proto
        results = list(
            run_aligned(
                ep,
                onx,
                kwargs=inputs,
                run_cls=ExtendedReferenceEvaluator,
                atol=1e-5,
                rtol=1e-5,
                verbose=1,
            ),
        )
        self.assertEqual(len(results), 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
