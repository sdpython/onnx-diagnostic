import unittest
from onnx_diagnostic.ext_test_case import (
    ExtTestCase,
    hide_stdout,
    ignore_warnings,
    ignore_errors,
)
from onnx_diagnostic.reference import ExtendedReferenceEvaluator
from onnx_diagnostic.torch_onnx.sbs import run_aligned

try:
    from experimental_experiment.torch_interpreter import to_onnx
except ImportError:
    to_onnx = None


class TestSideBySide(ExtTestCase):

    @hide_stdout()
    @unittest.skipIf(to_onnx is None, "to_onnx not installed")
    @ignore_errors(OSError)  # connectivity issues
    @ignore_warnings((UserWarning,))
    def test_ep_onnx_sync_exp(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x):
                ry = x.abs()
                rz = ry.exp()
                rw = rz + 1
                ru = rw.log() + rw
                return ru

        x = torch.randn((5, 4))
        Model()(x)
        ep = torch.export.export(
            Model(), (x,), dynamic_shapes=({0: torch.export.Dim("batch")},)
        )
        onx = to_onnx(ep)
        results = list(
            run_aligned(
                ep,
                onx,
                (x,),
                check_conversion_cls=dict(
                    cls=ExtendedReferenceEvaluator, atol=1e-5, rtol=1e-5
                ),
                verbose=1,
            ),
        )
        self.assertEqual(len(results), 4)

    @hide_stdout()
    @ignore_warnings((DeprecationWarning, FutureWarning, UserWarning))
    def test_ep_onnx_sync_a(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x):
                ry = x.abs()
                rz = ry.exp()
                rw = rz + 1
                ru = rw.log() + rw
                return ru

        x = torch.randn((5, 4))
        Model()(x)
        ep = torch.export.export(
            Model(), (x,), dynamic_shapes=({0: torch.export.Dim("batch")},)
        )
        epo = torch.onnx.export(ep, (x,), dynamic_shapes=({0: torch.export.Dim("batch")},))
        onx = epo.model_proto
        results = list(
            run_aligned(
                ep,
                onx,
                (x,),
                check_conversion_cls=dict(
                    cls=ExtendedReferenceEvaluator, atol=1e-4, rtol=1e-4
                ),
                verbose=1,
            ),
        )
        self.assertEqual(len(results), 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
