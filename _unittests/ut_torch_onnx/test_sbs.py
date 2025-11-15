import unittest
import onnx
from onnx_diagnostic.ext_test_case import (
    ExtTestCase,
    hide_stdout,
    ignore_warnings,
    ignore_errors,
    requires_cuda,
)
from onnx_diagnostic.reference import ExtendedReferenceEvaluator, OnnxruntimeEvaluator
from onnx_diagnostic.torch_export_patches.patch_inputs import use_dyn_not_str
from onnx_diagnostic.torch_onnx.sbs import run_aligned
from onnx_diagnostic.export.api import to_onnx


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
        onx = to_onnx(ep, exporter="custom").model_proto
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
        onx = to_onnx(
            ep,
            (x,),
            dynamic_shapes=({0: self.torch.export.Dim("batch")},),
            exporter="onnx-dynamo",
        ).model_proto
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
        epo = to_onnx(Model(), (), kwargs=inputs, dynamic_shapes=ds, exporter="onnx-dynamo")
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

    @hide_stdout()
    @ignore_warnings((DeprecationWarning, FutureWarning, UserWarning))
    def test_sbs_dict_onnxruntime(self):
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
        onx = to_onnx(ep, exporter="custom").model_proto
        results = list(
            run_aligned(
                ep,
                onx,
                kwargs=inputs,
                run_cls=OnnxruntimeEvaluator,
                atol=1e-5,
                rtol=1e-5,
                verbose=11,
            ),
        )
        self.assertEqual(len(results), 5)

    @hide_stdout()
    @ignore_warnings((DeprecationWarning, FutureWarning, UserWarning))
    def test_sbs_dict_tensor(self):
        class Model(self.torch.nn.Module):
            def forward(self, x):
                ry = x.abs()
                rz = ry.exp()
                rw = rz + 1
                ru = rw.log() + rw + ry
                return ru

        inputs = dict(x=self.torch.randn((5, 4)))
        ds = dict(x={0: "batch"})
        Model()(**inputs)
        ep = self.torch.export.export(
            Model(), (), kwargs=inputs, dynamic_shapes=use_dyn_not_str(ds)
        )
        onx = to_onnx(ep, exporter="custom").model_proto
        results = list(
            run_aligned(
                ep,
                onx,
                kwargs=inputs,
                run_cls=OnnxruntimeEvaluator,
                atol=1e-5,
                rtol=1e-5,
                verbose=11,
                use_tensor=True,
            ),
        )
        self.assertEqual(len(results), 6)
        self.clean_dump()

    @hide_stdout()
    @ignore_warnings((DeprecationWarning, FutureWarning, UserWarning))
    @requires_cuda()
    def test_sbs_dict_tensor_cuda(self):
        class Model(self.torch.nn.Module):
            def forward(self, x):
                ry = x.abs()
                rz = ry.exp()
                rw = rz + 1
                ru = rw.log() + rw + ry
                return ru

        inputs = dict(x=self.torch.randn((5, 4)).to("cuda"))
        ds = dict(x={0: "batch"})
        Model()(**inputs)
        ep = self.torch.export.export(
            Model(), (), kwargs=inputs, dynamic_shapes=use_dyn_not_str(ds)
        )
        onx = to_onnx(ep, exporter="custom").model_proto
        results = list(
            run_aligned(
                ep,
                onx,
                kwargs=inputs,
                run_cls=OnnxruntimeEvaluator,
                atol=1e-5,
                rtol=1e-5,
                verbose=11,
                use_tensor=True,
            ),
        )
        self.assertEqual(len(results), 6)
        self.assertEqual([r[-1]["dev"] for r in results], [0, 0, 0, 0, 0, 0])

    @hide_stdout()
    @ignore_warnings((DeprecationWarning, FutureWarning, UserWarning))
    @requires_cuda()
    def test_sbs_dict_tensor_cuda_reshape(self):
        class Model(self.torch.nn.Module):
            def forward(self, x):
                ry = x.abs()
                ry1 = ry.reshape((-1, 1))
                ry2 = ry.reshape((1, -1))
                prod = ry1 * ry2
                shape = prod.shape
                resh = prod.reshape((-1, shape[0] // 2, shape[1] // 2))
                return resh.transpose(2, 1)

        inputs = dict(x=self.torch.randn((16, 16)).to("cuda"))
        ds = dict(x={0: "batch"})
        Model()(**inputs)
        ep = self.torch.export.export(
            Model(), (), kwargs=inputs, dynamic_shapes=use_dyn_not_str(ds)
        )
        onx = to_onnx(ep, exporter="custom").model_proto
        results = list(
            run_aligned(
                ep,
                onx,
                kwargs=inputs,
                run_cls=OnnxruntimeEvaluator,
                atol=1e-5,
                rtol=1e-5,
                verbose=11,
                use_tensor=True,
            ),
        )
        self.assertEqual(len(results), 7)
        self.assertEqual([r[-1].get("dev", 0) for r in results], [0, 0, 0, 0, 0, 0, 0])

    @hide_stdout()
    @ignore_warnings((DeprecationWarning, FutureWarning, UserWarning))
    def test_sbs_dict_tensor_cpu_reshape(self):
        class Model(self.torch.nn.Module):
            def forward(self, x):
                ry = x.abs()
                ry1 = ry.reshape((-1, 1))
                ry2 = ry.reshape((1, -1))
                prod = ry1 * ry2
                shape = prod.shape
                resh = prod.reshape((-1, shape[0] // 2, shape[1] // 2))
                return resh.transpose(2, 1)

        inputs = dict(x=self.torch.randn((16, 16)))
        ds = dict(x={0: "batch"})
        Model()(**inputs)
        ep = self.torch.export.export(
            Model(), (), kwargs=inputs, dynamic_shapes=use_dyn_not_str(ds)
        )
        onx = to_onnx(ep, exporter="custom").model_proto
        results = list(
            run_aligned(
                ep,
                onx,
                kwargs=inputs,
                run_cls=OnnxruntimeEvaluator,
                atol=1e-5,
                rtol=1e-5,
                verbose=11,
                use_tensor=True,
            ),
        )
        self.assertEqual(len(results), 7)
        self.assertEqual([r[-1].get("dev", 0) for r in results], [0, 0, 0, 0, 0, 0, 0])

    @hide_stdout()
    @ignore_warnings((DeprecationWarning, FutureWarning, UserWarning))
    def test_sbs_model_with_weights(self):
        torch = self.torch

        class Model(self.torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.fc1 = torch.nn.Linear(10, 32)  # input size 10 → hidden size 32
                self.relu = torch.nn.ReLU()
                self.fc2 = torch.nn.Linear(32, 1)  # hidden → output

            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.fc2(x)
                return x

        inputs = dict(x=self.torch.randn((5, 10)))
        ds = dict(x={0: "batch"})
        Model()(**inputs)
        ep = self.torch.export.export(
            Model(), (), kwargs=inputs, dynamic_shapes=use_dyn_not_str(ds)
        )
        filename = self.get_dump_file("test_sbs_model_with_weights.onnx")
        to_onnx(ep, exporter="custom", filename=filename)
        onx = onnx.load(filename)
        results = list(
            run_aligned(
                ep,
                onx,
                kwargs=inputs,
                run_cls=OnnxruntimeEvaluator,
                verbose=11,
                use_tensor=True,
            ),
        )
        self.assertEqual(len(results), 7)
        self.assertEqual([r[-1].get("dev", 0) for r in results], [0, 0, 0, 0, 0, 0, 0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
