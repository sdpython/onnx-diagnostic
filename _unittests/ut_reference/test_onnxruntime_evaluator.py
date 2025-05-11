import unittest
import onnx
import torch
import onnxruntime
from onnx_diagnostic.ext_test_case import ExtTestCase
from onnx_diagnostic.reference import OnnxruntimeEvaluator, ExtendedReferenceEvaluator

try:
    from experimental_experiment.torch_interpreter import to_onnx, ExportOptions
except ImportError:
    to_onnx = None


class TestOnnxruntimeEvaluator(ExtTestCase):
    def test_ort_eval_scan_cdist_add(self):

        def dist(unused: torch.Tensor, x: torch.Tensor, samex: torch.Tensor):
            sub = samex - x.reshape((1, -1))
            sq = sub * sub
            rd = torch.sqrt(sq.sum(axis=1))
            # clone --> UnsupportedAliasMutationException:
            # Combine_fn might be aliasing the input!
            return [unused.clone(), rd]

        class ScanModel(torch.nn.Module):
            def forward(self, x):
                z = torch.tensor([0], dtype=torch.float32)
                y = x.clone()
                out = torch.ops.higher_order.scan(dist, [z], [x], additional_inputs=[y])
                return out[1]

        x = torch.tensor([[1, 2, 3, -1], [4, 5, 6, -1], [7, 8, 9, -1]], dtype=torch.float32)
        model = ScanModel()
        expected = model(x)
        onx = to_onnx(
            model,
            (x,),
            optimize=True,
            export_options=ExportOptions(decomposition_table="default", strict=False),
            inline=False,
        )
        filename = self.get_dump_file("test_ort_eval_scan_cdist_add.onnx")
        onnx.save(onx, filename)
        inits = [i.name for i in onx.graph.initializer]
        self.assertEqual(inits, ["c_lifted_tensor_0"])
        name = onx.graph.input[0].name

        sess = onnxruntime.InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, {name: x.numpy()})[0]
        self.assertEqualArray(expected, got)

        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, {name: x.numpy()})[0]
        self.assertEqualArray(expected, got)

        orte = OnnxruntimeEvaluator(onx)
        got = orte.run(None, {name: x.numpy()})[0]
        self.assertEqualArray(expected, got)

    def test_ort_eval_cond(self):
        import torch

        class TwoInputs(torch.nn.Module):
            def forward(self, x, y):
                def true_fn(x, y):
                    return torch.sin(x), torch.cos(x) + y

                def false_fn(x, y):
                    return torch.cos(x), torch.sin(x) + y

                return torch.cond(x.sum() > 0, true_fn, false_fn, [x, y])

        x, y = torch.rand(5, 3), torch.rand(5, 3)
        model = TwoInputs()
        onx = to_onnx(model, (x, y), inline=False)
        self.assertEqual(len(onx.functions), 2)

        # ExtendedReferenceEvaluator
        ref = ExtendedReferenceEvaluator(onx)
        for _x in (x, -x):
            expected = model(_x, y)
            got = ref.run(None, {"x": _x.detach().numpy(), "y": y.detach().numpy()})
            self.assertEqual(len(expected), len(got))
            for e, g in zip(expected, got):
                self.assertEqualArray(e, g, atol=1e-5)

        # OnnxruntimeEvaluator
        ref = OnnxruntimeEvaluator(onx)

        for _x in (x, -x):
            expected = model(_x, y)
            got = ref.run(None, {"x": _x.detach().numpy(), "y": y.detach().numpy()})
            self.assertEqual(len(expected), len(got))
            for e, g in zip(expected, got):
                self.assertEqualArray(e, g, atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
