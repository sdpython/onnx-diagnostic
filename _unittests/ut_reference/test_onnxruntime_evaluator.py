import unittest
import numpy as np
import onnx
import onnx.helper as oh
import torch
import onnxruntime
from onnx_diagnostic.ext_test_case import ExtTestCase
from onnx_diagnostic.helpers.onnx_helper import from_array_extended
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

    def test_constant_bool(self):
        node = oh.make_node(
            "Constant",
            [],
            ["cbool"],
            value=from_array_extended(np.array(True, dtype=np.bool_)),
        )
        ref = ExtendedReferenceEvaluator(node)
        got = ref.run(None, {})[0]
        self.assertEqual(got.dtype, np.bool_)
        self.assertEqual(got, True)
        ref = OnnxruntimeEvaluator(node, opsets=21)
        got = ref.run(None, {})[0]
        self.assertEqual(len(ref._cache), 1)
        values = list(ref._cache.values())
        _, sess = values[0]
        got2 = sess.run(None, {})[0]
        self.assertIn(got2.dtype, (torch.bool, np.bool_))
        self.assertEqual(got2, True)

        self.assertIn(got.dtype, (torch.bool, np.bool_))
        self.assertEqual(got, True)

    def test_constant_bool_array(self):
        node = oh.make_node(
            "Constant",
            [],
            ["cbool"],
            value=from_array_extended(np.array([True], dtype=np.bool_)),
        )
        ref = ExtendedReferenceEvaluator(node)
        got = ref.run(None, {})[0]
        self.assertEqual(got.dtype, np.bool_)
        self.assertEqual(got[0], True)
        ref = OnnxruntimeEvaluator(node, opsets=21)
        got = ref.run(None, {})[0]
        self.assertEqual(len(ref._cache), 1)
        values = list(ref._cache.values())
        _, sess = values[0]
        got2 = sess.run(None, {})[0]
        self.assertIn(got2.dtype, (torch.bool, np.bool_))
        self.assertEqual(got2[0], True)

        self.assertIn(got.dtype, (torch.bool, np.bool_))
        self.assertEqual(got[0], True)

    def test_constant_bool_input(self):
        node = oh.make_model(
            oh.make_graph(
                [oh.make_node("Identity", ["bin"], ["bout"])],
                "test",
                [oh.make_tensor_value_info("bin", onnx.TensorProto.BOOL, [1])],
                [oh.make_tensor_value_info("bin", onnx.TensorProto.BOOL, [1])],
            ),
            ir_version=10,
            opset_imports=[oh.make_opsetid("", 18)],
        )
        feeds = dict(bin=np.array([True], dtype=np.bool_))
        ref = ExtendedReferenceEvaluator(node)

        got = ref.run(None, feeds)[0]
        self.assertEqual(got.dtype, np.bool_)
        self.assertEqual(got[0], True)

        ref = OnnxruntimeEvaluator(node, opsets=21)
        got = ref.run(None, feeds)[0]
        self.assertEqual(got.dtype, np.bool_)
        self.assertEqual(got[0], True)

        feeds = dict(bin=torch.tensor([True], dtype=torch.bool))
        got = ref.run(None, feeds)[0]
        self.assertEqual(got.dtype, torch.bool)
        self.assertEqual(got[0], True)

    def test_ort_eval_loop(self):
        model = torch.nn.EmbeddingBag(num_embeddings=49157, embedding_dim=32, mode="sum")
        a = torch.tensor([[39906, 39906]]).long()
        example_args = (a,)
        model_eval = model.eval()
        expected = model(*example_args)

        onx = to_onnx(model_eval, example_args, optimize=True)
        self.assertIn("Loop", set(n.op_type for n in onx.graph.node))

        ref = OnnxruntimeEvaluator(onx, verbose=10)
        feeds = dict(
            zip([i.name for i in onx.graph.input], [t.detach().numpy() for t in example_args])
        )
        got = ref.run(None, feeds)
        self.assertEqualArray(expected, got[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
