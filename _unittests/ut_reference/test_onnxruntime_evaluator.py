import unittest
from typing import Optional
import numpy as np
import onnx
import onnx.helper as oh
import torch
import onnxruntime
from onnx_diagnostic.ext_test_case import ExtTestCase, hide_stdout, ignore_warnings
from onnx_diagnostic.helpers.onnx_helper import from_array_extended
from onnx_diagnostic.reference import (
    OnnxruntimeEvaluator,
    ExtendedReferenceEvaluator,
    ReportResultComparison,
)

try:
    from experimental_experiment.torch_interpreter import to_onnx, ExportOptions
except ImportError:
    to_onnx = None


TFLOAT = onnx.TensorProto.FLOAT


class TestOnnxruntimeEvaluator(ExtTestCase):
    def _range(self, *shape, bias: Optional[float] = None):
        n = np.prod(shape)
        x = np.arange(n).astype(np.float32) / n
        if bias:
            x = x + bias
        return x.reshape(tuple(shape)).astype(np.float32)

    @ignore_warnings(FutureWarning)
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

    @ignore_warnings((UserWarning, FutureWarning))
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

    @hide_stdout()
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

    @hide_stdout()
    def test_report_results_comparison_ort(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Cos", ["X"], ["nx"]),
                    oh.make_node("Sin", ["nx"], ["t"]),
                    oh.make_node("Exp", ["t"], ["u"]),
                    oh.make_node("Log", ["u"], ["uZ"]),
                    oh.make_node("Erf", ["uZ"], ["Z"]),
                ],
                "dummy",
                [oh.make_tensor_value_info("X", TFLOAT, ["a", "b"])],
                [oh.make_tensor_value_info("Z", TFLOAT, ["a", "b"])],
            ),
            ir_version=9,
            opset_imports=[oh.make_opsetid("", 18)],
        )
        x = torch.rand(5, 6, dtype=torch.float32)
        onnx.checker.check_model(model)
        cmp = ReportResultComparison(dict(r_x=x, r_cos=x.cos(), r_exp=x.cos().sin().exp()))
        cmp.clear()
        feeds = dict(zip([i.name for i in model.graph.input], (x,)))
        rt = OnnxruntimeEvaluator(model, verbose=10)
        rt.run(None, feeds, report_cmp=cmp)
        d = {k: d["abs"] for k, d in cmp.value.items()}
        self.assertLess(d[(0, "nx"), "r_cos"], 1e-6)
        self.assertLess(d[(2, "u"), "r_exp"], 1e-6)

    @hide_stdout()
    def test_skip_layer_normalization(self):
        node = oh.make_node(
            "SkipLayerNormalization",
            ["x", "skip", "beta", "gamma", "bias"],
            ["Z"],
            epsilon=1.0e-5,
            domain="com.microsoft",
        )
        feeds = dict(
            x=self._range(2, 3, 8),
            skip=self._range(2, 3, 8, bias=3),
            beta=self._range(8, bias=1),
            gamma=self._range(8, bias=2),
            bias=self._range(8, bias=0.1),
        )
        ref = ExtendedReferenceEvaluator(node)
        expected = ref.run(None, feeds)
        rt = OnnxruntimeEvaluator(node, verbose=10, opsets={"": 22})
        got = rt.run(None, feeds)
        self.assertEqualAny(expected, got, atol=1e-4)

    @hide_stdout()
    def test_skip_simplified_layer_normalization(self):
        node = oh.make_node(
            "SkipSimplifiedLayerNormalization",
            ["x", "skip", "beta", "gamma"],
            ["Z", "", "", "bias"],
            epsilon=1.0e-5,
            domain="com.microsoft",
        )
        feeds = dict(
            x=self._range(2, 3, 8),
            skip=self._range(2, 3, 8, bias=3),
            beta=self._range(8, bias=1),
            gamma=self._range(8, bias=2),
        )
        rt = OnnxruntimeEvaluator(node, verbose=10, opsets={"": 22})
        got = rt.run(None, feeds)
        self.assertEqual(len(got), 2)
        self.assertIsInstance(got[0], np.ndarray)
        self.assertIsInstance(got[1], np.ndarray)
        self.assertEqual(got[0].shape, feeds["x"].shape)
        self.assertEqual(got[0].dtype, feeds["x"].dtype)
        self.assertEqual(got[1].shape, feeds["x"].shape)
        self.assertEqual(got[1].dtype, feeds["x"].dtype)

    def test_function_proto_with_kwargs(self):
        linear_function = oh.make_function(
            "test_domain",
            "LinearRegression",
            ["x", "a", "b"],
            ["y"],
            [
                oh.make_node("Constant", [], ["eps"]),
                oh.make_node("Constant", [], ["zero"], value_ints=[0]),
                oh.make_node("Unsqueeze", ["eps", "zero"], ["eps1d"]),
                oh.make_node("MatMul", ["x", "a"], ["xa"]),
                oh.make_node("Add", ["b", "eps1d"], ["beps"]),
                oh.make_node("Add", ["xa", "beps"], ["y"]),
            ],
            [oh.make_opsetid("", 14)],
            ["epsilon"],
        )
        att = onnx.AttributeProto()
        att.name = "value_float"
        att.ref_attr_name = "epsilon"
        att.type = onnx.AttributeProto.FLOAT
        linear_function.node[0].attribute.append(att)
        feeds = dict(
            x=np.random.rand(4, 4).astype(np.float32),
            a=np.random.rand(4, 2).astype(np.float32),
            b=np.random.rand(1, 2).astype(np.float32),
        )
        epsilon = 15.6
        expected = feeds["x"] @ feeds["a"] + feeds["b"] + epsilon
        sess = OnnxruntimeEvaluator(
            linear_function, whole=True, function_kwargs=dict(epsilon=epsilon)
        )
        got = sess.run(None, feeds)
        self.assertEqualArray(expected, got[0], atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
