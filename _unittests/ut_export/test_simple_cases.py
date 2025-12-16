import unittest
import torch
from onnx_diagnostic.ext_test_case import ExtTestCase
from onnx_diagnostic.reference import ExtendedReferenceEvaluator


class TestSimpleCases(ExtTestCase):
    def test_getitem_index_put1(self):
        class Model(torch.nn.Module):
            def forward(self, x, value):
                x = x.clone()
                x[:, :, :, : value.shape[-1]] = value
                return x

        inputs = (torch.randn(2, 2, 3, 4), torch.randn(2, 2, 3, 3))
        model = Model()
        expected = model(*inputs)

        onx = self.to_onnx(model, inputs, dynamic_shapes=({3: "M"}, {3: "N"}))
        self.dump_onnx("test_getitem_index_put1.onnx", onx)
        feeds = dict(zip(["x", "value"], [x.detach().cpu().numpy() for x in inputs]))
        ref = ExtendedReferenceEvaluator(onx, verbose=0)
        got = ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-5)
        sess = self.ort().InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, feeds)[0]
        self.assertEqualArray(expected, got, atol=1e-5)

    def test_cond_calling_submodule_with_weights1(self):
        class Model(torch.nn.Module):
            def __init__(self, in_features, out_features):
                super().__init__()
                self.linear1 = torch.nn.Linear(in_features, out_features)
                self.linear2 = torch.nn.Linear(in_features, out_features)

            def forward(self, x):
                return torch.cond(
                    x.sum().item() > 0,
                    lambda x: self.linear1(x),
                    lambda x: self.linear2(x),
                    [x],
                )

        # Example usage
        model = Model(in_features=4, out_features=2)
        x = torch.randn(3, 4)
        expected = model(x)
        ep = torch.export.export(model, (x,), dynamic_shapes=({0: torch.export.Dim.DYNAMIC},))
        got = ep.module()(x)
        self.assertEqualArray(expected, got)
        self.assertEqualArray(model(-x), ep.module()(-x))

    def test_cond_calling_submodule_with_weights2(self):

        def branch1(x, fn):
            return fn(x)

        def branch2(x, fn):
            return fn(x)

        class Model(torch.nn.Module):
            def __init__(self, in_features, out_features):
                super().__init__()
                self.linear1 = torch.nn.Linear(in_features, out_features)
                self.linear2 = torch.nn.Linear(in_features, out_features)

            def forward(self, x):
                return torch.cond(
                    x.sum().item() > 0,
                    lambda x: branch1(x, fn=self.linear1),
                    lambda x: branch2(x, fn=self.linear2),
                    [x],
                )

        # Example usage
        model = Model(in_features=4, out_features=2)
        x = torch.randn(3, 4)
        expected = model(x)
        ep = torch.export.export(model, (x,), dynamic_shapes=({0: torch.export.Dim.DYNAMIC},))
        got = ep.module()(x)
        self.assertEqualArray(expected, got)
        self.assertEqualArray(model(-x), ep.module()(-x))


if __name__ == "__main__":
    unittest.main(verbosity=2)
