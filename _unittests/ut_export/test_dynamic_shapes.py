import unittest
import torch
from onnx_diagnostic.ext_test_case import ExtTestCase
from onnx_diagnostic.helpers import string_type
from onnx_diagnostic.export import ModelInputs


class TestDynamicShapes(ExtTestCase):
    def test_guess_dynamic_shapes_names(self):
        class Model(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        model = Model()
        x = torch.randn((5, 6))
        y = torch.randn((1, 6))
        model(x, y)
        self.assertNotEmpty(y)

        mi = ModelInputs(Model(), [])
        self.assertEqual(mi.name, "main")
        self.assertEqual(mi.true_model_name, "Model")
        self.assertEqual(mi.full_name, "main:Model")

    def test_guess_dynamic_shapes_none(self):
        class Model(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        model = Model()
        x = torch.randn((5, 6))
        y = torch.randn((1, 6))
        model(x, y)
        self.assertNotEmpty(y)

        mi = ModelInputs(Model(), [])
        ds = mi.guess_dynamic_shapes()
        self.assertEmpty(ds)

    def test_guess_dynamic_shapes_1args(self):
        class Model(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        model = Model()
        x = torch.randn((5, 6))
        y = torch.randn((1, 6))
        model(x, y)
        self.assertNotEmpty(y)

        inputs = [(x, y)]

        self.assertRaise(lambda: ModelInputs(Model(), {}), ValueError)
        mi = ModelInputs(Model(), inputs)
        expected = "#1[((T1s5x6,T1s1x6),{})]"
        self.assertEqual(expected, string_type(mi.inputs, with_shape=True))
        ds = mi.guess_dynamic_shapes()
        self.assertEqual(ds, (({}, {}), {}))

    def test_guess_dynamic_shapes_args(self):
        class Model(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        model = Model()
        x = torch.randn((5, 6))
        y = torch.randn((1, 6))
        model(x, y)
        self.assertNotEmpty(y)

        inputs = [(x, y), (torch.randn((7, 8)), torch.randn((1, 8)))]

        self.assertRaise(lambda: ModelInputs(Model(), {}), ValueError)
        mi = ModelInputs(Model(), inputs)
        expected = "#2[((T1s5x6,T1s1x6),{}),((T1s7x8,T1s1x8),{})]"
        self.assertEqual(expected, string_type(mi.inputs, with_shape=True))
        ds = mi.guess_dynamic_shapes()
        self.assertEqual(
            ds,
            (
                (
                    {0: torch.export.Dim.DYNAMIC, 1: torch.export.Dim.DYNAMIC},
                    {1: torch.export.Dim.DYNAMIC},
                ),
                {},
            ),
        )

    def test_guess_dynamic_shapes_kwargs(self):
        class Model(torch.nn.Module):
            def forward(self, x=None, y=None):
                return x + y

        model = Model()
        x = torch.randn((5, 6))
        y = torch.randn((1, 6))
        model(x=x, y=y)
        self.assertNotEmpty(y)

        inputs = [dict(x=x, y=y), dict(x=torch.randn((7, 8)), y=torch.randn((1, 8)))]

        mi = ModelInputs(Model(), inputs)
        expected = "#2[((),dict(x:T1s5x6,y:T1s1x6)),((),dict(x:T1s7x8,y:T1s1x8))]"
        self.assertEqual(expected, string_type(mi.inputs, with_shape=True))
        ds = mi.guess_dynamic_shapes()
        self.assertEqual(
            ds,
            (
                (),
                {
                    "x": {0: torch.export.Dim.DYNAMIC, 1: torch.export.Dim.DYNAMIC},
                    "y": {1: torch.export.Dim.DYNAMIC},
                },
            ),
        )

    def test_guess_dynamic_shapes_args_kwargs(self):
        class Model(torch.nn.Module):
            def forward(self, x, y=None):
                return x + y

        model = Model()
        x = torch.randn((5, 6))
        y = torch.randn((1, 6))
        model(x, y=y)
        self.assertNotEmpty(y)

        inputs = [((x,), dict(y=y)), ((torch.randn((7, 8)),), dict(y=torch.randn((1, 8))))]

        mi = ModelInputs(Model(), inputs)
        self.assertEqualAny(inputs, mi.inputs)
        expected = "#2[((T1s5x6,),dict(y:T1s1x6)),((T1s7x8,),dict(y:T1s1x8))]"
        self.assertEqual(expected, string_type(mi.inputs, with_shape=True))
        ds = mi.guess_dynamic_shapes()
        self.assertEqual(
            ds,
            (
                ({0: torch.export.Dim.DYNAMIC, 1: torch.export.Dim.DYNAMIC},),
                {"y": {1: torch.export.Dim.DYNAMIC}},
            ),
        )

    def test_guess_dynamic_shapes_kwargs_as_kwargs(self):
        class Model(torch.nn.Module):
            def forward(self, **kwargs):
                return kwargs["x"].abs()

        model = Model()
        x = torch.randn((5, 6))
        y = model(x=x)
        self.assertNotEmpty(y)

        inputs = [
            (tuple(), {"x": x}),
            (tuple(), {"x": torch.randn((6, 6))}),
        ]

        mi = ModelInputs(model, inputs)
        expected = "#2[((),dict(x:T1s5x6)),((),dict(x:T1s6x6))]"
        self.assertEqual(expected, string_type(mi.inputs, with_shape=True))
        ds = mi.guess_dynamic_shapes()
        self.assertEqual(ds, (tuple(), {"x": {0: torch.export.Dim.DYNAMIC}}))
        _a, _kw, ds = mi.move_to_kwargs(*mi.inputs[0], ds)
        self.assertEqual(ds, (tuple(), {"kwargs": {"x": {0: torch.export.Dim.DYNAMIC}}}))


if __name__ == "__main__":
    unittest.main(verbosity=2)
