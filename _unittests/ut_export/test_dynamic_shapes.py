import unittest
import torch
from onnx_diagnostic.ext_test_case import ExtTestCase
from onnx_diagnostic.cache_helper import make_dynamic_cache
from onnx_diagnostic.helper import string_type
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
        self.assertEqual(mi.module_name_type, "type(main)=Model")

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
        self.assertEqual(ds, ((), {}))

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
        self.assertEqual(
            (({}, {}), {}), ModelInputs(Model(), inputs[:1]).guess_dynamic_shapes()
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
        self.assertEqual(
            ((), {"x": {}, "y": {}}), ModelInputs(Model(), inputs[:1]).guess_dynamic_shapes()
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
        self.assertEqual(
            (({},), {"y": {}}), ModelInputs(Model(), inputs[:1]).guess_dynamic_shapes()
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
        self.assertEqual(
            ((), {"x": {}}), ModelInputs(Model(), inputs[:1]).guess_dynamic_shapes()
        )

    def test_guess_dynamic_shapes_exc(self):
        class Model(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        model = Model()
        x = torch.randn((5, 6))
        y = torch.randn((1, 6))
        model(x, y)
        self.assertNotEmpty(y)

        self.assertRaise(lambda: ModelInputs(Model(), [[]]), ValueError)

    def test_guess_dynamic_shapes_forward2(self):
        class Model(torch.nn.Module):
            def forward2(self, x, y):
                return x + y

        model = Model()
        x = torch.randn((5, 6))
        y = torch.randn((1, 6))
        model.forward2(x, y)
        self.assertNotEmpty(y)

        inputs = [(x, y), (torch.randn((7, 8)), torch.randn((1, 8)))]

        mi = ModelInputs(Model(), inputs, method_name="forward2")
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
        self.assertEqual(
            (({}, {}), {}), ModelInputs(Model(), inputs[:1]).guess_dynamic_shapes()
        )
        self.assertEqual("Model", mi.true_model_name)
        self.assertEqual("main:Model.forward2", mi.full_name)
        self.assertEqual("type(main)=Model.forward2", mi.module_name_type)

    def test_guess_dynamic_shapes_null_1(self):
        class Model(torch.nn.Module):
            def forward2(self, x, y):
                return x + y

        model = Model()
        x = torch.randn((5, 6))
        y = torch.randn((1, 6))
        model.forward2(x, y)
        self.assertNotEmpty(y)

        inputs = [(x, y), (torch.randn((7, 8)), torch.tensor([]).unsqueeze(0))]

        mi = ModelInputs(Model(), inputs, method_name="forward2")
        expected = "#2[((T1s5x6,T1s1x6),{}),((T1s7x8,T1s1x0),{})]"
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

    def test_guess_dynamic_shapes_null_0(self):
        class Model(torch.nn.Module):
            def forward2(self, x, y):
                return x + y

        model = Model()
        x = torch.randn((5, 6))
        y = torch.randn((1, 6))
        model.forward2(x, y)
        self.assertNotEmpty(y)

        inputs = [(x, y), (torch.randn((7, 8))[0:0, :], torch.randn((1, 8)))]

        mi = ModelInputs(Model(), inputs, method_name="forward2")
        expected = "#2[((T1s5x6,T1s1x6),{}),((T1s0x8,T1s1x8),{})]"
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

    def test_guess_dynamic_shapes_nested_tuple(self):
        class Model(torch.nn.Module):
            def forward2(self, xy, z):
                return xy[0] + xy[1] + z

        model = Model()
        x = torch.randn((5, 6))
        y = torch.randn((1, 6))
        z = torch.randn((1, 6))
        model.forward2((x, y), z)
        self.assertNotEmpty(y)

        inputs = [
            ((x, y), z),
            ((torch.randn((7, 8))[0:0, :], torch.randn((1, 8))), torch.randn((1, 8))),
        ]

        mi = ModelInputs(Model(), inputs, method_name="forward2")
        expected = "#2[(((T1s5x6,T1s1x6),T1s1x6),{}),(((T1s0x8,T1s1x8),T1s1x8),{})]"
        self.assertEqual(expected, string_type(mi.inputs, with_shape=True))
        ds = mi.guess_dynamic_shapes()
        self.assertEqual(
            ds,
            (
                (
                    (
                        {0: torch.export.Dim.DYNAMIC, 1: torch.export.Dim.DYNAMIC},
                        {1: torch.export.Dim.DYNAMIC},
                    ),
                    {1: torch.export.Dim.DYNAMIC},
                ),
                {},
            ),
        )

    def test_guess_dynamic_shapes_nested_list(self):
        class Model(torch.nn.Module):
            def forward(self, xy, z):
                return xy[0] + xy[1] + z

        model = Model()
        x = torch.randn((5, 6))
        y = torch.randn((1, 6))
        z = torch.randn((1, 6))
        model([x, y], z)
        self.assertNotEmpty(y)

        inputs = [
            ([x, y], z),
            ([torch.randn((7, 8))[0:0, :], torch.randn((1, 8))], torch.randn((1, 8))),
        ]

        mi = ModelInputs(Model(), inputs)
        expected = "#2[((#2[T1s5x6,T1s1x6],T1s1x6),{}),((#2[T1s0x8,T1s1x8],T1s1x8),{})]"
        self.assertEqual(expected, string_type(mi.inputs, with_shape=True))
        ds = mi.guess_dynamic_shapes()
        self.assertEqual(
            ds,
            (
                (
                    [
                        {0: torch.export.Dim.DYNAMIC, 1: torch.export.Dim.DYNAMIC},
                        {1: torch.export.Dim.DYNAMIC},
                    ],
                    {1: torch.export.Dim.DYNAMIC},
                ),
                {},
            ),
        )

    def test_guess_dynamic_shapes_nested_dict(self):
        class Model(torch.nn.Module):
            def forward(self, xy, z):
                return xy["x"] + xy["y"] + z

        model = Model()
        x = torch.randn((5, 6))
        y = torch.randn((1, 6))
        z = torch.randn((1, 6))
        model(dict(x=x, y=y), z)
        self.assertNotEmpty(y)

        inputs = [
            (dict(x=x, y=y), z),
            (dict(x=torch.randn((7, 8))[0:0, :], y=torch.randn((1, 8))), torch.randn((1, 8))),
        ]

        mi = ModelInputs(Model(), inputs)
        expected = (
            "#2[((dict(x:T1s5x6,y:T1s1x6),T1s1x6),{}),((dict(x:T1s0x8,y:T1s1x8),T1s1x8),{})]"
        )
        self.assertEqual(expected, string_type(mi.inputs, with_shape=True))
        ds = mi.guess_dynamic_shapes()
        self.assertEqual(
            ds,
            (
                (
                    dict(
                        x={0: torch.export.Dim.DYNAMIC, 1: torch.export.Dim.DYNAMIC},
                        y={1: torch.export.Dim.DYNAMIC},
                    ),
                    {1: torch.export.Dim.DYNAMIC},
                ),
                {},
            ),
        )

    def test_guess_dynamic_shapes_cache(self):
        class Model(torch.nn.Module):
            def forward(self, cache, z):
                return (
                    z
                    + cache.key_cache[0]
                    + cache.key_cache[1]
                    + cache.value_cache[0]
                    + cache.value_cache[1]
                )

        model = Model()

        n_layers = 2
        bsize, nheads, slen, dim = 2, 4, 3, 7
        cache = make_dynamic_cache(
            [
                (torch.randn(bsize, nheads, slen, dim), torch.randn(bsize, nheads, slen, dim))
                for i in range(n_layers)
            ]
        )
        z = torch.randn((1, 1, 1, 7))
        model(cache, z)

        cache2 = make_dynamic_cache(
            [
                (
                    torch.randn(bsize, nheads, slen, dim),
                    torch.randn(bsize + 1, nheads, slen + 1, dim + 1),
                )
                for i in range(n_layers)
            ]
        )
        inputs = [
            (cache, z),
            (cache2, torch.randn((1, 1, 1, 8))),
        ]

        mi = ModelInputs(Model(), inputs)
        self.assertIn("DynamicCache", string_type(mi.inputs, with_shape=True))
        ds = mi.guess_dynamic_shapes()
        self.assertEqual(
            ds,
            (
                (
                    [
                        [{}, {}],
                        [
                            {
                                0: torch.export.Dim.DYNAMIC,
                                2: torch.export.Dim.DYNAMIC,
                                3: torch.export.Dim.DYNAMIC,
                            },
                            {
                                0: torch.export.Dim.DYNAMIC,
                                2: torch.export.Dim.DYNAMIC,
                                3: torch.export.Dim.DYNAMIC,
                            },
                        ],
                    ],
                    {3: torch.export.Dim.DYNAMIC},
                ),
                {},
            ),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
