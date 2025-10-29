import unittest
import torch
import transformers
from onnx_diagnostic.ext_test_case import ExtTestCase, requires_transformers
from onnx_diagnostic.helpers import string_type
from onnx_diagnostic.helpers.cache_helper import make_dynamic_cache, CacheKeyValue
from onnx_diagnostic.export import ModelInputs, CoupleInputsDynamicShapes
from onnx_diagnostic.torch_export_patches import torch_export_patches
from onnx_diagnostic.torch_models.hghub.model_inputs import get_untrained_model_with_inputs


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

    def test_guess_dynamic_shapes_auto(self):
        class Model(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        model = Model()
        x = torch.randn((5, 6))
        y = torch.randn((1, 6))
        model(x, y)
        self.assertNotEmpty(y)

        mi = ModelInputs(Model(), [])
        ds = mi.guess_dynamic_shapes(auto=True)
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
                cache = CacheKeyValue(cache)
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
                        {},
                        {
                            0: torch.export.Dim.DYNAMIC,
                            2: torch.export.Dim.DYNAMIC,
                            3: torch.export.Dim.DYNAMIC,
                        },
                        {},
                        {
                            0: torch.export.Dim.DYNAMIC,
                            2: torch.export.Dim.DYNAMIC,
                            3: torch.export.Dim.DYNAMIC,
                        },
                    ],
                    {3: torch.export.Dim.DYNAMIC},
                ),
                {},
            ),
        )

    @requires_transformers("4.51")
    def test_guess_dynamic_shapes_cache_str(self):
        class Model(torch.nn.Module):
            def forward(self, cache, z):
                cache = CacheKeyValue(cache)
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
        ds = mi.guess_dynamic_shapes(auto="dim")
        self.assertEqual(
            ds,
            (
                (
                    [
                        {},
                        {0: "dim_0I_1o0", 2: "dim_0I_1o2", 3: "dim_0I_1o3"},
                        {},
                        {0: "dim_0I_3o0", 2: "dim_0I_3o2", 3: "dim_0I_3o3"},
                    ],
                    {3: "dim_1I3"},
                ),
                {},
            ),
        )

    def test_couple_input_ds_0(self):
        T3x4 = torch.rand((3, 4))
        T3x1 = torch.rand((3, 1))
        Cls = CoupleInputsDynamicShapes
        self.assertEmpty(Cls((T3x4,), {}, ({0: "batch"},)).invalid_dimensions_for_export())
        self.assertEmpty(Cls((T3x1,), {}, ({0: "batch"},)).invalid_dimensions_for_export())
        self.assertEmpty(
            Cls((), {"A": T3x1}, {"A": {0: "batch"}}).invalid_dimensions_for_export()
        )
        self.assertEmpty(
            Cls((), {"A": T3x4}, {"A": {0: "batch"}}).invalid_dimensions_for_export()
        )

        T1x4 = torch.rand((1, 4))
        T1x1 = torch.rand((1, 1))
        Cls = CoupleInputsDynamicShapes
        self.assertEqual(
            ({0: "d=[1]"},), Cls((T1x4,), {}, ({0: "batch"},)).invalid_dimensions_for_export()
        )
        self.assertEqual(
            ({0: "d=[1]"},), Cls((T1x1,), {}, ({0: "batch"},)).invalid_dimensions_for_export()
        )
        self.assertEqual(
            {"A": {0: "d=[1]"}},
            Cls((), {"A": T1x1}, {"A": {0: "batch"}}).invalid_dimensions_for_export(),
        )
        self.assertEqual(
            {"A": {0: "d=[1]"}},
            Cls((), {"A": T1x4}, {"A": {0: "batch"}}).invalid_dimensions_for_export(),
        )

    def test_couple_input_ds_1(self):
        T3x1 = torch.rand((3, 1))
        T3x4 = torch.rand((3, 4))
        ds_batch = {0: "batch"}
        ds_batch_seq = {0: "batch", 1: "seq"}
        args = (T3x4, T3x1)
        Cls = CoupleInputsDynamicShapes
        self.assertEqual(
            None, Cls(args, {}, (ds_batch, ds_batch)).invalid_dimensions_for_export()
        )
        self.assertEqual(
            (None, {1: "d=[1]"}),
            Cls(args, {}, (ds_batch, ds_batch_seq)).invalid_dimensions_for_export(),
        )

    def test_couple_input_ds_2(self):
        T3x1 = torch.rand((3, 1))
        T3x4 = torch.rand((3, 4))
        ds_batch = {0: "batch"}
        ds_batch_seq = {0: "batch", 1: "seq"}
        kwargs = {"A": T3x4, "B": T3x1}
        Cls = CoupleInputsDynamicShapes
        self.assertEqual(
            None,
            Cls((), kwargs, {"A": ds_batch, "B": ds_batch}).invalid_dimensions_for_export(),
        )
        self.assertEqual(
            {"B": {1: "d=[1]"}},
            Cls(
                (), kwargs, {"A": ds_batch, "B": ds_batch_seq}
            ).invalid_dimensions_for_export(),
        )

    def test_couple_input_ds_3(self):
        T3x1 = torch.rand((3, 1))
        T3x4 = torch.rand((3, 4))
        ds_batch = {0: "batch"}
        ds_batch_seq = {0: "batch", 1: "seq"}
        kwargs = {"A": T3x4, "B": (T3x1, T3x1)}
        Cls = CoupleInputsDynamicShapes
        self.assertEqual(
            None,
            Cls(
                (), kwargs, {"A": ds_batch, "B": (ds_batch, ds_batch)}
            ).invalid_dimensions_for_export(),
        )
        self.assertEqual(
            {"B": (None, {1: "d=[1]"})},
            Cls(
                (), kwargs, {"A": ds_batch, "B": (ds_batch, ds_batch_seq)}
            ).invalid_dimensions_for_export(),
        )

    def test_couple_input_ds_cache(self):
        T3x1 = torch.rand((3, 1))
        T3x4 = torch.rand((3, 4))
        ds_batch = {0: "batch"}
        ds_batch_seq = {0: "batch", 2: "seq"}

        n_layers = 2
        bsize, nheads, slen, dim = 2, 4, 1, 7
        cache = make_dynamic_cache(
            [
                (torch.randn(bsize, nheads, slen, dim), torch.randn(bsize, nheads, slen, dim))
                for i in range(n_layers)
            ]
        )

        kwargs = {"A": T3x4, "B": (T3x1, cache)}
        Cls = CoupleInputsDynamicShapes
        with torch_export_patches(patch_transformers=True):
            self.assertEqual(
                None,
                Cls(
                    (),
                    kwargs,
                    {
                        "A": ds_batch,
                        "B": (ds_batch, [ds_batch, ds_batch, ds_batch, ds_batch]),
                    },
                ).invalid_dimensions_for_export(),
            )
            self.assertEqual(
                {"B": (None, [None, {2: "d=[1]"}, None, {2: "d=[1]"}])},
                Cls(
                    (),
                    kwargs,
                    {
                        "A": ds_batch,
                        "B": (ds_batch, [ds_batch, ds_batch_seq, ds_batch, ds_batch_seq]),
                    },
                ).invalid_dimensions_for_export(),
            )

    def test_couple_input_ds_args_kwargs_0(self):
        T3x1 = torch.rand((3, 1))
        T3x4 = torch.rand((3, 4))
        T5x6 = torch.rand((5, 6))
        ds_batch = {0: "batch"}
        ds_batch_seq = {0: "batch", 1: "seq"}
        args = (T5x6,)
        kwargs = {"A": T3x4, "B": (T3x1, T3x1)}
        Cls = CoupleInputsDynamicShapes
        self.assertEqual(
            None,
            Cls(
                args, kwargs, {"A": ds_batch, "B": (ds_batch, ds_batch)}
            ).invalid_dimensions_for_export(),
        )
        self.assertEqual(
            None,
            Cls(
                args, kwargs, {"A": ds_batch, "B": (ds_batch, ds_batch)}, args_names=["X"]
            ).invalid_dimensions_for_export(),
        )
        self.assertEqual(
            {"B": (None, {1: "d=[1]"})},
            Cls(
                args, kwargs, {"A": ds_batch, "B": (ds_batch, ds_batch_seq)}
            ).invalid_dimensions_for_export(),
        )

    def test_couple_input_ds_args_kwargs_1(self):
        T3x1 = torch.rand((3, 1))
        T3x4 = torch.rand((3, 4))
        T5x1 = torch.rand((5, 1))
        ds_batch = {0: "batch"}
        ds_batch_seq = {0: "batch", 1: "seq"}
        args = (T5x1,)
        kwargs = {"A": T3x4, "B": (T3x1, T3x1)}
        Cls = CoupleInputsDynamicShapes
        self.assertEqual(
            None,
            Cls(
                args,
                kwargs,
                {"X": ds_batch, "A": ds_batch, "B": (ds_batch, ds_batch)},
                args_names=["X"],
            ).invalid_dimensions_for_export(),
        )
        self.assertEqual(
            {"X": {1: "d=[1]"}, "B": (None, {1: "d=[1]"})},
            Cls(
                args,
                kwargs,
                {"X": ds_batch_seq, "A": ds_batch, "B": (ds_batch, ds_batch_seq)},
                args_names=["X"],
            ).invalid_dimensions_for_export(),
        )

    def test_couple_input_ds_replace_string(self):
        T3x1 = torch.rand((3, 1))
        T3x4 = torch.rand((3, 4))
        T5x1 = torch.rand((5, 1))
        ds_batch = {0: "batch"}
        ds_batch_seq = {0: "batch", 1: "seq"}
        args = (T5x1,)
        kwargs = {"A": T3x4, "B": (T3x1, T3x1)}
        Cls = CoupleInputsDynamicShapes
        self.assertEqual(
            {"X": {0: "DYN"}, "A": {0: "DYN"}, "B": ({0: "DYN"}, {0: "DYN"})},
            Cls(
                args,
                kwargs,
                {"X": ds_batch, "A": ds_batch, "B": (ds_batch, ds_batch)},
                args_names=["X"],
            ).replace_string_by(value="DYN"),
        )
        self.assertEqual(
            {
                "A": {0: "DYN"},
                "B": ({0: "DYN"}, {0: "DYN", 1: "DYN"}),
                "X": {0: "DYN", 1: "DYN"},
            },
            Cls(
                args,
                kwargs,
                {"X": ds_batch_seq, "A": ds_batch, "B": (ds_batch, ds_batch_seq)},
                args_names=["X"],
            ).replace_string_by(value="DYN"),
        )

    def test_couple_input_ds_replace_by_string(self):
        T3x1 = torch.rand((3, 1))
        T3x4 = torch.rand((3, 4))
        T5x1 = torch.rand((5, 1))
        args = (T5x1,)
        kwargs = {"A": T3x4, "B": (T3x1, T3x1)}
        ds_batch = {0: "batch"}
        ds_batch_seq = {0: "batch", 1: "seq"}
        ds = {"X": ds_batch, "A": ds_batch, "B": (ds_batch, ds_batch)}
        Cls = CoupleInputsDynamicShapes
        res = Cls(
            args,
            kwargs,
            ds,
            args_names=["X"],
        ).replace_by_string()
        self.assertEqual(ds, res)

        ds_batch = {0: torch.export.Dim("batch")}
        ds_batch_seq = {0: torch.export.Dim("batch"), 1: torch.export.Dim.DYNAMIC}
        ds = {"X": ds_batch, "A": ds_batch_seq, "B": (ds_batch_seq, ds_batch_seq)}
        res = Cls(
            args,
            kwargs,
            ds,
            args_names=["X"],
        ).replace_by_string()
        self.assertEqual(
            {
                "X": {0: "batch"},
                "A": {0: "batch", 1: "Dim1"},
                "B": ({0: "batch", 1: "Dim2"}, {0: "batch", 1: "Dim3"}),
            },
            res,
        )

    def test_couple_input_ds_change_dynamic_dimensions(self):
        T257 = torch.arange(2 * 5 * 7).reshape((2, 5, 7))
        T29 = torch.arange(2 * 9).reshape((2, 9))
        inst = CoupleInputsDynamicShapes(
            (),
            {"A": T257, "B": T29},
            {"A": {0: "batch", 2: "last"}, "B": {0: "batch", 1: "seq"}},
        )
        new_input = inst.change_dynamic_dimensions()
        self.assertEqual((3, 5, 8), new_input["A"].shape)
        self.assertEqual((3, 10), new_input["B"].shape)

    def test_couple_input_ds_change_dynamic_dimensions_fixed(self):
        T257 = torch.arange(2 * 5 * 7).reshape((2, 5, 7))
        T29 = torch.arange(2 * 9).reshape((2, 9))
        inst = CoupleInputsDynamicShapes(
            (),
            {"A": T257, "B": T29},
            {"A": {0: "batch", 2: "last"}, "B": {0: "batch", 1: "seq"}},
        )
        new_input = inst.change_dynamic_dimensions({"seq": 50, "batch": 1})
        self.assertEqual((1, 5, 8), new_input["A"].shape)
        self.assertEqual((1, 50), new_input["B"].shape)

    def test_couple_input_ds_change_dynamic_dimensions_dynamic_cache(self):
        inst = CoupleInputsDynamicShapes(
            (),
            {"A": make_dynamic_cache([(torch.ones((2, 2, 2, 2)), torch.ones((2, 2, 2, 2)))])},
            {"A": [[{0: "batch", 2: "last"}], [{0: "batch", 2: "last"}]]},
        )
        with torch_export_patches(patch_transformers=True):
            new_inputs = inst.change_dynamic_dimensions()
        self.assertIsInstance(new_inputs["A"], transformers.cache_utils.DynamicCache)
        new_inputs_A = CacheKeyValue(new_inputs["A"])
        self.assertEqual((3, 2, 3, 2), new_inputs_A.key_cache[0].shape)
        self.assertEqual((3, 2, 3, 2), new_inputs_A.value_cache[0].shape)

    @requires_transformers("4.51")
    def test_dynamic_cache_replace_by_string(self):
        n_layers = 2
        bsize, nheads, slen, dim = 2, 4, 3, 7
        cache = make_dynamic_cache(
            [
                (torch.randn(bsize, nheads, slen, dim), torch.randn(bsize, nheads, slen, dim))
                for i in range(n_layers)
            ]
        )

        DYN = torch.export.Dim.DYNAMIC
        ds = {
            "cache": [{0: DYN, 1: DYN}, {0: DYN, 1: DYN}, {0: DYN, 1: DYN}, {0: DYN, 1: DYN}]
        }
        inst = CoupleInputsDynamicShapes((), dict(cache=cache), ds)
        as_string = inst.replace_by_string()
        self.assertEqual(
            {
                "cache": [
                    {0: "Dim0", 1: "Dim1"},
                    {0: "Dim2", 1: "Dim3"},
                    {0: "Dim4", 1: "Dim5"},
                    {0: "Dim6", 1: "Dim7"},
                ]
            },
            as_string,
        )

    @requires_transformers("4.51")
    def test_unbatch_inputs(self):
        data = get_untrained_model_with_inputs("arnir0/Tiny-LLM")
        cpl = CoupleInputsDynamicShapes(
            None, data["inputs"], dynamic_shapes=data["dynamic_shapes"]
        )
        new_dims = cpl.change_dynamic_dimensions(
            desired_values=dict(batch=1), only_desired=True
        )
        s = self.string_type(new_dims, with_shape=True)
        self.assertEqual(
            "dict(input_ids:T7s1x3,attention_mask:T7s1x33,position_ids:T7s1x3,"
            "past_key_values:DynamicCache("
            "key_cache=#1[T1s1x1x30x96], value_cache=#1[T1s1x1x30x96]))",
            s,
        )

    def test_guess_dynamic_cache_without_patches(self):
        n_layers = 2
        bsize, nheads, slen, dim = 2, 4, 3, 7
        cache = make_dynamic_cache(
            [
                (torch.randn(bsize, nheads, slen, dim), torch.randn(bsize, nheads, slen, dim))
                for i in range(n_layers)
            ]
        )
        z = torch.randn((1, 1, 1, 7))
        cache2 = make_dynamic_cache(
            [
                (
                    torch.randn(bsize + 1, nheads, slen + 1, dim + 1),
                    torch.randn(bsize + 1, nheads, slen + 1, dim + 1),
                )
                for i in range(n_layers)
            ]
        )
        inputs = [
            (cache, z),
            (cache2, torch.randn((1, 1, 1, 8))),
        ]

        class Model(torch.nn.Module):
            def forward(self, cache, z):
                cache = CacheKeyValue(cache)
                return (
                    z
                    + cache.key_cache[0]
                    + cache.key_cache[1]
                    + cache.value_cache[0]
                    + cache.value_cache[1]
                )

        mi = ModelInputs(Model(), inputs)
        ds = mi.guess_dynamic_shapes()
        DYN = torch.export.Dim.DYNAMIC
        self.assertEqual(
            (
                (
                    [
                        {0: DYN, 2: DYN, 3: DYN},
                        {0: DYN, 2: DYN, 3: DYN},
                        {0: DYN, 2: DYN, 3: DYN},
                        {0: DYN, 2: DYN, 3: DYN},
                    ],
                    {3: DYN},
                ),
                {},
            ),
            ds,
        )

    def test_invalid_dimensions_for_export(self):
        ags = []
        kws = dict(
            input_ids=torch.randint(0, 10, (2, 3)),
            attention_mask=torch.randint(0, 1, (2, 33)),
            position_ids=torch.randint(0, 10, (2, 3)),
            past_key_values=make_dynamic_cache(
                [torch.rand((2, 1, 30, 96)), torch.rand((2, 1, 30, 96))]
            ),
        )
        ds = dict(
            input_ids={0: "batch", 1: "seq_length"},
            attention_mask={0: "batch", 1: "seq_length"},
            position_ids={0: "batch", 1: "seq_length"},
            past_key_values=[{0: "batch", 2: "cache_length"}, {0: "batch", 2: "cache_length"}],
        )
        with torch_export_patches(patch_transformers=True):
            cpl = CoupleInputsDynamicShapes(ags, kws, ds)
            backed_size_oblivious = cpl.invalid_dimensions_for_export()
            self.assertFalse(backed_size_oblivious)


if __name__ == "__main__":
    unittest.main(verbosity=2)
