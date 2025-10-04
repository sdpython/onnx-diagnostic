import unittest
from typing import Callable
import torch
from torch._dynamo._trace_wrapped_higher_order_op import TransformGetItemToIndex
from onnx_diagnostic.ext_test_case import (
    ExtTestCase,
    requires_torch,
    requires_transformers,
    has_transformers,
    has_torch,
)
from onnx_diagnostic.helpers.cache_helper import CacheKeyValue, make_dynamic_cache
from onnx_diagnostic.helpers.torch_helper import torch_deepcopy
from onnx_diagnostic.torch_models.hghub import get_untrained_model_with_inputs
from onnx_diagnostic.torch_export_patches import torch_export_patches
from onnx_diagnostic.torch_export_patches.patch_inputs import use_dyn_not_str


class TestPatchPatchTorch(ExtTestCase):
    @requires_transformers("4.52")
    def test_vmap(self):
        from onnx_diagnostic.torch_export_patches.patches.patch_torch import patched_vmap

        f = lambda x, y: x * y + 1  # noqa: E731
        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.tensor([0.1, 0.2, 0.3])
        expected = torch.vmap(f)(x, y)
        got = patched_vmap(f)(x, y)
        self.assertEqualArray(expected, got)

    @requires_torch("2.10")
    def test_export_vmap(self):
        class Model(torch.nn.Module):
            def forward(self, x, y):
                f = lambda x, y: x * y + 1  # noqa: E731
                return torch.vmap(f)(x, y)

        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.tensor([0.1, 0.2, 0.3])
        DYN = torch.export.Dim.DYNAMIC
        ep = torch.export.export(Model(), (x, y), ({0: DYN}, {1: DYN}))
        self.assertEqualArray(Model()(x, y), ep.module()(x, y))

    @requires_torch("2.8")
    @requires_transformers("4.52")
    def test_export_patched_vmap(self):
        from onnx_diagnostic.torch_export_patches.patches.patch_torch import patched_vmap

        class Model(torch.nn.Module):
            def forward(self, x, y):
                f = lambda x, y: x * y + 1  # noqa: E731
                return patched_vmap(f)(x, y)

        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.tensor([0.1, 0.2, 0.3])
        ep = torch.export.export(Model(), (x, y))
        self.assertEqualArray(Model()(x, y), ep.module()(x, y))

    @requires_transformers("4.52")
    def test_vmap_outdim(self):
        from onnx_diagnostic.torch_export_patches.patches.patch_torch import patched_vmap

        f = lambda x: x**2  # noqa: E731
        x = torch.randn(2, 5)
        expected = torch.vmap(f, out_dims=1)(x)
        got = patched_vmap(f, out_dims=1)(x)
        self.assertEqualArray(expected, got)

    @requires_transformers("4.52")
    def test_vmap_dict(self):
        from onnx_diagnostic.torch_export_patches.patches.patch_torch import patched_vmap

        f = lambda d: torch.dot(d["x"], d["y"])  # noqa: E731
        x, y = torch.randn(2, 5), torch.randn(5)
        input = {"x": x, "y": y}
        _expected = torch.vmap(f, in_dims=({"x": 0, "y": None},))(input)
        self.assertRaise(
            lambda: patched_vmap(f, in_dims=({"x": 0, "y": None},))(input), AssertionError
        )
        # self.assertEqualArray(_expected, got)

    @requires_transformers("4.52")
    def test_vmap_tuple(self):
        from onnx_diagnostic.torch_export_patches.patches.patch_torch import patched_vmap

        x, y = torch.randn(2, 5), torch.randn(5)
        expected = torch.vmap(torch.dot, in_dims=(0, None))(x, y)
        got = patched_vmap(torch.dot, in_dims=(0, None))(x, y)
        self.assertEqualArray(expected, got, atol=1e-5)

    @requires_transformers("4.52")
    def test_vmap_transformers_scenario_vmap(self):
        from onnx_diagnostic.torch_export_patches.patches.patch_torch import patched_vmap

        def padding_mask_function(padding_mask: torch.Tensor) -> Callable:
            def inner_mask(batch_idx, head_idx, q_idx, kv_idx):
                return padding_mask[batch_idx, kv_idx]

            return inner_mask

        def and_masks(*mask_functions: list[Callable]) -> Callable:
            def and_mask(batch_idx, head_idx, q_idx, kv_idx):
                result = q_idx.new_ones((), dtype=torch.bool)
                for mask in mask_functions:
                    result = result & mask(batch_idx, head_idx, q_idx, kv_idx)
                return result

            return and_mask

        def causal_mask_function(
            batch_idx: int, head_idx: int, q_idx: int, kv_idx: int
        ) -> bool:
            return kv_idx <= q_idx

        def _vmap_for_bhqkv(mask_function: Callable, bh_indices: bool = True) -> Callable:
            dimensions = [(None, None, None, 0), (None, None, 0, None)]
            if bh_indices:
                dimensions.extend([(None, 0, None, None), (0, None, None, None)])
            for dims in dimensions:
                mask_function = torch.vmap(mask_function, in_dims=dims, out_dims=0)
            return mask_function

        def _vmap_for_bhqkv2(mask_function: Callable, bh_indices: bool = True) -> Callable:
            dimensions = [(None, None, None, 0), (None, None, 0, None)]
            if bh_indices:
                dimensions.extend([(None, 0, None, None), (0, None, None, None)])
            for dims in dimensions:
                mask_function = patched_vmap(mask_function, in_dims=dims, out_dims=0)
            return mask_function

        padding_mask = torch.ones((2, 33)).to(torch.bool)
        batch_arange = torch.tensor([0, 1], dtype=torch.int64)
        head_arange = torch.tensor([0, 1], dtype=torch.int64)
        cache_position = torch.tensor([30, 31, 32], dtype=torch.int64)
        kv_arange = torch.arange(33, dtype=torch.int64)
        mask_function = and_masks(causal_mask_function, padding_mask_function(padding_mask))
        with TransformGetItemToIndex():
            causal_mask = _vmap_for_bhqkv(mask_function)(
                batch_arange, head_arange, cache_position, kv_arange
            )
        with TransformGetItemToIndex():
            causal_mask2 = _vmap_for_bhqkv2(mask_function)(
                batch_arange, head_arange, cache_position, kv_arange
            )
        self.assertEqualArray(causal_mask, causal_mask2)

        class Model(torch.nn.Module):
            def forward(self, batch_arange, head_arange, cache_position, kv_arange):
                with TransformGetItemToIndex():
                    causal_mask2 = _vmap_for_bhqkv2(mask_function)(
                        batch_arange, head_arange, cache_position, kv_arange
                    )
                return causal_mask2

        inputs = batch_arange, head_arange, cache_position, kv_arange
        got = Model()(*inputs)
        self.assertEqualArray(causal_mask, got)

        if not requires_torch("4.10"):
            DYN = torch.export.Dim.DYNAMIC
            ds1 = {0: DYN}
            ds2 = {0: DYN, 1: DYN}
            ds = (ds2, ds1, ds1, ds1)
            ep = torch.export.export(Model(), inputs, dynamic_shapes=ds)
            self.assertEqualArray(causal_mask, ep.moule(*inputs))

    @requires_torch("2.8")
    @requires_transformers("4.53")
    def test_vmap_transformers_scenario_novmap(self):
        from onnx_diagnostic.torch_export_patches.patches.patch_transformers import (
            patched__vmap_for_bhqkv as _vmap_for_bhqkv2,
        )

        def padding_mask_function(padding_mask: torch.Tensor) -> Callable:
            def inner_mask(batch_idx, head_idx, q_idx, kv_idx):
                return padding_mask[batch_idx, kv_idx]

            return inner_mask

        def and_masks(*mask_functions: list[Callable]) -> Callable:
            def and_mask(batch_idx, head_idx, q_idx, kv_idx):
                result = q_idx.new_ones((), dtype=torch.bool)
                for mask in mask_functions:
                    result = result & mask(batch_idx, head_idx, q_idx, kv_idx)
                return result

            return and_mask

        def causal_mask_function(
            batch_idx: int, head_idx: int, q_idx: int, kv_idx: int
        ) -> bool:
            return kv_idx <= q_idx

        def _vmap_for_bhqkv(mask_function: Callable, bh_indices: bool = True) -> Callable:
            dimensions = [(None, None, None, 0), (None, None, 0, None)]
            if bh_indices:
                dimensions.extend([(None, 0, None, None), (0, None, None, None)])
            for dims in dimensions:
                mask_function = torch.vmap(mask_function, in_dims=dims, out_dims=0)
            return mask_function

        padding_mask = torch.ones((2, 33)).to(torch.bool)
        batch_arange = torch.tensor([0, 1], dtype=torch.int64)
        head_arange = torch.tensor([0, 1], dtype=torch.int64)
        cache_position = torch.tensor([30, 31, 32], dtype=torch.int64)
        kv_arange = torch.arange(33, dtype=torch.int64)
        mask_function = and_masks(causal_mask_function, padding_mask_function(padding_mask))
        with TransformGetItemToIndex():
            causal_mask = _vmap_for_bhqkv(mask_function)(
                batch_arange, head_arange, cache_position, kv_arange
            )
        with TransformGetItemToIndex():
            causal_mask2 = _vmap_for_bhqkv2(mask_function)(
                batch_arange, head_arange, cache_position, kv_arange
            )
        self.assertEqualArray(causal_mask, causal_mask2)

        class Model(torch.nn.Module):
            def forward(self, batch_arange, head_arange, cache_position, kv_arange):
                # with TransformGetItemToIndex():
                # This context as ignored in 2.8 and not any more in 2.9.
                causal_mask2 = _vmap_for_bhqkv2(mask_function)(
                    batch_arange, head_arange, cache_position, kv_arange
                )
                return causal_mask2

        inputs = batch_arange, head_arange, cache_position, kv_arange
        got = Model()(*inputs)
        self.assertEqualArray(causal_mask, got)

        DYN = torch.export.Dim.DYNAMIC
        ds1 = {0: DYN}
        ds2 = {0: DYN, 1: DYN}
        ds = (ds2, ds1, ds1, ds1)
        ep = torch.export.export(Model(), inputs, dynamic_shapes=ds)
        self.assertEqualArray(causal_mask, ep.module()(*inputs))

    @requires_torch("2.7")
    def test_export_unsqueeze(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return x.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        x = torch.tensor([7.0, 8.0])
        Model()(x)
        DYN = torch.export.Dim.DYNAMIC
        ep = torch.export.export(Model(), (x,), dynamic_shapes=({0: DYN},))
        self.assertEqualArray(Model()(x), ep.module()(x))

    def test_oblivious_for_dimension_01(self):
        class Model(torch.nn.Module):
            def forward(self, x, ind1, ind2):
                return x[ind1, ind2]

        inputs = (
            torch.randn(2, 1024),
            torch.tensor([[0, 1]], dtype=torch.int64).T,
            torch.arange(1024, dtype=torch.int64),
        )
        model = Model()
        expected = model(*inputs)

        dynamic_string = ({0: "A", 1: "B"}, {0: "C", 1: "D"}, {0: "E"})
        # ({0: DYN, 1: DYN}, {0: DYN, 1: DYN}, {0: DYN})

        dynamic_shapes = use_dyn_not_str(dynamic_string)
        with self.subTest(
            name="export 0/1 specialized due to hint of 1 for dimension",
            dynamic_shapes=dynamic_shapes,
        ):
            try:
                torch.export.export(model, inputs, dynamic_shapes=dynamic_shapes)
                raise AssertionError("torch fixed that case")
            except ValueError as e:
                self.assertIn("export 0/1 specialized due to hint of 1 for dimension", str(e))

        dynamic_shapes = use_dyn_not_str(dynamic_string, torch.export.Dim.AUTO)
        if has_torch("2.9"):
            with self.subTest(
                name="expected shape should be broadcastable to (>= 2.9)",
                dynamic_shapes=dynamic_shapes,
            ):
                try:
                    with torch.fx.experimental._config.patch(backed_size_oblivious=True):
                        torch.export.export(model, inputs, dynamic_shapes=dynamic_shapes)
                    raise AssertionError("torch fixed that case")
                except RuntimeError as e:
                    self.assertIn("expected shape should be broadcastable to", str(e))

        if not has_torch("2.9"):
            with self.subTest(
                name="expected shape should be broadcastable to (< 2.9)",
                dynamic_shapes=dynamic_shapes,
            ):
                try:
                    with torch.fx.experimental._config.patch(backed_size_oblivious=True):
                        torch.export.export(model, inputs, dynamic_shapes=dynamic_shapes)
                except RuntimeError as e:
                    self.assertIn(
                        "Expected input at *args[2].shape[0] to be equal to 1, but got 1024",
                        str(e),
                    )

        with self.subTest(name="patch for 0/1", dynamic_shapes=dynamic_shapes):
            with torch_export_patches():
                ep = torch.export.export(model, inputs, dynamic_shapes=dynamic_shapes)
            got = ep.module()(*inputs)
            self.assertEqualArray(expected, got)

        if has_torch("2.11"):
            # Missing PR https://github.com/pytorch/pytorch/pull/164225
            # Needs more thinking about the patch to apply for this particular example.
            with self.subTest(
                name="patch for 0/1 with oblivious", dynamic_shapes=dynamic_shapes
            ):
                with (
                    torch_export_patches(),
                    torch.fx.experimental._config.patch(backed_size_oblivious=True),
                ):
                    ep = torch.export.export(model, inputs, dynamic_shapes=dynamic_shapes)
                got = ep.module()(*inputs)
                self.assertEqualArray(expected, got)

    def test_patched__broadcast_in_dim_meta(self):
        class Model(torch.nn.Module):
            def forward(self, x, ind1, ind2):
                return x[ind1, ind2]

        inputs = (
            torch.randn(2, 1024),
            torch.tensor([[0, 1]], dtype=torch.int64).T,
            torch.arange(1024, dtype=torch.int64),
        )
        model = Model()
        expected = model(*inputs)

        with (
            torch.fx.experimental._config.patch(backed_size_oblivious=True),
            torch_export_patches(),
        ):
            ep = torch.export.export(
                model,
                inputs,
                dynamic_shapes=use_dyn_not_str(({0: "A", 1: "B"}, {0: "C", 1: "D"}, {0: "E"})),
            )
        self.assertEqualArray(expected, ep.module()(*inputs), atol=1e-2)

    @requires_torch("2.7.9999")
    @requires_transformers("4.49.9999")
    def test_export_with_patch_tiny_llm_dim_meta(self):
        data = get_untrained_model_with_inputs("arnir0/Tiny-LLM", verbose=0)
        model, inputs, ds = data["model"], data["inputs"], data["dynamic_shapes"]
        order = ["input_ids", "attention_mask", "position_ids", "past_key_values"]
        self.assertEqual(list(inputs), order)
        expected = model(**torch_deepcopy(inputs))
        with self.subTest(input="no01", backed_size_oblivious=False):
            with torch_export_patches(patch_transformers=True):
                ep = torch.export.export(
                    model, (), kwargs=inputs, dynamic_shapes=use_dyn_not_str(ds)
                )
            got = ep.module()(**torch_deepcopy(inputs))
            self.assertEqualArrayAny(expected, got)

        with self.subTest(input="no01", backed_size_oblivious=True):
            if not has_transformers("4.55"):
                raise unittest.SkipTest("test not working with transformers<4.55")
            with (
                torch.fx.experimental._config.patch(backed_size_oblivious=True),
                torch_export_patches(patch_transformers=True),
            ):
                ep = torch.export.export(
                    model, (), kwargs=inputs, dynamic_shapes=use_dyn_not_str(ds)
                )
            got = ep.module()(**torch_deepcopy(inputs))
            self.assertEqualArrayAny(expected, got)

        def _batch1(t):
            if t.__class__.__name__ == "DynamicCache":
                kv = CacheKeyValue(t)
                keys = [t[:1] for t in kv.key_cache]
                values = [t[:1] for t in kv.value_cache]
                return make_dynamic_cache(tuple(zip(keys, values)))
            if t.ndim > 1:
                return t[:1]
            return t

        export_inputs = {k: _batch1(v) for k, v in inputs.items()}

        # with self.subTest(input="batch1", backed_size_oblivious=False):
        #    with torch_export_patches(patch_transformers=True):
        #        ep = torch.export.export(
        #            model, (), kwargs=export_inputs, dynamic_shapes=use_dyn_not_str(ds)
        #        )
        #    got = ep.module()(**torch_deepcopy(inputs))
        #    self.assertEqualArrayAny(expected, got)

        with self.subTest(input="batch1", backed_size_oblivious=True):
            with (
                torch.fx.experimental._config.patch(backed_size_oblivious=True),
                torch_export_patches(patch_transformers=True),
            ):
                ep = torch.export.export(
                    model, (), kwargs=export_inputs, dynamic_shapes=use_dyn_not_str(ds)
                )
            try:
                got = ep.module()(**torch_deepcopy(inputs))
            except AssertionError as e:
                got = None
                if "Guard failed: position_ids.size()[0] == 1" not in str(e):
                    raise

            if got is not None:
                self.assertEqualArrayAny(expected, got)

        if "inputs_empty_cache" not in data:
            return

        export_inputs = data["inputs_empty_cache"]

        # with self.subTest(input="cache0", backed_size_oblivious=False):
        #    with torch_export_patches(patch_transformers=True):
        #        ep = torch.export.export(
        #            model, (), kwargs=export_inputs, dynamic_shapes=use_dyn_not_str(ds)
        #        )
        #    got = ep.module()(**torch_deepcopy(inputs))
        #    self.assertEqualArrayAny(expected, got)

        with self.subTest(input="cache0", backed_size_oblivious=True):
            with (
                torch.fx.experimental._config.patch(backed_size_oblivious=True),
                torch_export_patches(patch_transformers=True),
            ):
                ep = torch.export.export(
                    model, (), kwargs=export_inputs, dynamic_shapes=use_dyn_not_str(ds)
                )
            got = ep.module()(**torch_deepcopy(inputs))
            self.assertEqualArrayAny(expected, got)


if __name__ == "__main__":
    unittest.main(verbosity=2)
