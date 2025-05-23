import unittest
from typing import Callable
import torch
from torch._dynamo._trace_wrapped_higher_order_op import TransformGetItemToIndex
from onnx_diagnostic.ext_test_case import ExtTestCase, requires_torch
from onnx_diagnostic.torch_export_patches.patches.patch_torch import patched_vmap
from onnx_diagnostic.torch_export_patches.patches.patch_transformers import (
    patched__vmap_for_bhqkv as _vmap_for_bhqkv2,
)


class TestPatchPatchTorch(ExtTestCase):
    def test_vmap(self):
        f = lambda x, y: x * y + 1  # noqa: E731
        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.tensor([0.1, 0.2, 0.3])
        expected = torch.vmap(f)(x, y)
        got = patched_vmap(f)(x, y)
        self.assertEqualArray(expected, got)

    @requires_torch("2.9")
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
    def test_export_patched_vmap(self):
        class Model(torch.nn.Module):
            def forward(self, x, y):
                f = lambda x, y: x * y + 1  # noqa: E731
                return patched_vmap(f)(x, y)

        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.tensor([0.1, 0.2, 0.3])
        ep = torch.export.export(Model(), (x, y))
        self.assertEqualArray(Model()(x, y), ep.module()(x, y))

    def test_vmap_outdim(self):
        f = lambda x: x**2  # noqa: E731
        x = torch.randn(2, 5)
        expected = torch.vmap(f, out_dims=1)(x)
        got = patched_vmap(f, out_dims=1)(x)
        self.assertEqualArray(expected, got)

    def test_vmap_dict(self):
        f = lambda d: torch.dot(d["x"], d["y"])  # noqa: E731
        x, y = torch.randn(2, 5), torch.randn(5)
        input = {"x": x, "y": y}
        _expected = torch.vmap(f, in_dims=({"x": 0, "y": None},))(input)
        self.assertRaise(
            lambda: patched_vmap(f, in_dims=({"x": 0, "y": None},))(input), AssertionError
        )
        # self.assertEqualArray(_expected, got)

    def test_vmap_tuple(self):
        x, y = torch.randn(2, 5), torch.randn(5)
        expected = torch.vmap(torch.dot, in_dims=(0, None))(x, y)
        got = patched_vmap(torch.dot, in_dims=(0, None))(x, y)
        self.assertEqualArray(expected, got, atol=1e-5)

    def test_vmap_transformers_scenario_vmap(self):
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

    def test_vmap_transformers_scenario_novmap(self):
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
                with TransformGetItemToIndex():
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
