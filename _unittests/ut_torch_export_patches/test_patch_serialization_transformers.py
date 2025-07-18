import unittest
import torch
from transformers.modeling_outputs import BaseModelOutput
from onnx_diagnostic.ext_test_case import ExtTestCase, ignore_warnings, requires_torch
from onnx_diagnostic.helpers.cache_helper import (
    make_encoder_decoder_cache,
    make_dynamic_cache,
    make_static_cache,
    make_sliding_window_cache,
    flatten_unflatten_for_dynamic_shapes,
)
from onnx_diagnostic.torch_export_patches.onnx_export_errors import (
    torch_export_patches,
)
from onnx_diagnostic.helpers.torch_helper import torch_deepcopy


class TestPatchSerialization(ExtTestCase):
    @ignore_warnings(UserWarning)
    def test_encoder_decoder_cache_flatten(self):
        cache = make_encoder_decoder_cache(
            make_dynamic_cache([(torch.rand((4, 4, 4)), torch.rand((4, 4, 4)))]),
            make_dynamic_cache([(torch.rand((5, 5, 5)), torch.rand((5, 5, 5)))]),
        )
        with torch_export_patches():
            flat, _spec = torch.utils._pytree.tree_flatten(cache)
            self.assertEqual(
                "#4[T1s4x4x4,T1s4x4x4,T1s5x5x5,T1s5x5x5]",
                self.string_type(flat, with_shape=True),
            )
            cache2 = torch.utils._pytree.tree_unflatten(flat, _spec)
            self.assertEqual(
                self.string_type(cache, with_shape=True, with_min_max=True),
                self.string_type(cache2, with_shape=True, with_min_max=True),
            )

    @ignore_warnings(UserWarning)
    def test_encoder_decoder_cache_deepcopy(self):
        cache = make_encoder_decoder_cache(
            make_dynamic_cache([(torch.rand((4, 4, 4)), torch.rand((4, 4, 4)))]),
            make_dynamic_cache([(torch.rand((5, 5, 5)), torch.rand((5, 5, 5)))]),
        )
        with torch_export_patches():
            cache2 = torch_deepcopy([cache])
            self.assertEqualAny([cache], cache2)

    @ignore_warnings(UserWarning)
    def test_encoder_decoder_cache_export(self):
        class Model(torch.nn.Module):
            def forward(self, cache):
                return cache.self_attention_cache.key_cache[0]

        cache1 = make_dynamic_cache(
            [(torch.randn(2, 4, 3, 7), torch.randn(2, 4, 3, 7)) for i in range(3)]
        )
        cache2 = make_dynamic_cache(
            [(torch.randn(2, 4, 3, 7), torch.randn(2, 4, 3, 7)) for i in range(3)]
        )

        cache = make_encoder_decoder_cache(cache1, cache2)
        model = Model()
        model(cache)
        DYN = torch.export.Dim.DYNAMIC
        ds = [
            [[{0: DYN}, {0: DYN}, {0: DYN}], [{0: DYN}, {0: DYN}, {0: DYN}]],
            [[{0: DYN}, {0: DYN}, {0: DYN}], [{0: DYN}, {0: DYN}, {0: DYN}]],
        ]

        with torch_export_patches(patch_transformers=True):
            torch.export.export(model, (cache,), dynamic_shapes=(ds,))

    @ignore_warnings(UserWarning)
    def test_dynamic_cache_flatten(self):
        cache = make_dynamic_cache([(torch.rand((4, 4, 4)), torch.rand((4, 4, 4)))])
        with torch_export_patches():
            flat, _spec = torch.utils._pytree.tree_flatten(cache)
            self.assertEqual(
                "#2[T1s4x4x4,T1s4x4x4]",
                self.string_type(flat, with_shape=True),
            )
            cache2 = torch.utils._pytree.tree_unflatten(flat, _spec)
            self.assertEqual(
                self.string_type(cache, with_shape=True, with_min_max=True),
                self.string_type(cache2, with_shape=True, with_min_max=True),
            )

    @ignore_warnings(UserWarning)
    def test_dynamic_cache_export(self):
        class Model(torch.nn.Module):
            def forward(self, cache):
                return cache.key_cache[0]

        cache = make_dynamic_cache(
            [(torch.randn(2, 4, 3, 7), torch.randn(2, 4, 3, 7)) for i in range(3)]
        )
        model = Model()
        model(cache)
        DYN = torch.export.Dim.DYNAMIC
        ds = [[{0: DYN}, {0: DYN}, {0: DYN}], [{0: DYN}, {0: DYN}, {0: DYN}]]

        with torch_export_patches():
            torch.export.export(model, (cache,), dynamic_shapes=(ds,))

    @ignore_warnings(UserWarning)
    def test_dynamic_cache_deepcopy(self):
        cache = make_dynamic_cache([(torch.rand((4, 4, 4)), torch.rand((4, 4, 4)))])
        with torch_export_patches():
            cache2 = torch_deepcopy([cache])
            self.assertEqualAny([cache], cache2)

    @ignore_warnings(UserWarning)
    def test_base_model_output_deepcopy(self):
        bo = BaseModelOutput(last_hidden_state=torch.rand((4, 4, 4)))
        self.assertEqual(bo.__class__.__name__, "BaseModelOutput")
        with torch_export_patches():
            bo2 = torch_deepcopy([bo])
            self.assertIsInstance(bo2, list)
            self.assertEqual(bo2[0].__class__.__name__, "BaseModelOutput")
            self.assertEqualAny([bo], bo2)

    @ignore_warnings(UserWarning)
    def test_base_model_output_string_type(self):
        bo = BaseModelOutput(last_hidden_state=torch.rand((4, 4, 4)))
        with torch_export_patches():
            self.assertEqual(
                "BaseModelOutput(last_hidden_state:T1s4x4x4)",
                self.string_type(bo, with_shape=True),
            )

    @ignore_warnings(UserWarning)
    def test_base_model_output_flatten(self):
        bo = BaseModelOutput(last_hidden_state=torch.rand((4, 4, 4)))
        with torch_export_patches():
            flat, _spec = torch.utils._pytree.tree_flatten(bo)
            self.assertEqual(
                "#1[T1s4x4x4]",
                self.string_type(flat, with_shape=True),
            )
            bo2 = torch.utils._pytree.tree_unflatten(flat, _spec)
            self.assertEqual(
                self.string_type(bo, with_shape=True, with_min_max=True),
                self.string_type(bo2, with_shape=True, with_min_max=True),
            )

    @ignore_warnings(UserWarning)
    def test_base_model_output_export(self):
        class Model(torch.nn.Module):
            def forward(self, cache):
                return cache.last_hidden_state[0]

        bo = BaseModelOutput(last_hidden_state=torch.rand((4, 4, 4)))
        model = Model()
        model(bo)
        DYN = torch.export.Dim.DYNAMIC
        ds = [{0: DYN}]

        with torch_export_patches(patch_transformers=True):
            torch.export.export(model, (bo,), dynamic_shapes=(ds,))

    @ignore_warnings(UserWarning)
    def test_base_model_output_unflatten_flatten(self):
        bo = BaseModelOutput(last_hidden_state=torch.rand((4, 4, 4)))
        with torch_export_patches(patch_transformers=True):
            flat, _spec = torch.utils._pytree.tree_flatten(bo)
            unflat = flatten_unflatten_for_dynamic_shapes(bo, use_dict=True)
            self.assertIsInstance(unflat, list)
            self.assertEqual("#1[T1r3]", self.string_type(unflat))

    @ignore_warnings(UserWarning)
    def test_base_sliding_window_cache_unflatten_flatten(self):
        cache = make_sliding_window_cache(
            [(torch.rand((4, 4, 4, 4)), torch.rand((4, 4, 4, 4)))]
        )
        with torch_export_patches():
            cache2 = torch_deepcopy([cache])
            self.assertEqualAny([cache], cache2)

    @ignore_warnings(UserWarning)
    @requires_torch("2.7.99")
    def test_sliding_window_cache_export(self):
        class Model(torch.nn.Module):
            def forward(self, cache):
                return cache.key_cache[0]

        cache = make_sliding_window_cache(
            [
                (torch.rand((4, 4, 4, 4)), torch.rand((4, 4, 4, 4))),
                (torch.rand((4, 4, 4, 4)), torch.rand((4, 4, 4, 4))),
            ]
        )
        model = Model()
        model(cache)
        DYN = torch.export.Dim.DYNAMIC
        ds = [[{0: DYN}, {0: DYN}], [{0: DYN}, {0: DYN}]]

        with torch_export_patches(patch_transformers=True):
            torch.export.export(model, (cache,), dynamic_shapes=(ds,))

    @ignore_warnings(UserWarning)
    def test_sliding_window_cache_flatten(self):
        cache = make_sliding_window_cache(
            [(torch.rand((4, 4, 4, 4)), torch.rand((4, 4, 4, 4)))]
        )
        with torch_export_patches():
            flat, _spec = torch.utils._pytree.tree_flatten(cache)
            self.assertEqual(
                "#2[T1s4x4x4x4,T1s4x4x4x4]",
                self.string_type(flat, with_shape=True),
            )
            cache2 = torch.utils._pytree.tree_unflatten(flat, _spec)
            self.assertEqual(
                self.string_type(cache, with_shape=True, with_min_max=True),
                self.string_type(cache2, with_shape=True, with_min_max=True),
            )

    @ignore_warnings(UserWarning)
    @requires_torch("2.7.99")
    def test_static_cache(self):
        bo = make_static_cache(
            [
                (torch.rand((4, 5, 6, 7)), torch.rand((4, 5, 6, 7))),
                (torch.rand((4, 5, 6, 7)), torch.rand((4, 5, 6, 7))),
                (torch.rand((4, 5, 6, 7)), torch.rand((4, 5, 6, 7))),
            ],
            max_cache_len=15,
        )
        self.assertEqual(bo.__class__.__name__, "StaticCache")
        bo2 = torch_deepcopy([bo])
        self.assertIsInstance(bo2, list)
        self.assertEqual(
            "StaticCache(key_cache=#3[T1s4x5x15x7,T1s4x5x15x7,T1s4x5x15x7], "
            "value_cache=#3[T1s4x5x15x7,T1s4x5x15x7,T1s4x5x15x7])",
            self.string_type(bo, with_shape=True),
        )

        with torch_export_patches():
            # internal function
            bo2 = torch_deepcopy([bo])
            self.assertIsInstance(bo2, list)
            self.assertEqual(bo2[0].__class__.__name__, "StaticCache")
            self.assertEqualAny([bo], bo2)
            self.assertEqual(
                "StaticCache(key_cache=#3[T1s4x5x15x7,T1s4x5x15x7,T1s4x5x15x7], "
                "value_cache=#3[T1s4x5x15x7,T1s4x5x15x7,T1s4x5x15x7])",
                self.string_type(bo, with_shape=True),
            )

            # serialization
            flat, _spec = torch.utils._pytree.tree_flatten(bo)
            self.assertEqual(
                "#6[T1s4x5x15x7,T1s4x5x15x7,T1s4x5x15x7,T1s4x5x15x7,T1s4x5x15x7,T1s4x5x15x7]",
                self.string_type(flat, with_shape=True),
            )
            bo2 = torch.utils._pytree.tree_unflatten(flat, _spec)
            self.assertEqual(
                self.string_type(bo, with_shape=True, with_min_max=True),
                self.string_type(bo2, with_shape=True, with_min_max=True),
            )

            # flatten_unflatten
            flat, _spec = torch.utils._pytree.tree_flatten(bo)
            unflat = flatten_unflatten_for_dynamic_shapes(bo, use_dict=True)
            self.assertIsInstance(unflat, list)
            self.assertEqual(
                "#2[#3[T1r4,T1r4,T1r4],#3[T1r4,T1r4,T1r4]]", self.string_type(unflat)
            )

        # export
        class Model(torch.nn.Module):
            def forward(self, cache):
                return cache.key_cache[0]

        model = Model()
        model(bo)
        DYN = torch.export.Dim.DYNAMIC
        ds = [[{0: DYN}, {0: DYN}, {0: DYN}], [{0: DYN}, {0: DYN}, {0: DYN}]]

        with torch_export_patches(patch_transformers=True, stop_if_static=1):
            torch.export.export(model, (bo,), dynamic_shapes=(ds,))


if __name__ == "__main__":
    unittest.main(verbosity=2)
