import unittest
import torch
from onnx_diagnostic.ext_test_case import ExtTestCase, ignore_warnings
from onnx_diagnostic.helpers.cache_helper import make_encoder_decoder_cache, make_dynamic_cache
from onnx_diagnostic.torch_export_patches.onnx_export_errors import (
    bypass_export_some_errors,
)


class TestPatchSerialization(ExtTestCase):
    @ignore_warnings(UserWarning)
    def test_flatten_encoder_decoder_cache(self):
        cache = make_encoder_decoder_cache(
            make_dynamic_cache([(torch.rand((4, 4, 4)), torch.rand((4, 4, 4)))]),
            make_dynamic_cache([(torch.rand((5, 5, 5)), torch.rand((5, 5, 5)))]),
        )
        with bypass_export_some_errors():
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
    def test_export_encoder_decoder_cache(self):
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

        with bypass_export_some_errors():
            torch.export.export(model, (cache,), dynamic_shapes=(ds,))

    @ignore_warnings(UserWarning)
    def test_flatten_dynamic_cache(self):
        cache = make_dynamic_cache([(torch.rand((4, 4, 4)), torch.rand((4, 4, 4)))])
        with bypass_export_some_errors():
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
    def test_export_dynamic_cache(self):
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

        with bypass_export_some_errors():
            torch.export.export(model, (cache,), dynamic_shapes=(ds,))


if __name__ == "__main__":
    unittest.main(verbosity=2)
