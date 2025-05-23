import unittest
import torch
from onnx_diagnostic.ext_test_case import ExtTestCase, requires_transformers
from onnx_diagnostic.helpers import string_type
from onnx_diagnostic.helpers.cache_helper import (
    make_dynamic_cache,
    flatten_unflatten_for_dynamic_shapes,
)
from onnx_diagnostic.export import ModelInputs
from onnx_diagnostic.torch_export_patches import torch_export_patches


class TestSerialization(ExtTestCase):
    def _get_cache(self, n_layers=2, bsize=2, nheads=4, slen=1, dim=7):
        return make_dynamic_cache(
            [
                (torch.randn(bsize, nheads, slen, dim), torch.randn(bsize, nheads, slen, dim))
                for i in range(n_layers)
            ]
        )

    @requires_transformers("4.50")
    def test_dynamic_cache(self):
        class Model(torch.nn.Module):
            def forward(self, cache):
                return cache.key_cache[0]

        cache = self._get_cache()
        DYN = torch.export.Dim.DYNAMIC
        ds = {0: DYN, 1: DYN, 3: DYN}
        dynamic_shapes = ([[ds, ds], [ds, ds]],)
        exp = torch.export.export(Model(), (cache,), dynamic_shapes=dynamic_shapes)
        self.assertNotEmpty(exp)

    @requires_transformers("4.50")
    def test_dynamic_cache_flat_unflat(self):
        class Model(torch.nn.Module):
            def forward(self, cache):
                return cache.key_cache[0]

        cache = self._get_cache()
        flat_unflat = flatten_unflatten_for_dynamic_shapes(cache)
        s = string_type(flat_unflat, with_shape=True)
        self.assertEqual("#2[#2[T1s2x4x1x7,T1s2x4x1x7],#2[T1s2x4x1x7,T1s2x4x1x7]]", s)

    def test_dynamic_cache_bypass(self):
        class Model(torch.nn.Module):
            def forward(self, cache):
                return cache.key_cache[0]

        cache = self._get_cache()
        with torch_export_patches(patch_transformers=True):
            flat_unflat = flatten_unflatten_for_dynamic_shapes(cache)
            s = string_type(flat_unflat, with_shape=True)
            self.assertEqual("#2[#2[T1s2x4x1x7,T1s2x4x1x7],#2[T1s2x4x1x7,T1s2x4x1x7]]", s)

    def test_dynamic_cache_guess_static(self):
        class Model(torch.nn.Module):
            def forward(self, cache):
                return cache.key_cache[0]

        cache = self._get_cache()
        md = ModelInputs(Model(), [(cache,)])
        guessed = md.guess_dynamic_shapes()
        self.assertEqual(guessed, (([[{}, {}], [{}, {}]],), {}))

    def test_dynamic_cache_guess_auto(self):
        class Model(torch.nn.Module):
            def forward(self, cache):
                return cache.key_cache[0]

        cache = self._get_cache()
        md = ModelInputs(Model(), [(cache,)])
        guessed = md.guess_dynamic_shapes(auto=True)
        AUTO = torch.export.Dim.AUTO
        ds = {i: AUTO for i in range(4)}  # noqa: C420
        self.assertEqual(guessed, (([[ds, ds], [ds, ds]],), {}))

    def test_dynamic_cache_guess_dynamic(self):
        class Model(torch.nn.Module):
            def forward(self, cache):
                return cache.key_cache[0]

        md = ModelInputs(
            Model(), [(self._get_cache(),), (self._get_cache(bsize=3, nheads=5),)]
        )
        guessed = md.guess_dynamic_shapes()
        DYN = torch.export.Dim.DYNAMIC
        self.assertEqual(
            guessed,
            (
                (
                    [
                        [{0: DYN, 1: DYN}, {0: DYN, 1: DYN}],
                        [{0: DYN, 1: DYN}, {0: DYN, 1: DYN}],
                    ],
                ),
                {},
            ),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
