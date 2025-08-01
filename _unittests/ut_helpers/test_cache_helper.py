import unittest
import torch
import transformers
from onnx_diagnostic.ext_test_case import ExtTestCase, requires_transformers
from onnx_diagnostic.helpers import string_type, max_diff
from onnx_diagnostic.helpers.cache_helper import (
    flatten_unflatten_for_dynamic_shapes,
    make_dynamic_cache,
    make_encoder_decoder_cache,
    make_hybrid_cache,
    make_mamba_cache,
    make_sliding_window_cache,
    make_static_cache,
)
from onnx_diagnostic.helpers.torch_helper import torch_deepcopy
from onnx_diagnostic.export import CoupleInputsDynamicShapes
from onnx_diagnostic.torch_export_patches.patch_inputs import (
    convert_dynamic_axes_into_dynamic_shapes,
)
from onnx_diagnostic.torch_export_patches import torch_export_patches


class TestCacheHelpers(ExtTestCase):
    def test_string_type(self):
        DYN = torch.export.Dim.DYNAMIC
        self.assertEqual("DYNAMIC", string_type(DYN, verbose=0))
        AUTO = torch.export.Dim.AUTO
        self.assertEqual("AUTO", string_type(AUTO, verbose=0))
        self.assertEqual("#1[DYNAMIC]", string_type([DYN]))

        batch = torch.export.Dim("batch")
        dynamic_shapes = dict(
            input_ids={0: batch, 1: "seq"},
            attention_mask={0: batch, 1: "seq"},
            position_ids={0: batch, 1: "seq"},
            past_key_values=[[{0: batch, 2: "seq"}], [{0: batch, 2: "seq"}]],
        )
        self.assertEqual(
            "dict(input_ids:{0:Dim(batch),1:DYN(seq)},"
            "attention_mask:{0:Dim(batch),1:DYN(seq)},"
            "position_ids:{0:Dim(batch),1:DYN(seq)},"
            "past_key_values:#2[#1[{0:Dim(batch),2:DYN(seq)}],"
            "#1[{0:Dim(batch),2:DYN(seq)}]])",
            string_type(dynamic_shapes),
        )

    def test_replace_by(self):
        bsize, nheads, slen, dim = 2, 4, 3, 7

        past_key_values = make_dynamic_cache(
            [(torch.randn(bsize, nheads, slen, dim), torch.randn(bsize, nheads, slen, dim))]
        )
        self.assertEqual(
            "DynamicCache(key_cache=#1[T1s2x4x3x7], value_cache=#1[T1s2x4x3x7])",
            self.string_type(past_key_values, with_shape=True),
        )
        kwargs = dict(
            input_ids=torch.zeros(2, 3),
            attention_mask=torch.zeros(2, 3),
            position_ids=torch.zeros(2, 3),
            past_key_values=past_key_values,
        )
        batch = torch.export.Dim("batch")
        dynamic_shapes = dict(
            input_ids={0: batch, 1: "seq"},
            attention_mask={0: batch, 1: "seq"},
            position_ids={0: batch, 1: "seq"},
            past_key_values=[[{0: batch, 2: "seq"}], [{0: batch, 2: "seq"}]],
        )

        DYN = torch.export.Dim.DYNAMIC
        nargs, nkwargs, nds = convert_dynamic_axes_into_dynamic_shapes(
            None, args=tuple(), kwargs=kwargs, dynamic_axes=dynamic_shapes
        )
        self.assertEqual(dynamic_shapes, nds)

        with torch_export_patches(patch_transformers=True):
            cpl = CoupleInputsDynamicShapes(tuple(), kwargs, dynamic_shapes)
            res = cpl.replace_string_by()
        dsc = res["past_key_values"]
        self.assertEqual([[{0: batch, 2: DYN}], [{0: batch, 2: DYN}]], dsc)

    def test_unflatten_flatten_dynamic_cache(self):
        with torch_export_patches(patch_transformers=True):
            c1 = make_dynamic_cache([(torch.rand((4, 4, 4)), torch.rand((4, 4, 4)))])
            self.assertIsInstance(c1, transformers.cache_utils.DynamicCache)
            unflat = flatten_unflatten_for_dynamic_shapes(c1)
            self.assertEqual(
                "#2[#1[T1s4x4x4],#1[T1s4x4x4]]", self.string_type(unflat, with_shape=True)
            )
            self.assertEqual(
                "DynamicCache(key_cache=#1[T1s4x4x4], value_cache=#1[T1s4x4x4])",
                self.string_type(c1, with_shape=True),
            )

    def test_unflatten_flatten_encoder_decoder_cache(self):
        with torch_export_patches(patch_transformers=True):
            c2 = make_encoder_decoder_cache(
                make_dynamic_cache(
                    [
                        (torch.rand((4, 4, 4)), torch.rand((4, 4, 4))),
                        (torch.rand((4, 4, 4)), torch.rand((4, 4, 4))),
                        (torch.rand((4, 4, 4)), torch.rand((4, 4, 4))),
                    ]
                ),
                make_dynamic_cache(
                    [
                        (torch.rand((5, 5, 5)), torch.rand((5, 5, 5))),
                        (torch.rand((5, 5, 5)), torch.rand((5, 5, 5))),
                        (torch.rand((5, 5, 5)), torch.rand((5, 5, 5))),
                    ]
                ),
            )
            self.assertEqual(0, max_diff(c2, c2)["abs"])
            self.assertIsInstance(c2, transformers.cache_utils.EncoderDecoderCache)
            flat, _spec = torch.utils._pytree.tree_flatten(c2)
            self.assertIsInstance(flat, list)
            self.assertEqual(len(flat), 12)
            self.assertIsInstance(flat[0], torch.Tensor)
            unflat = flatten_unflatten_for_dynamic_shapes(c2)
            self.assertIsInstance(unflat, list)
            self.assertEqual(len(unflat), 2)
            self.assertIsInstance(unflat[0], list)
            self.assertEqual(len(unflat[0]), 2)
            self.assertIsInstance(unflat[0][0], list)
            self.assertEqual(len(unflat[0][0]), 3)
            self.assertEqual(
                "#2[#3[T1s4x4x4,T1s4x4x4,T1s4x4x4],#3[T1s4x4x4,T1s4x4x4,T1s4x4x4]]",
                self.string_type(unflat[0], with_shape=True),
            )
            self.assertEqual(
                "#2[#2[#3[T1s4x4x4,T1s4x4x4,T1s4x4x4],#3[T1s4x4x4,T1s4x4x4,T1s4x4x4]],"
                "#2[#3[T1s5x5x5,T1s5x5x5,T1s5x5x5],#3[T1s5x5x5,T1s5x5x5,T1s5x5x5]]]",
                self.string_type(unflat, with_shape=True),
            )
            self.assertEqual(
                "EncoderDecoderCache(self_attention_cache=DynamicCache("
                "key_cache=#3[T1s4x4x4,T1s4x4x4,T1s4x4x4], value_cache=#3"
                "[T1s4x4x4,T1s4x4x4,T1s4x4x4]), cross_attention_cache=DynamicCache"
                "(key_cache=#3[T1s5x5x5,T1s5x5x5,T1s5x5x5], value_cache=#3"
                "[T1s5x5x5,T1s5x5x5,T1s5x5x5]))",
                self.string_type(c2, with_shape=True),
            )

    @requires_transformers("4.51")  # the structure changes
    def test_make_mamba_cache(self):
        cache = make_mamba_cache(
            [
                (torch.rand((4, 4, 4)), torch.rand((4, 4, 4))),
                (torch.rand((4, 4, 4)), torch.rand((4, 4, 4))),
                (torch.rand((4, 4, 4)), torch.rand((4, 4, 4))),
            ]
        )
        text = self.string_type(cache, with_shape=True)
        self.assertEqual(
            "MambaCache(conv_states=#3[T1s4x4x4,T1s4x4x4,T1s4x4x4], "
            "ssm_states=#3[T1s4x4x4,T1s4x4x4,T1s4x4x4])",
            text,
        )
        self.assertEqual(0, max_diff(cache, cache)["abs"])

    def test_make_sliding_window_cache(self):
        cache = make_sliding_window_cache(
            [
                (torch.rand((4, 5, 6, 7)), torch.rand((4, 5, 6, 7))),
                (torch.rand((4, 5, 6, 7)), torch.rand((4, 5, 6, 7))),
                (torch.rand((4, 5, 6, 7)), torch.rand((4, 5, 6, 7))),
            ]
        )
        text = self.string_type(cache, with_shape=True)
        self.assertEqual(
            "SlidingWindowCache(key_cache=#3[T1s4x5x6x7,T1s4x5x6x7,T1s4x5x6x7], "
            "value_cache=#3[T1s4x5x6x7,T1s4x5x6x7,T1s4x5x6x7])",
            text,
        )
        self.assertEqual(0, max_diff(cache, cache)["abs"])

    def test_make_static_cache(self):
        cache = make_static_cache(
            [
                (torch.rand((4, 5, 6, 7)), torch.rand((4, 5, 6, 7))),
                (torch.rand((4, 5, 6, 7)), torch.rand((4, 5, 6, 7))),
                (torch.rand((4, 5, 6, 7)), torch.rand((4, 5, 6, 7))),
            ],
            max_cache_len=15,
        )
        text = self.string_type(cache, with_shape=True)
        self.assertEqual(
            "StaticCache(key_cache=#3[T1s4x5x15x7,T1s4x5x15x7,T1s4x5x15x7], "
            "value_cache=#3[T1s4x5x15x7,T1s4x5x15x7,T1s4x5x15x7])",
            text,
        )
        self.assertEqual(0, max_diff(cache, cache)["abs"])

    def test_unflatten_flatten_static_cache(self):
        with torch_export_patches(patch_transformers=True):
            c2 = make_static_cache(
                [
                    (torch.rand((4, 5, 6, 7)), torch.rand((4, 5, 6, 7))),
                    (torch.rand((4, 5, 6, 7)), torch.rand((4, 5, 6, 7))),
                    (torch.rand((4, 5, 6, 7)), torch.rand((4, 5, 6, 7))),
                ],
                max_cache_len=6,
            )
            self.assertEqual(0, max_diff(c2, c2)["abs"])
            self.assertIsInstance(c2, transformers.cache_utils.StaticCache)
            flat, _spec = torch.utils._pytree.tree_flatten(c2)
            self.assertIsInstance(flat, list)
            self.assertEqual(len(flat), 6)
            unflat = flatten_unflatten_for_dynamic_shapes(c2)
            self.assertIsInstance(unflat, list)
            self.assertEqual(len(unflat), 2)
            self.assertEqual(
                "#2[#3[T1s4x5x6x7,T1s4x5x6x7,T1s4x5x6x7],#3[T1s4x5x6x7,T1s4x5x6x7,T1s4x5x6x7]]",
                self.string_type(unflat, with_shape=True),
            )

    def test_make_hybrid_cache(self):
        cache = make_hybrid_cache(
            [
                (torch.rand((4, 5, 6, 7)), torch.rand((4, 5, 6, 7))),
                (torch.rand((4, 5, 6, 7)), torch.rand((4, 5, 6, 7))),
                (torch.rand((4, 5, 6, 7)), torch.rand((4, 5, 6, 7))),
            ],
        )
        text = self.string_type(cache, with_shape=True)
        self.assertEqual(
            "HybridCache(key_cache=#3[T1s4x5x6x7,T1s4x5x6x7,T1s4x5x6x7], "
            "value_cache=#3[T1s4x5x6x7,T1s4x5x6x7,T1s4x5x6x7])",
            text,
        )
        self.assertEqual(0, max_diff(cache, cache)["abs"])
        self.assertEqual(0, max_diff(cache, torch_deepcopy(cache))["abs"])

    def test_unflatten_flatten_hybrid_cache(self):
        with torch_export_patches(patch_transformers=True):
            c2 = make_hybrid_cache(
                [
                    (torch.rand((4, 5, 6, 7)), torch.rand((4, 5, 6, 7))),
                    (torch.rand((4, 5, 6, 7)), torch.rand((4, 5, 6, 7))),
                    (torch.rand((4, 5, 6, 7)), torch.rand((4, 5, 6, 7))),
                ],
            )
            self.assertEqual(0, max_diff(c2, c2)["abs"])
            self.assertIsInstance(c2, transformers.cache_utils.HybridCache)
            flat, _spec = torch.utils._pytree.tree_flatten(c2)
            self.assertIsInstance(flat, list)
            self.assertEqual(len(flat), 6)
            unflat = flatten_unflatten_for_dynamic_shapes(c2)
            self.assertIsInstance(unflat, list)
            self.assertEqual(len(unflat), 2)
            self.assertEqual(
                "#2[#3[T1s4x5x6x7,T1s4x5x6x7,T1s4x5x6x7],#3[T1s4x5x6x7,T1s4x5x6x7,T1s4x5x6x7]]",
                self.string_type(unflat, with_shape=True),
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
