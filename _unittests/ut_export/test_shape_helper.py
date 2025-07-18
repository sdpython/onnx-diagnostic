import unittest
import torch
from onnx_diagnostic.ext_test_case import ExtTestCase, requires_transformers, requires_torch
from onnx_diagnostic.export.shape_helper import (
    all_dynamic_shape_from_inputs,
    guess_dynamic_shapes_from_inputs,
)
from onnx_diagnostic.helpers.cache_helper import (
    make_dynamic_cache,
    make_sliding_window_cache,
    make_encoder_decoder_cache,
    make_static_cache,
)
from onnx_diagnostic.torch_models.hghub import get_untrained_model_with_inputs
from onnx_diagnostic.torch_export_patches import torch_export_patches


class TestShapeHelper(ExtTestCase):
    @requires_transformers("4.52")
    @requires_torch("2.7.99")
    def test_all_dynamic_shape_from_cache(self):
        cache = make_dynamic_cache([(torch.ones((2, 2)), (torch.ones((2, 2)) * 2))])
        ds = all_dynamic_shape_from_inputs(cache)
        self.assertEqual([[{0: "d_0_0", 1: "d_0_1"}], [{0: "d_1_0", 1: "d_1_1"}]], ds)

    @requires_torch("2.7.99")
    def test_all_dynamic_shape_all_transformers_cache(self):
        caches = [
            (
                make_dynamic_cache([(torch.ones((2, 2)), (torch.ones((2, 2)) * 2))]),
                [[{0: "d_0_0", 1: "d_0_1"}], [{0: "d_1_0", 1: "d_1_1"}]],
            ),
            (
                make_encoder_decoder_cache(
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
                ),
                [
                    [
                        [
                            {0: "d_0_0", 1: "d_0_1", 2: "d_0_2"},
                            {0: "d_1_0", 1: "d_1_1", 2: "d_1_2"},
                            {0: "d_2_0", 1: "d_2_1", 2: "d_2_2"},
                        ],
                        [
                            {0: "d_3_0", 1: "d_3_1", 2: "d_3_2"},
                            {0: "d_4_0", 1: "d_4_1", 2: "d_4_2"},
                            {0: "d_5_0", 1: "d_5_1", 2: "d_5_2"},
                        ],
                    ],
                    [
                        [
                            {0: "d_6_0", 1: "d_6_1", 2: "d_6_2"},
                            {0: "d_7_0", 1: "d_7_1", 2: "d_7_2"},
                            {0: "d_8_0", 1: "d_8_1", 2: "d_8_2"},
                        ],
                        [
                            {0: "d_9_0", 1: "d_9_1", 2: "d_9_2"},
                            {0: "d_10_0", 1: "d_10_1", 2: "d_10_2"},
                            {0: "d_11_0", 1: "d_11_1", 2: "d_11_2"},
                        ],
                    ],
                ],
            ),
            (
                make_sliding_window_cache(
                    [
                        (torch.rand((4, 5, 6, 7)), torch.rand((4, 5, 6, 7))),
                        (torch.rand((4, 5, 6, 7)), torch.rand((4, 5, 6, 7))),
                        (torch.rand((4, 5, 6, 7)), torch.rand((4, 5, 6, 7))),
                    ]
                ),
                [
                    [
                        {0: "d_0_0", 1: "d_0_1", 2: "d_0_2", 3: "d_0_3"},
                        {0: "d_1_0", 1: "d_1_1", 2: "d_1_2", 3: "d_1_3"},
                        {0: "d_2_0", 1: "d_2_1", 2: "d_2_2", 3: "d_2_3"},
                    ],
                    [
                        {0: "d_3_0", 1: "d_3_1", 2: "d_3_2", 3: "d_3_3"},
                        {0: "d_4_0", 1: "d_4_1", 2: "d_4_2", 3: "d_4_3"},
                        {0: "d_5_0", 1: "d_5_1", 2: "d_5_2", 3: "d_5_3"},
                    ],
                ],
            ),
            (
                make_static_cache(
                    [
                        (torch.rand((4, 5, 6, 7)), torch.rand((4, 5, 6, 7))),
                        (torch.rand((4, 5, 6, 7)), torch.rand((4, 5, 6, 7))),
                        (torch.rand((4, 5, 6, 7)), torch.rand((4, 5, 6, 7))),
                    ],
                    max_cache_len=15,
                ),
                [
                    [
                        {0: "d_0_0", 1: "d_0_1", 2: "d_0_2", 3: "d_0_3"},
                        {0: "d_1_0", 1: "d_1_1", 2: "d_1_2", 3: "d_1_3"},
                        {0: "d_2_0", 1: "d_2_1", 2: "d_2_2", 3: "d_2_3"},
                    ],
                    [
                        {0: "d_3_0", 1: "d_3_1", 2: "d_3_2", 3: "d_3_3"},
                        {0: "d_4_0", 1: "d_4_1", 2: "d_4_2", 3: "d_4_3"},
                        {0: "d_5_0", 1: "d_5_1", 2: "d_5_2", 3: "d_5_3"},
                    ],
                ],
            ),
        ]
        with torch_export_patches(patch_transformers=True):
            for cache, exds in caches:
                with self.subTest(cache_name=cache.__class__.__name__):
                    ds = all_dynamic_shape_from_inputs(cache)
                    self.assertEqual(exds, ds)

    @requires_transformers("4.52")
    @requires_torch("2.7.99")
    def test_all_dynamic_shape_from_inputs(self):
        ds = all_dynamic_shape_from_inputs((torch.randn((5, 6)), torch.randn((1, 6))))
        self.assertEqual(({0: "d_0_0", 1: "d_0_1"}, {0: "d_1_0", 1: "d_1_1"}), ds)
        ds = all_dynamic_shape_from_inputs([torch.randn((5, 6)), torch.randn((1, 6))])
        self.assertEqual([{0: "d_0_0", 1: "d_0_1"}, {0: "d_1_0", 1: "d_1_1"}], ds)
        ds = all_dynamic_shape_from_inputs(
            (torch.randn((5, 6)), torch.randn((1, 6))), dim_prefix=torch.export.Dim.AUTO
        )
        self.assertEqual(
            (
                {0: torch.export.Dim.AUTO, 1: torch.export.Dim.AUTO},
                {0: torch.export.Dim.AUTO, 1: torch.export.Dim.AUTO},
            ),
            ds,
        )

    @requires_transformers("4.52")
    @requires_torch("2.7.99")
    def test_all_dynamic_shape_from_inputs_dynamic_cache(self):
        data = get_untrained_model_with_inputs("arnir0/Tiny-LLM")
        ds = all_dynamic_shape_from_inputs(data["inputs"])
        self.assertEqual(
            {
                "input_ids": {0: "d_0_0", 1: "d_0_1"},
                "attention_mask": {0: "d_1_0", 1: "d_1_1"},
                "position_ids": {0: "d_2_0", 1: "d_2_1"},
                "past_key_values": [
                    [{0: "d_3_0", 1: "d_3_1", 2: "d_3_2", 3: "d_3_3"}],
                    [{0: "d_4_0", 1: "d_4_1", 2: "d_4_2", 3: "d_4_3"}],
                ],
            },
            ds,
        )

    @requires_transformers("4.52")
    @requires_torch("2.7.99")
    def test_guess_dynamic_shapes_from_inputs(self):
        data = get_untrained_model_with_inputs("arnir0/Tiny-LLM", add_second_input=True)
        guessed = guess_dynamic_shapes_from_inputs(
            [data["inputs"], data["inputs2"]], auto="dd"
        )
        self.assertEqual(
            (
                (),
                {
                    "attention_mask": {0: "dd_0I0", 1: "dd_0I1"},
                    "input_ids": {0: "dd_1I0", 1: "dd_1I1"},
                    "past_key_values": [
                        [{0: "dd_2I_0o_0l0", 2: "dd_2I_0o_0l2"}],
                        [{0: "dd_2I_1o_0l0", 2: "dd_2I_1o_0l2"}],
                    ],
                    "position_ids": {0: "dd_3I0", 1: "dd_3I1"},
                },
            ),
            guessed,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
