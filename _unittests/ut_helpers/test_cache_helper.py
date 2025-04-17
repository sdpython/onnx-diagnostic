import unittest
import torch
from onnx_diagnostic.ext_test_case import ExtTestCase
from onnx_diagnostic.helpers import string_type
from onnx_diagnostic.helpers.cache_helper import make_dynamic_cache
from onnx_diagnostic.export import CoupleInputsDynamicShapes
from onnx_diagnostic.torch_export_patches.patch_inputs import (
    convert_dynamic_axes_into_dynamic_shapes,
)


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

        cpl = CoupleInputsDynamicShapes(tuple(), kwargs, dynamic_shapes)
        res = cpl.replace_string_by()
        dsc = res["past_key_values"]
        self.assertEqual([[{0: batch, 2: DYN}], [{0: batch, 2: DYN}]], dsc)


if __name__ == "__main__":
    unittest.main(verbosity=2)
