import unittest
import torch
from onnx_diagnostic.ext_test_case import ExtTestCase, ignore_warnings, requires_diffusers
from onnx_diagnostic.helpers.cache_helper import flatten_unflatten_for_dynamic_shapes
from onnx_diagnostic.torch_export_patches.onnx_export_errors import (
    torch_export_patches,
)
from onnx_diagnostic.helpers.torch_helper import torch_deepcopy


class TestPatchSerializationDiffusers(ExtTestCase):
    @ignore_warnings(UserWarning)
    @requires_diffusers("0.30")
    def test_unet_2d_condition_output(self):
        from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput

        bo = UNet2DConditionOutput(sample=torch.rand((4, 4, 4)))
        self.assertEqual(bo.__class__.__name__, "UNet2DConditionOutput")
        bo2 = torch_deepcopy([bo])
        self.assertIsInstance(bo2, list)
        self.assertEqual(
            "UNet2DConditionOutput(sample:T1s4x4x4)",
            self.string_type(bo, with_shape=True),
        )

        with torch_export_patches(patch_diffusers=True):
            # internal function
            bo2 = torch_deepcopy([bo])
            self.assertIsInstance(bo2, list)
            self.assertEqual(bo2[0].__class__.__name__, "UNet2DConditionOutput")
            self.assertEqualAny([bo], bo2)
            self.assertEqual(
                "UNet2DConditionOutput(sample:T1s4x4x4)",
                self.string_type(bo, with_shape=True),
            )

            # serialization
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

            # flatten_unflatten
            flat, _spec = torch.utils._pytree.tree_flatten(bo)
            unflat = flatten_unflatten_for_dynamic_shapes(bo, use_dict=True)
            self.assertIsInstance(unflat, dict)
            self.assertEqual(list(unflat), ["sample"])

        # export
        class Model(torch.nn.Module):
            def forward(self, cache):
                return cache.sample[0]

        bo = UNet2DConditionOutput(sample=torch.rand((4, 4, 4)))
        model = Model()
        model(bo)
        DYN = torch.export.Dim.DYNAMIC
        ds = [{0: DYN}]

        with torch_export_patches(patch_diffusers=True):
            torch.export.export(model, (bo,), dynamic_shapes=(ds,))


if __name__ == "__main__":
    unittest.main(verbosity=2)
