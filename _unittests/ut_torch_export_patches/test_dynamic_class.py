import unittest
import torch
from onnx_diagnostic.ext_test_case import ExtTestCase, ignore_warnings, hide_stdout
from onnx_diagnostic.helpers import string_type
from onnx_diagnostic.cache_helpers import make_dynamic_cache
from onnx_diagnostic.torch_export_patches.onnx_export_errors import (
    bypass_export_some_errors,
)


class TestOnnxExportErrors(ExtTestCase):
    @ignore_warnings(UserWarning)
    @hide_stdout()
    def test_export_dynamic_cache_update(self):
        for strict in self.subloop([True, False], verbose=1):

            class SubModelCache(torch.nn.Module):
                def forward(self, cache):
                    d = cache.__class__()
                    d.update(cache.key_cache[0] + 1, cache.value_cache[0] + 2, 0)
                    d.update(cache.key_cache[0] + 3, cache.value_cache[0] + 5, 1)
                    return d

            class SubModel(torch.nn.Module):
                def forward(self, x, cache):
                    return x + cache.key_cache[0] + cache.value_cache[0]

            class Model(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.sub = SubModel()
                    self.subcache = SubModelCache()

                def forward(self, x, cache):
                    return self.sub(x, self.subcache(cache))

            # no patch
            cache = make_dynamic_cache(
                [(torch.ones((5, 6, 5, 6)), torch.ones((5, 6, 5, 6)) + 2)]
            )
            model = Model()
            inputs = (torch.randn((5, 6, 5, 6)), cache)
            expected = model(*inputs)

            DYN = torch.export.Dim.DYNAMIC
            ep = torch.export.export(
                model,
                inputs,
                dynamic_shapes=({0: DYN, 2: DYN}, [[{0: DYN, 2: DYN}], [{0: DYN, 2: DYN}]]),
                strict=strict,
            )
            mod = ep.module()
            got = mod(*inputs)
            self.assertEqualArray(expected, got)

            # patching
            with bypass_export_some_errors(patch_transformers=True):
                got = model(*inputs)
                self.assertEqualArray(expected, got)
                ep2 = torch.export.export(
                    model,
                    inputs,
                    dynamic_shapes=(
                        {0: DYN, 2: DYN},
                        [[{0: DYN, 2: DYN}], [{0: DYN, 2: DYN}]],
                    ),
                    strict=strict,
                )
                mod = ep2.module()
                got = mod(*inputs)
                self.assertEqualArray(expected, got)


if __name__ == "__main__":
    unittest.main(verbosity=2)
