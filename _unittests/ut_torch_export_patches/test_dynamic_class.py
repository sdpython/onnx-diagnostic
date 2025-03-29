import copy
import unittest
import torch
from onnx_diagnostic.ext_test_case import ExtTestCase, ignore_warnings, hide_stdout
from onnx_diagnostic.helpers import string_type
from onnx_diagnostic.cache_helpers import make_dynamic_cache
from onnx_diagnostic.torch_export_patches.onnx_export_errors import (
    bypass_export_some_errors,
)
from onnx_diagnostic.torch_models.hghub.model_inputs import get_untrained_model_with_inputs


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

    @ignore_warnings(UserWarning)
    def test_phi2_export_module(self):
        data = get_untrained_model_with_inputs("microsoft/phi-2")
        model, inputs, dyn_shapes = data["model"], data["inputs"], data["dynamic_shapes"]
        str_inputs = string_type(inputs, with_shape=True, with_min_max=True)
        inputs_copied = copy.deepcopy(inputs)
        expected = model(**inputs_copied)
        self.maxDiff = None
        self.assertEqual(str_inputs, string_type(inputs, with_shape=True, with_min_max=True))

        # The cache is modified inplace, that's why, we copied it.
        self.assertNotEqual(
            string_type(inputs, with_shape=True, with_min_max=True),
            string_type(inputs_copied, with_shape=True, with_min_max=True),
        )
        inputs_copied = copy.deepcopy(inputs)
        self.assertEqual(
            str_inputs, string_type(inputs_copied, with_shape=True, with_min_max=True)
        )

        with bypass_export_some_errors(patch_transformers=True):
            ep = torch.export.export(
                model,
                (),
                kwargs=inputs,
                dynamic_shapes=dyn_shapes,
                strict=False,  # True works but then the it fails during the execution
            )
            mod = ep.module()
            inputs_copied = copy.deepcopy(inputs)
            self.assertEqual(
                str_inputs, string_type(inputs_copied, with_shape=True, with_min_max=True)
            )
            got = mod(**inputs_copied)
            self.assertEqualAny(expected, got)

        inputs_copied = copy.deepcopy(inputs)
        self.assertEqual(
            str_inputs, string_type(inputs_copied, with_shape=True, with_min_max=True)
        )
        mod = ep.module()
        got = mod(**inputs_copied)
        self.assertEqualAny(expected, got)

    @ignore_warnings(UserWarning)
    def test_phi2_export_interpreter(self):
        data = get_untrained_model_with_inputs("microsoft/phi-2")
        model, inputs, dyn_shapes = data["model"], data["inputs"], data["dynamic_shapes"]
        str_inputs = string_type(inputs, with_shape=True, with_min_max=True)
        inputs_copied = copy.deepcopy(inputs)
        expected = model(**inputs_copied)
        self.maxDiff = None
        self.assertEqual(str_inputs, string_type(inputs, with_shape=True, with_min_max=True))

        # The cache is modified inplace, that's why, we copied it.
        self.assertNotEqual(
            string_type(inputs, with_shape=True, with_min_max=True),
            string_type(inputs_copied, with_shape=True, with_min_max=True),
        )
        inputs_copied = copy.deepcopy(inputs)
        self.assertEqual(
            str_inputs, string_type(inputs_copied, with_shape=True, with_min_max=True)
        )

        with bypass_export_some_errors(patch_transformers=True):
            ep = torch.export.export(
                model,
                (),
                kwargs=inputs,
                dynamic_shapes=dyn_shapes,
                strict=False,  # True works but then the it fails during the execution
            )

        # from experimental_experiment.torch_interpreter.tracing import CustomTracer
        # CustomTracer.remove_unnecessary_slices(ep.graph)
        memorize = []

        class MyInterpreter(torch.fx.Interpreter):
            def call_function(self, target, args, kwargs):
                res = super().call_function(target, args, kwargs)
                memorize.append((target, args, kwargs, res))
                return res

        inputs_copied = copy.deepcopy(inputs)
        self.assertEqual(
            str_inputs, string_type(inputs_copied, with_shape=True, with_min_max=True)
        )
        args, _spec = torch.utils._pytree.tree_flatten(inputs_copied)
        got = MyInterpreter(ep.module()).run(*args)
        self.assertEqualAny(expected, got)


if __name__ == "__main__":
    unittest.main(verbosity=2)
