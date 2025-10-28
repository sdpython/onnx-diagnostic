import copy
import os
import unittest
from typing import Any, Dict, List, Tuple
import torch

try:
    import transformers.masking_utils as masking_utils
except ImportError:
    masking_utils = None
from onnx_diagnostic.ext_test_case import (
    ExtTestCase,
    ignore_warnings,
    hide_stdout,
    requires_torch,
)
from onnx_diagnostic.helpers import string_type
from onnx_diagnostic.helpers.cache_helper import make_dynamic_cache, CacheKeyValue
from onnx_diagnostic.torch_export_patches.onnx_export_errors import (
    torch_export_patches,
)
from onnx_diagnostic.torch_export_patches.patch_inputs import use_dyn_not_str
from onnx_diagnostic.torch_models.hghub.model_inputs import get_untrained_model_with_inputs
import onnx_diagnostic.torch_export_patches.patches.patch_transformers as patch_transformers


class TestOnnxExportErrors(ExtTestCase):
    @ignore_warnings(UserWarning)
    @hide_stdout()
    def test_export_dynamic_cache_update(self):
        class SubModelCache(torch.nn.Module):
            def forward(self, cache):
                cc = CacheKeyValue(cache)
                # If not patched...
                # Fails with transformers>=4.54 because function ``parse_processor_args``
                # relies in inspect and the exporter is not very fond of that.
                # torch._dynamo.exc.Unsupported: id() with unsupported args
                # Explanation: Dynamo doesn't know how to trace id()
                # call with args
                # (GetAttrVariable(ConstantVariable(NoneType: None), __init__),)
                # Hint: Supported args are Tensors, and functions/nn.Modules/user-defined
                # objects from outside the compiled region.
                # Hint: It may be possible to write Dynamo tracing rules for this code.
                d = cache.__class__()
                d.update(cc.key_cache[0] + 1, cc.value_cache[0] + 2, 0)
                d.update(cc.key_cache[0] + 3, cc.value_cache[0] + 5, 1)
                return d

        class SubModel(torch.nn.Module):
            def forward(self, x, cache):
                cc = CacheKeyValue(cache)
                y = cc.key_cache[0] + cc.value_cache[0]
                return x + y

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.sub = SubModel()
                self.subcache = SubModelCache()

            def forward(self, x, cache):
                return self.sub(x, self.subcache(cache))

        # no patch
        cache = make_dynamic_cache([(torch.ones((5, 6, 5, 6)), torch.ones((5, 6, 5, 6)) + 2)])
        model = Model()
        inputs = (torch.randn((5, 6, 5, 6)), cache)
        expected = model(*inputs)

        DYN = torch.export.Dim.DYNAMIC

        # patching
        with torch_export_patches(patch_transformers=True, verbose=10):
            got = model(*inputs)
            self.assertEqualArray(expected, got)
            ep = torch.export.export(
                model,
                inputs,
                dynamic_shapes=({0: DYN, 2: DYN}, [{0: DYN, 2: DYN}, {0: DYN, 2: DYN}]),
                strict=False,
            )
            mod = ep.module()
            got = mod(*inputs)
            self.assertEqualArray(expected, got)

            class MyInterpreter(torch.fx.Interpreter):
                def call_function(self, target, args, kwargs):
                    res = super().call_function(target, args, kwargs)
                    return res

            args, _spec = torch.utils._pytree.tree_flatten(inputs)
            got = MyInterpreter(ep.module()).run(*args)
            self.assertEqualAny(expected, got)

    @ignore_warnings(UserWarning)
    @requires_torch(
        "2.7",
        "MyCache78'> does not have a flatten_fn_spec, "
        "use torch.fx._pytree.register_pytree_flatten_spec",
    )
    def test_export_mycache_list_cat(self):
        TreeContext = torch.utils._pytree.Context
        MappingKey = torch.utils._pytree.MappingKey
        KeyEntry = torch.utils._pytree.KeyEntry

        class MyCache77:
            def __init__(self, key=None, value=None):
                self.key_cache = [key] if key is not None else []
                self.value_cache = [value] if value is not None else []

        class ModelMyCache(torch.nn.Module):
            def forward(self, x, dc):
                y = (
                    (
                        torch.cat(dc.key_cache, axis=1) + torch.cat(dc.value_cache, axis=1)
                    ).reshape((-1, x.shape[1]))
                ).transpose(1, 0)
                return x @ y

        inputs = {
            "x": torch.randn(3, 8),
            "dc": MyCache77(torch.ones((3, 8, 3, 8)), torch.ones((3, 8, 3, 8))),
        }
        model = ModelMyCache()
        expected = model(**inputs)

        def flatten_my_cache77(cache: MyCache77) -> Tuple[List[Any], TreeContext]:
            flat = [
                (k, getattr(cache, k))
                for k in ["key_cache", "value_cache"]
                if hasattr(cache, k)
            ]
            return [f[1] for f in flat], [f[0] for f in flat]

        def flatten_with_keys_my_cache77(
            d: Dict[Any, Any],
        ) -> Tuple[List[Tuple[KeyEntry, Any]], TreeContext]:
            values, context = flatten_my_cache77(d)
            return [(MappingKey(k), v) for k, v in zip(context, values)], context

        def unflatten_my_cache_77(
            values: List[Any], context: TreeContext, output_type=None
        ) -> MyCache77:
            cache = MyCache77()
            values = dict(zip(context, values))
            for k, v in values.items():
                setattr(cache, k, v)
            return cache

        torch.utils._pytree.register_pytree_node(
            MyCache77,
            flatten_my_cache77,
            unflatten_my_cache_77,
            serialized_type_name="MyCache77",
            flatten_with_keys_fn=flatten_with_keys_my_cache77,
        )

        # DYN = torch.export.Dim.DYNAMIC
        ep = torch.export.export(model, (), kwargs=inputs)

        args, _spec = torch.utils._pytree.tree_flatten(inputs)
        got = torch.fx.Interpreter(ep.module()).run(*args)
        self.assertEqualAny(expected, got)

        mod = ep.module()
        got = mod(**inputs)
        self.assertEqualArray(expected, got)

    @ignore_warnings(UserWarning)
    @requires_torch(
        "2.7",
        "MyCache78'> does not have a flatten_fn_spec, "
        "use torch.fx._pytree.register_pytree_flatten_spec",
    )
    def test_export_mycache_dict_cat(self):
        TreeContext = torch.utils._pytree.Context

        class MyCache78:
            def __init__(self, key=None, value=None):
                self.key_cache = [key] if key is not None else []
                self.value_cache = [value] if value is not None else []

        class ModelMyCache(torch.nn.Module):
            def forward(self, x, dc):
                y = (
                    (
                        torch.cat(dc.key_cache, axis=1) + torch.cat(dc.value_cache, axis=1)
                    ).reshape((-1, x.shape[1]))
                ).transpose(1, 0)
                return x @ y

        inputs = {
            "x": torch.randn(3, 8),
            "dc": MyCache78(torch.ones((3, 8, 3, 8)), torch.ones((3, 8, 3, 8))),
        }
        model = ModelMyCache()
        expected = model(**inputs)

        def flatten_my_cache78(cache: MyCache78):
            dictionary = {
                "key_cache": cache.key_cache,
                "value_cache": cache.value_cache,
            }
            return torch.utils._pytree._dict_flatten(dictionary)

        def flatten_with_keys_my_cache78(cache: MyCache78):
            dictionary = {
                "key_cache": cache.key_cache,
                "value_cache": cache.value_cache,
            }
            return torch.utils._pytree._dict_flatten_with_keys(dictionary)

        def unflatten_my_cache_78(values, context: TreeContext, output_type=None) -> MyCache78:
            dictionary = torch.utils._pytree._dict_unflatten(values, context)
            cache = MyCache78()
            for k, v in dictionary.items():
                setattr(cache, k, v)
            return cache

        torch.utils._pytree.register_pytree_node(
            MyCache78,
            flatten_my_cache78,
            unflatten_my_cache_78,
            serialized_type_name="MyCache78",
            flatten_with_keys_fn=flatten_with_keys_my_cache78,
        )

        # DYN = torch.export.Dim.DYNAMIC
        ep = torch.export.export(model, (), kwargs=inputs)

        args, _spec = torch.utils._pytree.tree_flatten(inputs)
        got = torch.fx.Interpreter(ep.module()).run(*args)
        self.assertEqualAny(expected, got)

        mod = ep.module()
        got = mod(**inputs)
        self.assertEqualArray(expected, got)

    @ignore_warnings(UserWarning)
    def test_export_dynamic_cache_cat(self):

        class ModelDynamicCache(torch.nn.Module):
            def forward(self, x, dc):
                cc = CacheKeyValue(dc)
                y = (
                    (
                        torch.cat(cc.key_cache, axis=1) + torch.cat(cc.value_cache, axis=1)
                    ).reshape((-1, x.shape[1]))
                ).transpose(1, 0)
                return x @ y

        inputs = {
            "x": torch.randn(3, 8),
            "dc": make_dynamic_cache(
                [(torch.ones((3, 8, 3, 8)), (torch.ones((3, 8, 3, 8)) * 2))]
            ),
        }
        model = ModelDynamicCache()
        expected = model(**inputs)

        # DYN = torch.export.Dim.DYNAMIC
        NOBYPASS = int(os.environ.get("NOBYBASS", "0"))
        if NOBYPASS:
            ep = torch.export.export(model, (), kwargs=inputs)

            args, _spec = torch.utils._pytree.tree_flatten(inputs)
            got = torch.fx.Interpreter(ep.module()).run(*args)
            self.assertEqualAny(expected, got)

            mod = ep.module()
            got = mod(**inputs)
            self.assertEqualArray(expected, got)
            return

        with torch_export_patches(patch_transformers=True):
            ep = torch.export.export(model, (), kwargs=inputs)

            args, _spec = torch.utils._pytree.tree_flatten(inputs)
            got = torch.fx.Interpreter(ep.module()).run(*args)
            self.assertEqualAny(expected, got)

            mod = ep.module()
            got = mod(**inputs)
            self.assertEqualArray(expected, got)

    @ignore_warnings(UserWarning)
    @requires_torch("2.9")
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

        with torch_export_patches(patch_transformers=True):
            ep = torch.export.export(
                model,
                (),
                kwargs=inputs,
                dynamic_shapes=use_dyn_not_str(dyn_shapes),
                strict=False,  # True works but then the it fails during the execution
            )
            # ep = ep.run_decompositions()
            mod = ep.module()
            inputs_copied = copy.deepcopy(inputs)
            self.assertEqual(
                str_inputs, string_type(inputs_copied, with_shape=True, with_min_max=True)
            )
            got = mod(**inputs_copied)
            self.assertEqualAny(expected, got)

    @ignore_warnings(UserWarning)
    @requires_torch("2.9")
    @hide_stdout()
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

        with torch_export_patches(patch_transformers=True, verbose=1):
            if masking_utils is not None:
                self.assertEqual(
                    masking_utils.ALL_MASK_ATTENTION_FUNCTIONS["sdpa"],
                    patch_transformers.patched_sdpa_mask_recent_torch,
                )
            ep = torch.export.export(
                model,
                (),
                kwargs=inputs,
                dynamic_shapes=use_dyn_not_str(dyn_shapes),
                strict=False,  # True works but then the it fails during the execution
            )
            # ep = ep.run_decompositions()

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
