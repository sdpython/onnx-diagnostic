import unittest
from onnx_diagnostic.ext_test_case import (
    ExtTestCase,
    requires_torch,
    requires_transformers,
    skipif_ci_windows,
    ignore_warnings,
    hide_stdout,
    has_transformers,
)
from onnx_diagnostic.helpers import string_type
from onnx_diagnostic.torch_export_patches.onnx_export_errors import (
    torch_export_patches,
)


class TestOnnxExportErrors(ExtTestCase):
    @requires_transformers("4.49.999")
    @skipif_ci_windows("not working on Windows")
    @ignore_warnings(UserWarning)
    @hide_stdout()
    def test_pytree_flatten_mamba_cache(self):
        import torch
        import torch.utils._pytree as py_pytree
        from transformers.cache_utils import MambaCache

        class _config:
            def __init__(self):
                self.intermediate_size = 8
                self.state_size = 16
                self.conv_kernel = 32
                self.num_hidden_layers = 64
                self.dtype = torch.float16

        cache = MambaCache(_config(), max_batch_size=1, device="cpu")

        with torch_export_patches(verbose=1):
            values, spec = py_pytree.tree_flatten(cache)
            cache2 = py_pytree.tree_unflatten(values, spec)
            self.assertEqual(cache.max_batch_size, cache2.max_batch_size)
            self.assertEqual(cache.intermediate_size, cache2.intermediate_size)
            self.assertEqual(cache.ssm_state_size, cache2.ssm_state_size)
            self.assertEqual(cache.conv_kernel_size, cache2.conv_kernel_size)
            self.assertEqualArrayAny(cache.conv_states, cache2.conv_states)
            self.assertEqualArrayAny(cache.ssm_states, cache2.ssm_states)

    @requires_transformers("4.43")
    @requires_torch("2.7")
    @skipif_ci_windows("not working on Windows")
    @ignore_warnings(UserWarning)
    @hide_stdout()
    def test_exportable_mamba_cache(self):
        import torch
        from transformers.models.mamba.modeling_mamba import MambaCache

        class _config:
            def __init__(self):
                self.intermediate_size = 8
                self.state_size = 16
                self.conv_kernel = 32
                self.num_hidden_layers = 64
                self.dtype = torch.float16

        class Model(torch.nn.Module):
            def forward(self, x: torch.Tensor, cache: MambaCache):
                x1 = cache.ssm_states[0] + x
                x2 = cache.conv_states[0][:, :, ::2] + x1
                return x2

        cache = MambaCache(_config(), max_batch_size=1, device="cpu")
        if has_transformers("4.50"):
            # MambaCache was updated in 4.50
            self.assertEqual(
                "MambaCache(conv_states=#64[T10r3,...], ssm_states=#64[T10r3,...])",
                string_type(cache),
            )
        x = torch.ones(2, 8, 16).to(torch.float16)
        model = Model()
        model(x, cache)

        with torch_export_patches(verbose=1):
            cache = MambaCache(_config(), max_batch_size=1, device="cpu")
            torch.export.export(Model(), (x, cache))

    @requires_transformers("4.49.999")
    @skipif_ci_windows("not working on Windows")
    @ignore_warnings(UserWarning)
    def test_exportable_mamba_cache_dynamic(self):
        import torch
        from transformers.models.mamba.modeling_mamba import MambaCache

        class _config:
            def __init__(self):
                self.intermediate_size = 8
                self.state_size = 16
                self.conv_kernel = 32
                self.num_hidden_layers = 2
                self.dtype = torch.float16

        class Model(torch.nn.Module):
            def forward(self, x: torch.Tensor, cache: MambaCache):
                x1 = cache.ssm_states[0] + x
                x2 = cache.conv_states[0][:, :, ::2] + x1
                return x2

        cache = MambaCache(_config(), max_batch_size=1, device="cpu")
        self.assertEqual(
            string_type(cache),
            "MambaCache(conv_states=#2[T10r3,T10r3], ssm_states=#2[T10r3,T10r3])",
        )
        x = torch.ones(2, 8, 16).to(torch.float16)
        model = Model()
        model(x, cache)
        DYN = torch.export.Dim.DYNAMIC

        with torch_export_patches():
            cache = MambaCache(_config(), max_batch_size=2, device="cpu")
            torch.export.export(
                Model(),
                (x, cache),
                dynamic_shapes=({0: DYN}, [[{0: DYN}, {0: DYN}], [{0: DYN}, {0: DYN}]]),
            )

    @ignore_warnings(UserWarning)
    def test_exportable_dynamic_shapes_constraints(self):
        import torch

        class CustomCache:
            def __init__(self, shape=None):
                self.cache = [torch.zeros((shape)), torch.zeros((shape))] if shape else []

        def flatten_cache(cache):
            return [cache.cache], ["cache"]

        def unflatten_cache(values, context, output_type=None):
            cache = CustomCache()
            cache.cache = values[0]
            return cache

        def flatten_with_keys_cache(d):
            values, context = flatten_cache(d)
            return [
                (torch.utils._pytree.MappingKey(k), v) for k, v in zip(context, values)
            ], context

        torch.utils._pytree.register_pytree_node(
            CustomCache,
            flatten_cache,
            unflatten_cache,
            serialized_type_name=f"{CustomCache.__module__}.{CustomCache.__name__}",
            flatten_with_keys_fn=flatten_with_keys_cache,
        )

        class Model(torch.nn.Module):
            def forward(self, x, cache):
                return cache.cache[0][0, :] + x

        model = Model()
        model.eval()
        x, cache = torch.rand((2, 4)), CustomCache((2, 4))
        model(x, cache)
        DYN = torch.export.Dim.DYNAMIC
        torch.export.export(
            model, (x, cache), dynamic_shapes=({0: DYN}, [[{0: DYN}, {0: DYN}]])
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
