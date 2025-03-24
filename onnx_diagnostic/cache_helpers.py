from typing import List, Tuple
import packaging.version as pv
import torch
import transformers
import transformers.cache_utils

if pv.Version(transformers.__version__) > pv.Version("4.49.99999"):

    def make_dynamic_cache(
        key_value_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> transformers.cache_utils.DynamicCache:
        """
        Creates an instance of :class:`transformers.cache_utils.DynamicCache`.
        This version is valid for ``transformers >= 4.50``.

        :param key_value_pairs: list of pairs of (key, values)
        :return: :class:`transformers.cache_utils.DynamicCache`

        Example:

        .. runpython::
            :showcode:

            import torch
            from onnx_diagnostic.cache_helpers import make_dynamic_cache
            from onnx_diagnostic.helpers import string_type

            n_layers = 2
            bsize, nheads, slen, dim = 2, 4, 3, 7

            past_key_values = make_dynamic_cache(
                [
                    (
                        torch.randn(bsize, nheads, slen, dim),
                        torch.randn(bsize, nheads, slen, dim),
                    )
                    for i in range(n_layers)
                ]
            )
            print(string_type(past_key_values, with_shape=True))
        """
        return transformers.cache_utils.DynamicCache(key_value_pairs)

else:

    def make_dynamic_cache(
        key_value_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> transformers.cache_utils.DynamicCache:
        """
        Creates an instance of :class:`transformers.cache_utils.DynamicCache`.
        This version is valid for ``transformers < 4.50``.

        :param key_value_pairs: list of pairs of (key, values)
        :return: :class:`transformers.cache_utils.DynamicCache`

        Example:

        .. runpython::
            :showcode:

            import torch
            from onnx_diagnostic.cache_helpers import make_dynamic_cache
            from onnx_diagnostic.helpers import string_type

            n_layers = 2
            bsize, nheads, slen, dim = 2, 4, 3, 7

            past_key_values = make_dynamic_cache(
                [
                    (
                        torch.randn(bsize, nheads, slen, dim),
                        torch.randn(bsize, nheads, slen, dim),
                    )
                    for i in range(n_layers)
                ]
            )
            print(string_type(past_key_values, with_shape=True))
        """
        cache = transformers.cache_utils.DynamicCache(len(key_value_pairs))
        for i, (key, value) in enumerate(key_value_pairs):
            cache.update(key, value, i)
        return cache
