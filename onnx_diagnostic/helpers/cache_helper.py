from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import packaging.version as pv
import torch
import transformers
import transformers.cache_utils


class CacheKeyValue:
    """
    Starting transformers>=4.54, the cache API has deprecated
    ``cache.key_cache`` and ``cache.value_cache``.
    This class wraps a cache independently from transformers version and enables
    attributes ``key_cache`` and ``value_cache``.

    .. code-block:: python

        capi = CacheKeyValue(cache)
        capi.key_cache
        capi.value_cache
    """

    def __init__(self, cache=None):
        if hasattr(cache, "layers"):
            layers = [
                layer
                for layer in cache.layers
                if layer is not None and layer.keys is not None and layer.values is not None
            ]
            self.key_cache = [layer.keys for layer in layers]
            self.value_cache = [layer.values for layer in layers]
            if None in self.key_cache or None in self.value_cache:
                from .helper import string_type

                raise AssertionError(
                    f"issue with key_cache={string_type(self.key_cache)}, "
                    f"or value_cache={string_type(self.value_cache)}, "
                    f"cache.layers={string_type(cache.layers)}"
                )
        elif cache is not None and hasattr(cache, "key_cache"):
            self.key_cache = cache.key_cache
            self.value_cache = cache.value_cache
        elif cache is None:
            self.key_cache = None
            self.value_cache = None
        else:
            raise NotImplementedError(f"type(cache)={type(cache)}")

    def make_dynamic_cache(self):
        """Does the reverse operation."""
        return make_dynamic_cache(list(zip(self.key_cache, self.value_cache)))

    @property
    def n_layers(self) -> int:
        """Returns the number of layers."""
        return len(self.key_cache) if self.key_cache else 0


def flatten_unflatten_for_dynamic_shapes(
    obj: Any,
    use_dict: bool = False,
    change_function: Optional[Callable[[torch.Tensor], Any]] = None,
) -> Any:
    """
    Returns the object in a different structure similar to what
    the definition of the dynamic shapes should use.

    :param obj: object from a custom class
    :param use_dict: closer to the original result but
        :func:`torch.export.export` only considers the values,
        the context gives the dictionary keys but it is not expressed
        in the dynamic shapes, these specifications seems to be different
        for the strict and non strict mode. It also preserves tuple.
    :param change_function: to modifies the tensor in the structure itself,
        like replace them by a shape
    :return: the serialized object
    """
    if isinstance(obj, torch.Tensor):
        return change_function(obj) if change_function else obj
    flat, spec = torch.utils._pytree.tree_flatten(obj)
    start = 0
    end = 0
    subtrees = []
    for subspec in spec.children():
        end += subspec.num_leaves
        value = subspec.unflatten(flat[start:end])
        value = flatten_unflatten_for_dynamic_shapes(
            value, use_dict=use_dict, change_function=change_function
        )
        subtrees.append(value)
        start = end
    if use_dict:
        if spec.type is dict:
            # This a dictionary.
            return dict(zip(spec.context, subtrees))
        if spec.type is tuple:
            return tuple(subtrees)
        if spec.type is list:
            return list(subtrees)
        if spec.type is None and not subtrees:
            return None
        if spec.context:
            # This is a custom class with attributes.
            # It is returned as a list.
            return list(subtrees)
        raise ValueError(
            f"Unable to interpret spec type {spec.type} "
            f"(type is {type(spec.type)}, context is {spec.context}), "
            f"spec={spec}, subtrees={subtrees}"
        )
    # This is a list.
    return subtrees


def is_cache_dynamic_registered(fast: bool = False) -> bool:
    """
    Tells if class :class:`transformers.cache_utils.DynamicCache` can be
    serialized and deserialized. Only then, :func:`torch.export.export`
    can export a model.

    :param fast: if True, do not check the serialization is ok as well
    :return: result
    """
    if fast:
        return transformers.cache_utils.DynamicCache in torch.utils._pytree.SUPPORTED_NODES
    bsize, nheads, slen, dim = 2, 4, 3, 7
    cache = make_dynamic_cache(
        [
            (
                torch.randn(bsize, nheads, slen, dim),
                torch.randn(bsize, nheads, slen, dim),
            )
            for i in range(2)
        ]
    )
    values, spec = torch.utils._pytree.tree_flatten(cache)
    cache2 = torch.utils._pytree.tree_unflatten(values, spec)
    if hasattr(cache2, "layers") and hasattr(cache, "layers"):
        return len(cache2.layers) == len(cache.layers)
    return len(cache2.key_cache) == len(cache.value_cache)


def make_dynamic_shapes_kv_cache(
    cache: transformers.cache_utils.Cache, shape_of_one: Dict[int, Any]
) -> List[Dict[int, Any]]:
    """
    Returns the dynamic shapes for key-value cache

    :param cache: a cache
    :param shape_of_one: shape of one element
    :return: dynamic shapes
    """
    return [shape_of_one for _ in range(CacheKeyValue(cache).n_layers * 2)]


def _preprocess_key_value_pairs(
    key_value_pairs: Union[List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]],
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    if not key_value_pairs or isinstance(key_value_pairs[0], tuple):
        return key_value_pairs
    return list(zip(key_value_pairs[::2], key_value_pairs[1::2]))


if pv.Version(transformers.__version__) > pv.Version("4.49.99999"):

    def make_dynamic_cache(
        key_value_pairs: Union[List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]],
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
            from onnx_diagnostic.helpers import string_type
            from onnx_diagnostic.helpers.cache_helper import make_dynamic_cache

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

        The function is fully able to handle ``FakeTensor`` with dynamic dimensions if
        ``transformers>=4.56``. Before that version, only FakeTensor with static dimensions
        are supported.
        """
        key_value_pairs = _preprocess_key_value_pairs(key_value_pairs)
        if (
            key_value_pairs
            and isinstance(key_value_pairs[0][0], torch._subclasses.fake_tensor.FakeTensor)
            and pv.Version(transformers.__version__) >= pv.Version("4.56")
        ):
            cache = transformers.cache_utils.DynamicCache()
            cache.layers.extend(
                [transformers.cache_utils.DynamicLayer() for _ in key_value_pairs]
            )
            for i, layer in enumerate(cache.layers):
                k, v = key_value_pairs[i][0], key_value_pairs[i][1]
                layer.dtype = k.dtype
                layer.device = k.device
                layer.keys = k
                layer.values = v
                layer.is_initialized = True
            assert not hasattr(cache, "layers") or len(key_value_pairs) == len(cache.layers), (
                f"Unexpected number of layers in the cache ({len(cache.layers)}), "
                f"{len(key_value_pairs)} expected."
            )
            return finalize_cache(cache)

        cache = transformers.cache_utils.DynamicCache(key_value_pairs)
        if hasattr(cache, "layers") and len(key_value_pairs) < len(cache.layers):
            # The cache constructor contains the two following lines
            # (in cache_utils.py) which append empty layers when the cache is
            # initialized. We need to remove them.
            # self.num_hidden_layers = getattr(config, "num_hidden_layers", 1)
            # self.append_new_layers(self.num_hidden_layers - 1)
            cache.layers[:] = cache.layers[-len(key_value_pairs) :]
        assert not hasattr(cache, "layers") or len(key_value_pairs) == len(cache.layers), (
            f"Unexpected number of layers in the cache ({len(cache.layers)}), "
            f"{len(key_value_pairs)} expected."
        )
        return finalize_cache(cache)

else:

    def make_dynamic_cache(
        key_value_pairs: Union[List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]],
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
            from onnx_diagnostic.helpers import string_type
            from onnx_diagnostic.helpers.cache_helper import make_dynamic_cache

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
        key_value_pairs = _preprocess_key_value_pairs(key_value_pairs)
        cache = transformers.cache_utils.DynamicCache(len(key_value_pairs))  # type: ignore
        for i, (key, value) in enumerate(key_value_pairs):
            cache.update(key, value, i)
        return cache


def make_static_cache(
    key_value_pairs: Union[List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]],
    max_cache_len: Optional[int] = None,
) -> transformers.cache_utils.DynamicCache:
    """
    Creates an instance of :class:`transformers.cache_utils.StaticCache`.
    :param key_value_pairs: list of pairs of (key, values)
    :param max_cache_len: max_cache_length or something inferred from the vector
    :return: :class:`transformers.cache_utils.StaticCache`

    Example:

    .. runpython::
        :showcode:

        import torch
        from onnx_diagnostic.helpers import string_type
        from onnx_diagnostic.helpers.cache_helper import make_static_cache

        n_layers = 2
        bsize, nheads, slen, dim = 2, 4, 3, 7

        past_key_values = make_static_cache(
            [
                (
                    torch.randn(bsize, nheads, slen, dim),
                    torch.randn(bsize, nheads, slen, dim),
                )
                for i in range(n_layers)
            ],
            max_cache_len=10,
        )
        print(string_type(past_key_values, with_shape=True))
    """
    key_value_pairs = _preprocess_key_value_pairs(key_value_pairs)

    class _config:
        def __init__(self):
            self.head_dim = key_value_pairs[0][0].shape[-1]
            self.num_attention_heads = key_value_pairs[0][0].shape[1]
            self.num_hidden_layers = len(key_value_pairs)

        def get_text_config(self, *args, **kwargs):
            return self

    assert max_cache_len is not None, (
        f"max_cache_len={max_cache_len} cannot be setup "
        f"automatically yet from shape {key_value_pairs[0][0].shape}"
    )
    torch._check(
        max_cache_len >= key_value_pairs[0][0].shape[2],
        (
            f"max_cache_len={max_cache_len} cannot be smaller "
            f"shape[2]={key_value_pairs[0][0].shape[2]} in shape "
            f"{key_value_pairs[0][0].shape}"
        ),
    )
    cache = transformers.cache_utils.StaticCache(
        config=_config(),
        max_batch_size=key_value_pairs[0][0].shape[0],
        device=key_value_pairs[0][0].device,
        dtype=key_value_pairs[0][0].dtype,
        max_cache_len=max_cache_len,
    )
    ca = CacheKeyValue(cache)
    if hasattr(cache, "layers") and len(ca.key_cache) == 0:
        # transformers>= 4.55.2, layers are empty
        for i, (key, value) in enumerate(key_value_pairs):
            cache.update(key, value, i)
        return cache

    torch._check(
        not hasattr(cache, "layers") or len(key_value_pairs) == len(cache.layers),
        lambda: (
            f"Length mismatch len(key_value_pairs)={len(key_value_pairs)}, "
            f"len(cache.layers)={len(cache.layers)}"
        ),
    )
    torch._check(
        len(key_value_pairs) == len(ca.key_cache),
        lambda: (
            f"Length mismatch len(key_value_pairs)={len(key_value_pairs)}, "
            f"len(ca.key_cache)={len(ca.key_cache)}"
        ),
    )
    torch._check(
        len(key_value_pairs) == len(ca.value_cache),
        lambda: (
            f"Length mismatch len(key_value_pairs)={len(key_value_pairs)}, "
            f"len(ca.value_cache)={len(ca.value_cache)}"
        ),
    )
    for i in range(len(key_value_pairs)):
        assert (
            key_value_pairs[i][0].shape == key_value_pairs[i][1].shape
        ), f"Shape mismatch {key_value_pairs[i][0].shape} != {key_value_pairs[i][1].shape}"
        d = key_value_pairs[i][1].shape[2]
        ca.key_cache[i][:, :, :d, :] = key_value_pairs[i][0]
        ca.value_cache[i][:, :, :d, :] = key_value_pairs[i][1]
    if hasattr(cache, "layers") and len(key_value_pairs) < len(cache.layers):
        # The cache constructor contains the two following lines
        # (in cache_utils.py) which append empty layers when the cache is
        # initialized. We need to remove them.
        # self.num_hidden_layers = getattr(config, "num_hidden_layers", 1)
        # self.append_new_layers(self.num_hidden_layers - 1)
        cache.layers[:] = cache.layers[-len(key_value_pairs) :]
    assert not hasattr(cache, "layers") or len(key_value_pairs) == len(cache.layers), (
        f"Unexpected number of layers in the cache ({len(cache.layers)}), "
        f"{len(key_value_pairs)} expected."
    )
    return finalize_cache(cache)


if hasattr(transformers.cache_utils, "EncoderDecoderCache"):

    def make_encoder_decoder_cache(
        self_attention_cache: transformers.cache_utils.DynamicCache,
        cross_attention_cache: transformers.cache_utils.DynamicCache,
    ) -> transformers.cache_utils.EncoderDecoderCache:
        """Creates an EncoderDecoderCache."""
        return transformers.cache_utils.EncoderDecoderCache(
            # self_attention_cache=self_attention_cache,
            # cross_attention_cache=cross_attention_cache
            self_attention_cache,
            cross_attention_cache,
        )

else:
    make_encoder_decoder_cache = None  # type: ignore[assignment]


def make_mamba_cache(
    key_value_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
) -> "MambaCache":  # noqa: F821
    "Creates a ``MambaCache``."
    # import is moved here because this part is slow.
    try:
        from transformers.models.mamba.modeling_mamba import MambaCache
    except ImportError:
        from transformers.cache_utils import MambaCache
    dtype = key_value_pairs[0][0].dtype

    class _config:
        def __init__(self):
            self.intermediate_size = key_value_pairs[0][0].shape[1]
            self.conv_kernel = key_value_pairs[0][0].shape[-1]
            self.state_size = key_value_pairs[0][1].shape[-1]
            self.num_hidden_layers = len(key_value_pairs)
            self.dtype = dtype

        def get_text_config(self, *args, **kwargs):
            return self

    cache = MambaCache(
        _config(),
        max_batch_size=key_value_pairs[0][0].shape[0],
        device=key_value_pairs[0][0].device,
        dtype=dtype,
    )
    for i in range(len(key_value_pairs)):
        assert cache.conv_states[i].dtype == dtype, (
            f"Type mismatch for cache.conv_states[{i}].dtype="
            f"{cache.conv_states[i].dtype} != {dtype}"
        )
        assert cache.ssm_states[i].dtype == dtype, (
            f"Type mismatch for cache.ssm_states[{i}].dtype="
            f"{cache.ssm_states[i].dtype} != {dtype}"
        )
        assert cache.conv_states[i].shape == key_value_pairs[i][0].shape, (
            f"Shape mismatch, expected {cache.conv_states[i].shape}, "
            f"got {key_value_pairs[i][0].shape}"
        )
        cache.conv_states[i][:, :, :] = key_value_pairs[i][0]
        assert cache.ssm_states[i].shape == key_value_pairs[i][1].shape, (
            f"Shape mismatch, expected {cache.ssm_states[i].shape}, "
            f"got {key_value_pairs[i][1].shape}"
        )
        cache.ssm_states[i][:, :, :] = key_value_pairs[i][1]
    return finalize_cache(cache)


if hasattr(transformers.cache_utils, "SlidingWindowCache"):

    def make_sliding_window_cache(
        key_value_pairs: Union[List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]],
    ) -> transformers.cache_utils.SlidingWindowCache:
        "Creates a :class:`transformers.cache_utils.SlidingWindowCache`."
        key_value_pairs = _preprocess_key_value_pairs(key_value_pairs)

        class _config:
            def __init__(self):
                self.head_dim = key_value_pairs[0][0].shape[-1]
                self.num_attention_heads = key_value_pairs[0][0].shape[1]
                self.num_hidden_layers = len(key_value_pairs)
                self.sliding_window = key_value_pairs[0][0].shape[2]

            def get_text_config(self, *args, **kwargs):
                return self

        cache = transformers.cache_utils.SlidingWindowCache(
            config=_config(),
            max_batch_size=key_value_pairs[0][0].shape[0],
            max_cache_len=key_value_pairs[0][0].shape[2],  # same as sliding_window
            device=key_value_pairs[0][0].device,
            dtype=key_value_pairs[0][0].dtype,
        )
        ca = CacheKeyValue(cache)
        if hasattr(cache, "layers") and len(ca.key_cache) == 0:
            # transformers>= 4.55.2, layers are empty
            cache_position = torch.arange(key_value_pairs[0][0].shape[2], dtype=torch.int64)
            for i, (key, value) in enumerate(key_value_pairs):
                cache.update(key, value, i, cache_kwargs={"cache_position": cache_position})
            return cache

        for i in range(len(key_value_pairs)):
            assert ca.key_cache[i].shape == key_value_pairs[i][0].shape, (
                f"Shape mismatch, expected {cache.key_cache[i].shape}, "
                f"got {key_value_pairs[i][0].shape}"
            )
            ca.key_cache[i][:, :, :, :] = key_value_pairs[i][0]
            assert ca.value_cache[i].shape == key_value_pairs[i][1].shape, (
                f"Shape mismatch, expected {cache.value_cache[i].shape}, "
                f"got {key_value_pairs[i][1].shape}"
            )
            ca.value_cache[i][:, :, :, :] = key_value_pairs[i][1]
        if hasattr(cache, "layers") and len(key_value_pairs) < len(cache.layers):
            # The cache constructor contains the two following lines
            # (in cache_utils.py) which append empty layers when the cache is
            # initialized. We need to remove them.
            # self.num_hidden_layers = getattr(config, "num_hidden_layers", 1)
            # self.append_new_layers(self.num_hidden_layers - 1)
            cache.layers[:] = cache.layers[-len(key_value_pairs) :]
        assert not hasattr(cache, "layers") or len(key_value_pairs) == len(cache.layers), (
            f"Unexpected number of layers in the cache ({len(cache.layers)}), "
            f"{len(key_value_pairs)} expected."
        )
        return finalize_cache(cache)

else:
    make_sliding_window_cache = None  # type: ignore[assignment]

if hasattr(transformers.cache_utils, "HybridCache"):

    def make_hybrid_cache(
        key_value_pairs: Union[List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]],
        max_cache_len: Optional[int] = None,
        max_batch_size: Optional[int] = None,
        sliding_window: Optional[int] = None,
    ) -> transformers.cache_utils.HybridCache:
        """
        Creates an instance of :class:`transformers.cache_utils.HybridCache`.
        This version is valid for ``transformers < 4.50``.

        :param key_value_pairs: list of pairs of (key, values)
        :return: :class:`transformers.cache_utils.HybridCache`

        Example:

        .. runpython::
            :showcode:

            import torch
            from onnx_diagnostic.helpers import string_type
            from onnx_diagnostic.helpers.cache_helper import make_hybrid_cache

            n_layers = 2
            bsize, nheads, slen, dim = 2, 4, 3, 7

            past_key_values = make_hybrid_cache(
                [
                    (
                        torch.randn(bsize, nheads, slen, dim),
                        torch.randn(bsize, nheads, slen, dim),
                    )
                    for i in range(n_layers)
                ]
            )
            print(string_type(past_key_values, with_shape=True))

        This part defines how the shapes are working in one HybridCache.

        .. code-block:: python

            self.max_cache_len = (
                max_cache_len if max_cache_len is not None else config.max_position_embeddings)

            # Sliding layers can't be larger than the overall max cache len
            self.sliding_window_len = min(config.sliding_window, self.max_cache_len)
            self.max_batch_size = max_batch_size

            self.head_dim = (
                config.head_dim if hasattr(config, "head_dim")
                else config.hidden_size // config.num_attention_heads
            )

            self._dtype = dtype
            self.num_key_value_heads = (
                config.num_attention_heads
                if getattr(config, "num_key_value_heads", None) is None
                else config.num_key_value_heads
            )

            # If the attribute does not exist in the config, fallback to a simple StaticCache
            if hasattr(config, "layer_types"):
                self.is_sliding = [
                    layer_type != "full_attention" for layer_type in config.layer_types]
            else:
                self.is_sliding = [False] * config.num_hidden_layers

            self.key_cache: list[torch.Tensor] = []
            self.value_cache: list[torch.Tensor] = []
            global_cache_shape = (self.max_batch_size, self.num_key_value_heads,
                                    self.max_cache_len, self.head_dim)
            sliding_cache_shape = (self.max_batch_size, self.num_key_value_heads,
                                    self.sliding_window_len, self.head_dim)
            self.sliding_window = min(config.sliding_window, max_cache_len)
            device = torch.device(device) if device is not None else None
            for i in range(config.num_hidden_layers):
                layer_device = layer_device_map[i] if layer_device_map is not None else device
                cache_shape = sliding_cache_shape if self.is_sliding[i] else global_cache_shape
                new_layer_key_cache = torch.zeros(
                    cache_shape, dtype=self._dtype, device=layer_device)
                new_layer_value_cache = torch.zeros(
                    cache_shape, dtype=self._dtype, device=layer_device)
                torch._dynamo.mark_static_address(new_layer_key_cache)
                torch._dynamo.mark_static_address(new_layer_value_cache)
                self.key_cache.append(new_layer_key_cache)
                self.value_cache.append(new_layer_value_cache)
        """
        key_value_pairs = _preprocess_key_value_pairs(key_value_pairs)
        layer_types = None
        if key_value_pairs:
            assert (
                not max_batch_size and not max_cache_len
            ), "key_value_pairs is not empty, do not specify max_cache_len and max_batch_size"
            max_batch_size = key_value_pairs[0][0].shape[0]
            sets_of_dim = set(kv[0].shape[2] for kv in key_value_pairs)
            if len(sets_of_dim) == 1:
                max_cache_len = sets_of_dim.pop()
                sliding_window = max_cache_len
            else:
                assert (
                    len(sets_of_dim) == 2
                ), f"Not implemented for more than 2 dimensions {sets_of_dim}"
                max_cache_len = max(sets_of_dim)
                sliding_window = min(sets_of_dim)
                layer_types = [
                    "full_attention" if i == max_cache_len else "sliding_attention"
                    for i in [kv[0].shape[2] for kv in key_value_pairs]
                ]
        else:
            assert (
                max_batch_size and max_cache_len
            ), "key_value_pairs is empty, max_batch_size and max_cache_len are required"
            if sliding_window is None:
                sliding_window = max_cache_len
        _max_cache_len = max_cache_len
        _sliding_window = sliding_window

        class _config:
            max_cache_len = _max_cache_len
            batch_size = max_batch_size
            num_heads = key_value_pairs[0][0].shape[1] if key_value_pairs else None
            head_dim = key_value_pairs[0][0].shape[-1] if key_value_pairs else None
            num_attention_heads = key_value_pairs[0][1].shape[1] if key_value_pairs else None
            num_hidden_layers = len(key_value_pairs)
            sliding_window = _sliding_window
            num_key_value_heads = key_value_pairs[0][1].shape[1]  # transformers 4.48.3

            def get_text_config(self, *args, **kwargs):
                return self

        if layer_types:
            _config.layer_types = layer_types  # type: ignore[attr-defined]

        cache = transformers.cache_utils.HybridCache(
            config=_config(), max_cache_len=max_cache_len, max_batch_size=max_batch_size
        )
        for i, (key, value) in enumerate(key_value_pairs):
            cache.update(
                key,
                value,
                i,
                cache_kwargs={
                    "cache_position": torch.arange(0, key.shape[2], dtype=torch.int64).to(
                        key.device
                    )
                },
            )
        if hasattr(cache, "layers") and len(key_value_pairs) < len(cache.layers):
            # The cache constructor contains the two following lines
            # (in cache_utils.py) which append empty layers when the cache is
            # initialized. We need to remove them.
            # self.num_hidden_layers = getattr(config, "num_hidden_layers", 1)
            # self.append_new_layers(self.num_hidden_layers - 1)
            cache.layers[:] = cache.layers[-len(key_value_pairs) :]
        assert not hasattr(cache, "layers") or len(key_value_pairs) == len(cache.layers), (
            f"Unexpected number of layers in the cache ({len(cache.layers)}), "
            f"{len(key_value_pairs)} expected."
        )
        return finalize_cache(cache)

else:
    make_hybrid_cache = None  # type: ignore[assignment]


def finalize_cache(cache: transformers.cache_utils.Cache) -> transformers.cache_utils.Cache:
    """
    Ensures the created cache is consistent.
    Returns the cache modified inplace.
    """
    if (
        hasattr(cache, "layer_class_to_replicate")
        and hasattr(cache, "layers")
        and cache.layers
        and not cache.layer_class_to_replicate
    ):
        # This is used to expand the cache when it does not contains enough layers.
        # This is needed since transformers>4.55.3
        cache.layer_class_to_replicate = cache.layers[0].__class__
    return cache
