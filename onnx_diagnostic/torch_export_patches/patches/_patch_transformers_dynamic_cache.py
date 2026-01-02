from typing import List, Optional, Tuple
import packaging.version as pv
import torch
import transformers
from .patch_helper import _has_transformers

patch_is_initialized = _has_transformers("4.56.99")
patch_DynamicCache = pv.Version(transformers.__version__) < pv.Version("4.51")

try:
    # transformers>= 4.55.1
    from transformers.cache_utils import DynamicLayer

    patch_DynamicLayer = hasattr(DynamicLayer, "lazy_initialization")
except ImportError:
    patch_DynamicLayer = False


if patch_DynamicLayer:

    class patched_DynamicLayer:
        _PATCHES_ = ["lazy_initialization"]
        _PATCHED_CLASS_ = DynamicLayer

        def lazy_initialization(self, key_states: torch.Tensor):
            self.dtype, self.device = key_states.dtype, key_states.device
            assert (
                hasattr(key_states, "shape") and key_states is not None
            ), f"Attribute 'shape' is wrong for type {type(key_states)}"
            like = torch.narrow(key_states, dim=-2, start=0, length=0)
            # PATCHED: used a tensor with an empty shape and not en empty list to initialize
            if isinstance(key_states, torch._subclasses.fake_tensor.FakeTensor):
                with key_states.fake_mode:
                    self.keys = torch.empty_like(like, dtype=self.dtype, device=self.device)
                    self.values = torch.empty_like(like, dtype=self.dtype, device=self.device)
            else:
                self.keys = torch.empty_like(like, dtype=self.dtype, device=self.device)
                self.values = torch.empty_like(like, dtype=self.dtype, device=self.device)
            if patch_is_initialized:
                self.is_initialized = True


if patch_DynamicCache:
    from typing import Any, Dict
    from transformers.cache_utils import DynamicCache

    class patched_DynamicCache:
        """
        Applies modifications implemented in PR
        `transformers/#36652 <https://github.com/huggingface/transformers/pull/36652>`_.
        """

        _PATCHES_ = ["reorder_cache", "update", "crop", "from_batch_splits", "get_seq_length"]
        _PATCHED_CLASS_ = transformers.cache_utils.DynamicCache

        def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
            """Returns the sequence length of the cached states.
            A layer index can be optionally passed."""
            # TODO: deprecate this function in favor of `cache_position`
            is_empty_layer = (
                len(self.key_cache) == 0  # no cache in any layer
                or len(self.key_cache)
                <= layer_idx  # skipped `layer_idx` and hasn't run a layer with cache after it
                or self.key_cache[layer_idx].numel() == 0  # the layer has no cache
            )
            layer_seq_length = self.key_cache[layer_idx].shape[-2] if not is_empty_layer else 0
            return layer_seq_length

        def reorder_cache(self, beam_idx: torch.LongTensor):
            """Reorders the cache for beam search, given the selected beam indices."""
            for layer_idx in range(len(self.key_cache)):
                if self.key_cache[layer_idx].numel():
                    device = self.key_cache[layer_idx].device
                    self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(
                        0, beam_idx.to(device)
                    )
                if self.value_cache[layer_idx].numel():
                    device = self.value_cache[layer_idx].device
                    self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(
                        0, beam_idx.to(device)
                    )

        def update(
            self,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            layer_idx: int,
            cache_kwargs: Optional[Dict[str, Any]] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Updates the cache with the new `key_states`
            and `value_states` for the layer `layer_idx`.
            Parameters:
                key_states (`torch.Tensor`):
                    The new key states to cache.
                value_states (`torch.Tensor`):
                    The new value states to cache.
                layer_idx (`int`):
                    The index of the layer to cache the states for.
                cache_kwargs (`Dict[str, Any]`, `optional`):
                    Additional arguments for the cache subclass.
                    No additional arguments are used in `DynamicCache`.
            Return:
                A tuple containing the updated key and value states.
            """
            # Update the number of seen tokens
            if layer_idx == 0:
                if hasattr(self, "_seen_tokens"):
                    self._seen_tokens += key_states.shape[-2]

            # Update the cache
            if key_states is not None:
                if len(self.key_cache) <= layer_idx:
                    # There may be skipped layers, fill them with empty lists
                    for _ in range(len(self.key_cache), layer_idx):
                        self.key_cache.append(torch.tensor([], dtype=key_states.dtype))
                        self.value_cache.append(torch.tensor([], dtype=key_states.dtype))
                    self.key_cache.append(key_states)
                    self.value_cache.append(value_states)
                elif not self.key_cache[
                    layer_idx
                ].numel():  # prefers not t.numel() to len(t) == 0 to export the model
                    # fills previously skipped layers; checking for tensor causes errors
                    self.key_cache[layer_idx] = key_states
                    self.value_cache[layer_idx] = value_states
                else:
                    torch._check(
                        len(self.key_cache[layer_idx].shape) == len(key_states.shape),
                        lambda: (
                            f"Rank mismatch len(self.key_cache[layer_idx].shape)="
                            f"{len(self.key_cache[layer_idx].shape)}, "
                            f"len(key_states.shape)={len(key_states.shape)}"
                        ),
                    )
                    self.key_cache[layer_idx] = torch.cat(
                        [self.key_cache[layer_idx], key_states], dim=-2
                    )
                    self.value_cache[layer_idx] = torch.cat(
                        [self.value_cache[layer_idx], value_states], dim=-2
                    )
            return self.key_cache[layer_idx], self.value_cache[layer_idx]

        def crop(self, max_length: int):
            """Crop the past key values up to a new `max_length`
            in terms of tokens. `max_length` can also be
            negative to remove `max_length` tokens.
            This is used in assisted decoding and contrastive search.
            """
            # In case it is negative
            if max_length < 0:
                max_length = self.get_seq_length() - abs(max_length)

            if self.get_seq_length() <= max_length:
                return

            if hasattr(self, "_seen_tokens"):
                self._seen_tokens = max_length
            for idx in range(len(self.key_cache)):
                if self.key_cache[idx].numel():
                    self.key_cache[idx] = self.key_cache[idx][..., :max_length, :]
                    self.value_cache[idx] = self.value_cache[idx][..., :max_length, :]

        @classmethod
        def from_batch_splits(cls, splits: List[DynamicCache]) -> DynamicCache:
            """This is the opposite of the above `batch_split()` method.
            This will be used by `stack_model_outputs` in
            `generation.utils`"""
            cache = cls()
            for idx in range(len(splits[0])):
                key_cache = [
                    current.key_cache[idx]
                    for current in splits
                    if current.key_cache[idx].numel()
                ]
                value_cache = [
                    current.value_cache[idx]
                    for current in splits
                    if current.value_cache[idx].numel()
                ]
                if key_cache != []:
                    layer_keys = torch.cat(key_cache, dim=0)
                    layer_values = torch.cat(value_cache, dim=0)
                    cache.update(layer_keys, layer_values, idx)
            return cache
