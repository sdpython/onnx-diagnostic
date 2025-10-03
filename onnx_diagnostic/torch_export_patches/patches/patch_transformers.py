import inspect
import math
import os
from dataclasses import dataclass
from functools import wraps
from typing import Callable, List, Optional, Tuple, Union
import packaging.version as pv
import torch
import transformers
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.cache_utils import StaticCache, Cache
from transformers.generation.utils import (
    GenerateNonBeamOutput,
    GenerationConfig,
    StoppingCriteriaList,
    LogitsProcessorList,
)

try:
    from transformers.cache_utils import parse_processor_args  # noqa: F401

    patch_parse_processor_args = True
except ImportError:
    patch_parse_processor_args = False

try:
    import transformers.masking_utils

    patch_masking_utils = True
except ImportError:
    patch_masking_utils = False


try:
    # transformers>= 4.55.1
    from transformers.cache_utils import DynamicLayer

    patch_DynamicLayer = hasattr(DynamicLayer, "lazy_initialization")
except ImportError:
    patch_DynamicLayer = False

from ...ext_test_case import has_transformers
from ...helpers.torch_helper import is_torchdynamo_exporting

patch_is_initialized = pv.Version(transformers.__version__) > pv.Version("4.56.99")


if patch_masking_utils:
    # Introduced in 4.52
    from transformers.masking_utils import (
        causal_mask_function,
        padding_mask_function,
        and_masks,
        _ignore_causal_mask_sdpa,
        prepare_padding_mask,
    )

    def patched__vmap_for_bhqkv(mask_function: Callable, bh_indices: bool = True) -> Callable:
        """manual patch for function ``transformers.masking_utils._vmap_for_bhqkv``."""
        from ...helpers import string_type

        dimensions: List[Tuple[Optional[int], ...]] = [
            (None, None, None, 0),
            (None, None, 0, None),
        ]
        if bh_indices:
            dimensions.extend([(None, 0, None, None), (0, None, None, None)])
        # reshape
        dimensions = [tuple(1 if d is None else -1 for d in shape) for shape in dimensions]
        dimensions = tuple(reversed(dimensions))
        indices = tuple(shape.index(-1) for shape in dimensions)

        # unsqueeze
        udimensions = [
            tuple(di for di, d in enumerate(shape) if d == 1) for shape in dimensions
        ]

        def vector_mask_function(
            *args, mask_function=mask_function, dimensions=dimensions, indices=indices
        ):
            assert len(args) == len(dimensions) == len(udimensions), (
                f"Mismatch between args={string_type(args)} and dimensions={dimensions} "
                f"and udimensions={udimensions}."
            )
            assert len(indices) == len(args), (
                f"Mismatch between args={string_type(args)} and indices={indices}, "
                f"they should have the same length."
            )
            for a in args:
                assert (
                    a.ndim == 1
                ), f"Expected a tensor with 1 dimension not {string_type(a, with_shape=True)}"
                torch._check(a.shape[0] > 0)

            new_args = [a.reshape(shape) for a, shape in zip(args, dimensions)]
            # new_args = [
            #    a.unsqueeze(dims[0]).unsqueeze(dims[1]).unsqueeze(dims[2])
            #    for a, dims in zip(args, udimensions)
            # ]
            max_shape = tuple(args[i].shape[0] for i in indices)
            # if is_torchdynamo_exporting():
            #     for a in args:
            #         # The exporter should export with a dimension > 1
            #         # to make sure it is dynamic.
            #         torch._check(a.shape[0] > 1)
            expanded_args = [a.expand(max_shape) for a in new_args]
            return mask_function(*expanded_args)

        return vector_mask_function

    def patched_eager_mask(
        batch_size: int,
        cache_position: torch.Tensor,
        kv_length: int,
        kv_offset: int = 0,
        mask_function: Callable = causal_mask_function,
        attention_mask: Optional[torch.Tensor] = None,
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ) -> torch.Tensor:
        """manual patch for function ``transformers.masking_utils.eager_mask``."""
        # The masks for eager attention are simply boolean mask from sdpa, casted to 0 and -inf
        _ = kwargs.pop("allow_is_causal_skip", None)
        # PATCHED: this line called the patched version of sdpa_mask
        mask = patched_sdpa_mask_recent_torch(
            batch_size=batch_size,
            cache_position=cache_position,
            kv_length=kv_length,
            kv_offset=kv_offset,
            mask_function=mask_function,
            attention_mask=attention_mask,
            allow_is_causal_skip=False,
            allow_torch_fix=False,
            **kwargs,
        )
        min_dtype = torch.finfo(dtype).min
        # PATCHED: the following line
        # we need 0s where the tokens should be taken into account,
        # and -inf otherwise (mask is already of boolean type)
        # mask =
        #   torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), min_dtype)
        mask = (~mask).to(dtype) * min_dtype
        return mask

    def patched_sdpa_mask_recent_torch(
        batch_size: int,
        cache_position: torch.Tensor,
        kv_length: int,
        kv_offset: int = 0,
        mask_function: Callable = causal_mask_function,
        attention_mask: Optional[torch.Tensor] = None,
        local_size: Optional[int] = None,
        allow_is_causal_skip: bool = True,
        **kwargs,
    ) -> Optional[torch.Tensor]:
        """manual patch for function ``transformers.masking_utils.sdpa_mask_recent_torch``."""
        q_length = cache_position.shape[0]
        padding_mask = prepare_padding_mask(attention_mask, kv_length, kv_offset, _slice=False)
        if allow_is_causal_skip and _ignore_causal_mask_sdpa(
            padding_mask, q_length, kv_length, kv_offset, local_size
        ):
            return None
        kv_arange = torch.arange(kv_length, device=cache_position.device)
        kv_arange += kv_offset
        if padding_mask is not None:
            mask_function = and_masks(mask_function, padding_mask_function(padding_mask))
        batch_arange = torch.arange(batch_size, device=cache_position.device)
        head_arange = torch.arange(1, device=cache_position.device)
        # PATCHED: this line calls the patched version of vmap_for_bhqkv
        causal_mask = patched__vmap_for_bhqkv(mask_function)(
            batch_arange, head_arange, cache_position, kv_arange
        )
        return causal_mask


if patch_parse_processor_args:

    def _init_cache_inspect():
        res = {}
        for processor_class in transformers.cache_utils.PROCESSOR_CLASS_MAP.values():
            try:
                params = list(inspect.signature(processor_class.__init__).parameters)[2:]
                res[processor_class.__init__] = params
            except Exception:
                res[processor_class.__init__] = None
        return res

    _cache_inspect = _init_cache_inspect()

    def patched_parse_processor_args(
        processor_class: Optional[type["CacheProcessor"]], kwargs: dict  # noqa: F821
    ) -> tuple[dict, dict]:
        """[patch:transformers.cache_utils.parse_processor_args]"""
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
        #
        # The patch is caching the signature to avoid any call to inspect.
        if processor_class is None:
            return {}, kwargs
        params = _cache_inspect[processor_class.__init__]
        if params is None:
            return {}, kwargs
        processor_kwargs = {k: kwargs[k] for k in params if k in kwargs}
        remaining_kwargs = {k: v for k, v in kwargs.items() if k not in processor_kwargs}
        return processor_kwargs, remaining_kwargs


if patch_DynamicLayer:

    class patched_DynamicLayer:
        _PATCHES_ = ["lazy_initialization"]
        _PATCHED_CLASS_ = DynamicLayer

        def lazy_initialization(self, key_states: torch.Tensor):
            self.dtype, self.device = key_states.dtype, key_states.device
            new_shape = list(key_states.shape)
            new_shape[-2] = 0
            # PATCHED: used a tensor with an empty shape and not en empty list to initialize
            self.keys = torch.empty(new_shape, dtype=self.dtype, device=self.device)
            self.values = torch.empty(new_shape, dtype=self.dtype, device=self.device)
            if patch_is_initialized:
                self.is_initialized = True


def _patch_make_causal_mask(
    input_ids_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
    sliding_window: Optional[int] = None,
):
    """Patched method."""
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)

    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat(
            [
                torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device),
                mask,
            ],
            dim=-1,
        )

    if sliding_window is not None:
        diagonal = past_key_values_length - sliding_window - 1

        context_mask = torch.tril(torch.ones_like(mask, dtype=torch.bool), diagonal=diagonal)
        # PATCHED: removed if is_torchdynamo_compiling(): mask = mask.clone()
        # and used masked_fill instead of masked_fill_
        # In this case, the current implementation of torch fails (17/12/2024).
        # Try model Phi-3.5-Mini-Instruct.
        mask = mask.masked_fill(context_mask, torch.finfo(dtype).min)

    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


@dataclass
class patched_AttentionMaskConverter:
    """
    Patches
    ``transformers.modeling_attn_mask_utils.AttentionMaskConverter._make_causal_mask``.
    """

    # This method was fixed in 4.51 at least.
    _PATCHES_ = ["_make_causal_mask"] if not has_transformers("4.48.3") else []
    _PATCHED_CLASS_ = AttentionMaskConverter

    @staticmethod
    def _make_causal_mask(
        *args,
        **kwargs,
        # input_ids_shape: torch.Size,
        # dtype: torch.dtype,
        # device: torch.device,
        # past_key_values_length: int = 0,
        # sliding_window: Optional[int] = None,
    ):
        """
        Patched method.

        This static method may be called with ``AttentionMaskConverter._make_causal_mask``
        or ``self._make_causal_mask``. That changes this argument is receives.
        That should not matter but...
        The patch should be implemented in another way. static methods do not play well
        with a simple replacement.
        Fortunately, this patch does not seem to be needed anymore with transformers>=4.48.3.
        """
        if args:
            index = 0 if isinstance(args[0], (tuple, torch.Size)) else 1
            names = [
                "input_ids_shape",
                "dtype",
                "device",
                "past_key_values_length",
                "sliding_window",
            ]
            for i, a in enumerate(args):
                if i < index:
                    continue
                kwargs[names[i - index]] = a
        return _patch_make_causal_mask(**kwargs)


if pv.Version(transformers.__version__) < pv.Version("4.51"):
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


class patched_GenerationMixin:
    """
    Applies modifications implemented in PR
    `transformers/#36652 <https://github.com/huggingface/transformers/pull/36652>`_.
    """

    _PATCHES_ = [
        "_cache_dependant_input_preparation",
        "_cache_dependant_input_preparation_exporting",
        (
            None
            if pv.Version(transformers.__version__) >= pv.Version("4.56")
            else "prepare_inputs_for_generation"
        ),
        (
            "_sample"
            if pv.Version(transformers.__version__) == pv.Version("4.57.0.dev0")
            else None
        ),
    ]
    _PATCHED_CLASS_ = transformers.generation.utils.GenerationMixin

    def _cache_dependant_input_preparation(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: Optional[torch.FloatTensor],
        cache_position: Optional[torch.LongTensor],
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        """
        Generic cache-dependent input preparation
        The code is put in a separate function to allow granular unit testing
        as it needs a different implementation to be exportable.

        If we have cache: let's slice `input_ids` through `cache_position`,
        to keep only the unprocessed tokens
        - Exception 1: when passing input_embeds,
          input_ids may be missing entries
        - Exception 2: some generation methods do special slicing of input_ids,
          so we don't need to do it here
        - Exception 3: with synced GPUs cache_position may go out of bounds,
          but we only want dummy token in that case.
        - Exception 4: If input_embeds are passed then slice it through
          `cache_position`, to keep only the unprocessed tokens and
          generate the first token for each sequence.
          Later use the generated Input ids for continuation.

        The current implementation does not rely on ``self`` and could be
        a class method. It is left as a standard method to be easily rewritten.
        """
        if is_torchdynamo_exporting():
            return self._cache_dependant_input_preparation_exporting(
                input_ids, inputs_embeds, cache_position
            )
        if inputs_embeds is not None and input_ids.shape[1] == 0:  # Exception 4
            inputs_embeds = inputs_embeds[:, -cache_position.shape[0] :]
        elif inputs_embeds is not None or (  # Exception 1
            cache_position[-1] >= input_ids.shape[1]
        ):  # Exception 3
            input_ids = input_ids[:, -cache_position.shape[0] :]
        elif (
            input_ids.shape[1] != cache_position.shape[0]
        ):  # Default case (the "else", a no op, is Exception 2)
            input_ids = input_ids[:, cache_position]
        return inputs_embeds, input_ids

    def _cache_dependant_input_preparation_exporting(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: Optional[torch.FloatTensor],
        cache_position: Optional[torch.LongTensor],
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        """
        This method implements method ``_cache_dependant_input_preparation``
        with :func:`torch.cond` to make it exportable with :func:`torch.export.export`.
        The code is put in a separate function to allow granular unit testing.
        """
        if inputs_embeds is None:
            input_ids = input_ids[:, cache_position]
        else:
            # This is the code we need to implemented with torch.cond.
            # if input_ids.shape[1] == 0:
            #     inputs_embeds = inputs_embeds[:, -cache_position.shape[0] :]
            # else:
            #     if cache_position[-1] >= input_ids.shape[1]:
            #         input_ids = input_ids[:, -cache_position.shape[0] :]
            #     else:
            #         if input_ids.shape[1] != cache_position.shape[0]:
            #             input_ids = input_ids[:, cache_position]
            def branch_1(inputs_embeds, cache_position):
                return inputs_embeds[:, -cache_position.shape[0] :]

            def branch_2(input_ids, cache_position):
                return input_ids[:, -cache_position.shape[0] :]

            def branch_3(input_ids, cache_position):
                return input_ids[:, cache_position]

            inputs_embeds, input_ids = torch.cond(
                input_ids.shape[1] == 0,
                (
                    lambda input_ids, inputs_embeds, cache_position: (
                        branch_1(inputs_embeds, cache_position),
                        input_ids,
                    )
                ),
                (
                    lambda input_ids, inputs_embeds, cache_position: (
                        inputs_embeds,
                        torch.cond(
                            cache_position[-1] >= input_ids.shape[1],
                            branch_2,
                            lambda input_ids, cache_position: (
                                torch.cond(
                                    input_ids.shape[1] != cache_position.shape[0],
                                    branch_3,
                                    (lambda input_ids, cache_position: input_ids),
                                    [input_ids, cache_position],
                                )
                            ),
                            [input_ids, cache_position],
                        ),
                    )
                ),
                [input_ids, inputs_embeds, cache_position],
            )
        return inputs_embeds, input_ids

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        """
        Prepare the model inputs for generation.
        In includes operations like computing the 4D attention mask or
        slicing inputs given the existing cache.

        See the forward pass in the model documentation
        for expected arguments (different models might have different
        requirements for e.g. `past_key_values`).
        This function should work as is for most LLMs.
        """

        # 1. Handle BC:
        model_inputs = {}
        # - some models don't have `Cache` support
        # (which implies they don't expect `cache_position` in `forward`)
        if getattr(self, "_supports_cache_class", False):
            model_inputs["cache_position"] = cache_position
        # - `cache_position` was not a mandatory input in
        # `prepare_inputs_for_generation` for those models, and this
        #   function may be called outside of `generate`.
        # Handle most use cases by creating `cache_position` on the fly
        #   (this alternative is not as robust as calling
        # `generate` and letting it create `cache_position`)
        elif cache_position is None:
            past_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
            cache_position = torch.arange(
                past_length, input_ids.shape[1], dtype=torch.long, device=input_ids.device
            )

        # 2. Generic cache-dependent input preparation
        if past_key_values is not None:
            model_inputs["past_key_values"] = past_key_values
            inputs_embeds, input_ids = self._cache_dependant_input_preparation(
                input_ids, inputs_embeds, cache_position
            )

        # 3. Prepare base model inputs
        input_ids_key = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
        # if `inputs_embeds` are passed, we only want
        # to use them in the 1st generation step for every prompt.
        if not self.config.is_encoder_decoder:
            if inputs_embeds is not None and len(cache_position) == inputs_embeds.shape[1]:
                model_inputs[input_ids_key] = None
                model_inputs["inputs_embeds"] = inputs_embeds
            else:
                # `clone` calls in this function ensure a consistent stride. See #32227
                model_inputs[input_ids_key] = input_ids.clone(
                    memory_format=torch.contiguous_format
                )
                model_inputs["inputs_embeds"] = None
        else:
            model_inputs[input_ids_key] = input_ids.clone(
                memory_format=torch.contiguous_format
            )

        # 4. Create missing `position_ids` on the fly
        encoder_attention_mask = attention_mask if self.config.is_encoder_decoder else None
        attention_mask = (
            kwargs.pop("decoder_attention_mask", None)
            if self.config.is_encoder_decoder
            else attention_mask
        )
        attention_mask_key = (
            "decoder_attention_mask" if self.config.is_encoder_decoder else "attention_mask"
        )
        position_ids_key = (
            "decoder_position_ids" if self.config.is_encoder_decoder else "position_ids"
        )
        if (
            attention_mask is not None
            and kwargs.get(position_ids_key) is None
            and position_ids_key in set(inspect.signature(self.forward).parameters.keys())
        ):
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            kwargs[position_ids_key] = (
                position_ids  # placed in kwargs for further processing (see below)
            )

        # 5. Slice model inputs if it's an input
        # that should have the same length as `input_ids`
        for model_input_name in ["position_ids", "token_type_ids", "decoder_position_ids"]:
            model_input = kwargs.get(model_input_name)
            if model_input is not None:
                if past_key_values is not None:
                    current_input_length = (
                        model_inputs["inputs_embeds"].shape[1]
                        if model_inputs.get("inputs_embeds") is not None
                        else model_inputs[input_ids_key].shape[1]
                    )
                    model_input = model_input[:, -current_input_length:]
                    model_input = model_input.clone(memory_format=torch.contiguous_format)
                model_inputs[model_input_name] = model_input

        # 6. Create 4D attention mask is we are using a
        # `StaticCache` (important for performant compiled forward pass)
        if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
                device = model_inputs["inputs_embeds"].device
            else:
                batch_size, sequence_length = model_inputs[input_ids_key].shape
                device = model_inputs[input_ids_key].device

            # Create the causal mask with fixed shape in advance,
            # to reduce recompilations. If the function to create
            # the 4D causal mask exists,
            # it should be present in the base model (XXXModel class).
            base_model = getattr(self, self.base_model_prefix, None)
            if base_model is None:
                causal_mask_creation_function = getattr(
                    self, "_prepare_4d_causal_attention_mask_with_cache_position", None
                )
            else:
                causal_mask_creation_function = getattr(
                    base_model, "_prepare_4d_causal_attention_mask_with_cache_position", None
                )
            if causal_mask_creation_function is None:
                pass
                # logger.warning_once(
                #     f"{self.__class__.__name__} has no "
                #     "`_prepare_4d_causal_attention_mask_with_cache_position` method "
                #     "defined in its base modeling class. "
                #     "Compiled forward passes will be sub-optimal. If you're "
                #     "writing code, see Llama for an example implementation. "
                #     "If you're a user, please report this "
                #     "issue on GitHub."
                # )
            else:
                attention_mask = causal_mask_creation_function(
                    attention_mask,
                    sequence_length=sequence_length,
                    target_length=past_key_values.get_max_cache_shape(),
                    dtype=self.dtype,
                    device=device,
                    cache_position=cache_position,
                    batch_size=batch_size,
                    config=self.config,
                    past_key_values=past_key_values,
                )
        if attention_mask is not None:
            model_inputs[attention_mask_key] = attention_mask

        if encoder_attention_mask is not None:
            model_inputs["attention_mask"] = encoder_attention_mask

        # 7. Forward ALL kwargs that are uninitialized (e.g. `use_cache`).
        for key, value in kwargs.items():
            if key not in model_inputs:
                model_inputs[key] = value

        # 8. Remove unexpected `generate` inputs (TODO @joao: fix trainer and examples)
        model_inputs.pop("labels", None)
        return model_inputs

    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: "LogitsProcessorList",  # noqa: F821
        stopping_criteria: "StoppingCriteriaList",  # noqa: F821
        generation_config: "GenerationConfig",  # noqa: F821
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,  # noqa: F821
        **model_kwargs,
    ) -> Union["GenerateNonBeamOutput", torch.LongTensor]:  # noqa: F821
        """
        2025/09/29: updates for Gemma3 models, fix for eager mode as well as the export.
        """
        # init values
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        has_eos_stopping_criteria = any(
            hasattr(criteria, "eos_token_id") for criteria in stopping_criteria
        )
        do_sample = generation_config.do_sample

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = (
            () if (return_dict_in_generate and output_hidden_states) else None
        )

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = (
                model_kwargs["encoder_outputs"].get("attentions")
                if output_attentions
                else None
            )
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states")
                if output_hidden_states
                else None
            )

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape[:2]
        this_peer_finished = False
        unfinished_sequences = torch.ones(
            batch_size, dtype=torch.long, device=input_ids.device
        )
        model_kwargs = self._get_initial_cache_position(
            cur_len, input_ids.device, model_kwargs
        )

        model_forward = self.__call__
        compile_forward = self._valid_auto_compile_criteria(model_kwargs, generation_config)
        if compile_forward:
            os.environ["TOKENIZERS_PARALLELISM"] = "0"
            # If we use FA2 and a static cache, we cannot compile with fullgraph
            if self.config._attn_implementation == "flash_attention_2":
                # only raise warning if the user passed an explicit compile-config
                if (
                    generation_config.compile_config is not None
                    and generation_config.compile_config.fullgraph
                ):
                    generation_config.compile_config.fullgraph = False
            model_forward = self.get_compiled_call(generation_config.compile_config)

        if generation_config.prefill_chunk_size is not None:
            model_kwargs = self._prefill_chunking(input_ids, generation_config, **model_kwargs)
            is_prefill = False
        else:
            is_prefill = True

        while self._has_unfinished_sequences(
            this_peer_finished, synced_gpus, device=input_ids.device
        ):
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            if is_prefill:
                outputs = self(**model_inputs, return_dict=True)
                is_prefill = False
            else:
                outputs = model_forward(**model_inputs, return_dict=True)

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            if synced_gpus and this_peer_finished:
                continue

            next_token_logits = outputs.logits[:, -1, :].to(
                copy=True, dtype=torch.float32, device=input_ids.device
            )

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,)
                        if self.config.is_encoder_decoder
                        else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # token selection
            if do_sample:
                probs = torch.nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (
                    1 - unfinished_sequences
                )

            # update generated ids, model inputs, and length for next step
            # PATCHED: the two following lines, next_tokens can 2D already for this model
            next_tokens_2d = (
                next_tokens if len(next_tokens.shape) == 2 else next_tokens[:, None]
            )
            input_ids = torch.cat([input_ids, next_tokens_2d], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1

            # This is needed to properly delete outputs.logits which may be very large
            # for first iteration
            # Otherwise a reference to outputs is kept which keeps
            # the logits alive in the next iteration
            del outputs

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return transformers.generation.utils.GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return transformers.generation.utils.GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return input_ids


def patched__compute_dynamic_ntk_parameters(
    config: Optional[transformers.PretrainedConfig] = None,
    device: Optional["torch.device"] = None,
    seq_len: Optional[int] = None,
    **rope_kwargs,
) -> Tuple["torch.Tensor", float]:
    """
    manual patch:
    ``[patch:transformers.modeling_rope_utils._compute_dynamic_ntk_parameters]``

    Computes the inverse frequencies with NTK scaling.
    Credits to the Reddit users /u/bloc97 and /u/emozilla

    Args:
        config ([`~transformers.PretrainedConfig`]):
            The model configuration.
        device (`torch.device`):
            The device to use for initialization of the inverse frequencies.
        seq_len (`int`, *optional*):
            The current sequence length,
            used to update the dynamic RoPE at inference time.
        rope_kwargs (`Dict`, *optional*):
            BC compatibility with the previous
            RoPE class instantiation, will be removed in v4.45.

    Returns:
        Tuple of (`torch.Tensor`, `float`),
        containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the
        omputed cos/sin (unused in this type of RoPE).
    """
    if config is not None and len(rope_kwargs) > 0:
        raise ValueError(
            "Unexpected arguments: `**rope_kwargs` and `config` are mutually exclusive in "
            f"`_compute_dynamic_ntk_parameters`, got "
            f"`rope_kwargs`={rope_kwargs} and `config`={config}"
        )
    if len(rope_kwargs) > 0:
        base = rope_kwargs["base"]
        dim = rope_kwargs["dim"]
        max_position_embeddings = rope_kwargs["max_position_embeddings"]
        factor = rope_kwargs["factor"]
    elif config is not None:
        base = config.rope_theta
        partial_rotary_factor = (
            config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
        )
        head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        dim = int(head_dim * partial_rotary_factor)
        max_position_embeddings = config.max_position_embeddings
        factor = config.rope_scaling["factor"]

    attention_factor = 1.0  # Unused in this type of RoPE

    # seq_len: default to max_position_embeddings, e.g. at init time
    # seq_len = seq_len if seq_len is not None and
    #       seq_len > max_position_embeddings else max_position_embeddings
    if seq_len is None:
        seq_len = max_position_embeddings
    else:
        # PATCHED: remove the line using max
        torch._check(isinstance(seq_len, torch.Tensor))
        seq_len = torch.maximum(
            seq_len,
            torch.tensor(max_position_embeddings, dtype=seq_len.dtype, device=seq_len.device),
        )

    # Compute the inverse frequencies
    base = base * ((factor * seq_len / max_position_embeddings) - (factor - 1)) ** (
        dim / (dim - 2)
    )
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float)
            / dim
        )
    )
    return inv_freq, attention_factor


def patched_dynamic_rope_update(rope_forward):
    """manual patch: ``[patch:transformers.modeling_rope_utils.dynamic_rope_update]``

    ``rope_type`` is determined in the constructor of class
    :class:`transformers.models.phi3.modeling_phi3.Phi3RotaryEmbedding`.

    .. code-block:: python

        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get(
                "rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"

    The original code of the patched function:

    .. code-block:: python

        def dynamic_rope_update(rope_forward):
            def longrope_frequency_update(self, position_ids, device):
                seq_len = torch.max(position_ids) + 1
                if hasattr(self.config, "original_max_position_embeddings"):
                    original_max_position_embeddings =
                        self.config.original_max_position_embeddings
                else:
                    original_max_position_embeddings =
                        self.config.max_position_embeddings
                if seq_len > original_max_position_embeddings:
                    if not hasattr(self, "long_inv_freq"):
                        self.long_inv_freq, _ = self.rope_init_fn(
                            self.config, device, seq_len=original_max_position_embeddings + 1
                        )
                    self.register_buffer("inv_freq", self.long_inv_freq, persistent=False)
                else:
                    self.original_inv_freq = self.original_inv_freq.to(device)
                    self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)

            def dynamic_frequency_update(self, position_ids, device):
                seq_len = torch.max(position_ids) + 1
                if seq_len > self.max_seq_len_cached:  # growth
                    inv_freq, self.attention_scaling = self.rope_init_fn(
                        self.config, device, seq_len=seq_len)
                    self.register_buffer("inv_freq", inv_freq, persistent=False)
                    self.max_seq_len_cached = seq_len

                if seq_len < self.original_max_seq_len and
                        self.max_seq_len_cached > self.original_max_seq_len:
                    self.original_inv_freq = self.original_inv_freq.to(device)
                    self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
                    self.max_seq_len_cached = self.original_max_seq_len

            @wraps(rope_forward)
            def wrapper(self, x, position_ids):
                if "dynamic" in self.rope_type:
                    dynamic_frequency_update(self, position_ids, device=x.device)
                elif self.rope_type == "longrope":
                    longrope_frequency_update(self, position_ids, device=x.device)
                return rope_forward(self, x, position_ids)

            return wrapper

    """

    def longrope_frequency_update(self, position_ids, device):
        # It is no use to patch the function after the model is created
        # as rope_init_fn is an attribute set to one function when the model
        # is created and when no patch is applied yet.
        # So we select the patched version here.
        rope_init_fn = (
            patched__compute_dynamic_ntk_parameters
            if self.rope_init_fn
            is transformers.modeling_rope_utils._compute_dynamic_ntk_parameters
            else self.rope_init_fn
        )
        seq_len = torch.max(position_ids) + 1
        if hasattr(self.config, "original_max_position_embeddings"):
            original_max_position_embeddings = self.config.original_max_position_embeddings
        else:
            original_max_position_embeddings = self.config.max_position_embeddings
        # At export time, seq_len is unknown.
        long_inv_freq, _ = rope_init_fn(
            self.config, device, seq_len=original_max_position_embeddings + 1
        )
        original_inv_freq = self.original_inv_freq.to(device)

        # PATCHED: uses torch.cond instead of a test
        cond = (seq_len > original_max_position_embeddings).item()
        inv_freq = torch.cond(
            cond,
            (lambda x, y: x.clone()),
            (lambda x, y: y.clone()),
            [long_inv_freq, original_inv_freq],
        )
        self.inv_freq = inv_freq
        # if seq_len > original_max_position_embeddings:
        #    self.inv_freq = self.long_inv_freq
        # else:
        #    self.inv_freq = self.original_inv_freq

    def dynamic_frequency_update(self, position_ids, device):
        # constructor:
        # - self.max_seq_len_cached = config.max_position_embeddings
        # - self.original_max_seq_len = config.max_position_embeddings
        # - inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)

        # It is no use to patch the function after the model is created
        # as rope_init_fn is an attribute set to one function when the model
        # is created and when no patch is applied yet.
        # So we select the patched version here.
        rope_init_fn = (
            patched__compute_dynamic_ntk_parameters
            if self.rope_init_fn
            is transformers.modeling_rope_utils._compute_dynamic_ntk_parameters
            else self.rope_init_fn
        )

        # This behaviour is difficult to translate.
        # The sequence always grows.
        # The test should always True.
        # So:  self.max_seq_len_cached = max(self.max_seq_len_cached, seq_len) --> seq_len
        #
        # if seq_len > self.max_seq_len_cached:  # growth
        #    inv_freq, self.attention_scaling = self.rope_init_fn(
        #        self.config, device, seq_len=seq_len
        #    )
        #    self.register_buffer("inv_freq", inv_freq, persistent=False)
        #    self.max_seq_len_cached = seq_len
        #
        # So we should not need what follows.
        #
        # cond = (seq_len > self.max_seq_len_cached).item()
        # self.attention_scaling = torch.cond(
        #    cond,
        #    (lambda x, y: x.clone()),
        #    (lambda x, y: y.clone()),
        #    [attention_scaling, self.attention_scaling],
        # )

        seq_len = torch.max(position_ids) + 1
        long_inv_freq, self.attention_scaling = rope_init_fn(
            self.config, device, seq_len=seq_len
        )

        # Second test to translate.
        # Let's keep in mind, self.max_seq_len_cached = seq_len is likely to be True.
        # But in that case the following condition is a way to restore the original cache.

        # if (
        #    seq_len < self.original_max_seq_len
        #    and self.max_seq_len_cached > self.original_max_seq_len
        # ):
        #    self.original_inv_freq = self.original_inv_freq.to(device)
        #    self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
        #    self.max_seq_len_cached = self.original_max_seq_len

        original_inv_freq = self.original_inv_freq.to(device)
        cond = (seq_len >= self.original_max_seq_len).item()
        # PATCHED: uses torch.cond instead of a test
        inv_freq = torch.cond(
            cond,
            (lambda x, y: x.clone()),
            (lambda x, y: y.clone()),
            [long_inv_freq, original_inv_freq],
        )
        self.inv_freq = inv_freq

    @wraps(rope_forward)
    def wrapper(self, x, position_ids):
        if "dynamic" in self.rope_type:
            dynamic_frequency_update(self, position_ids, device=x.device)
        elif self.rope_type == "longrope":
            longrope_frequency_update(self, position_ids, device=x.device)
        return rope_forward(self, x, position_ids)

    return wrapper


def common_eager_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: Optional[float] = None,
    dropout: float = 0.0,
    head_mask: Optional[torch.Tensor] = None,
    **kwargs,
):
    if scaling is None:
        scaling = query.size(-1) ** -0.5

    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        # PATCHED
        # The two following lines were added.
        if attention_mask is not None and attention_mask.ndim == 4:
            attention_mask = attention_mask[:, :, :, : key.shape[-2]]
        attn_weights = attn_weights + attention_mask

    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)

    if head_mask is not None:
        attn_weights = attn_weights * head_mask.view(1, -1, 1, 1)

    attn_weights = torch.nn.functional.dropout(
        attn_weights, p=dropout, training=module.training
    )
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def patched_model_bart_eager_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: Optional[float] = None,
    dropout: float = 0.0,
    head_mask: Optional[torch.Tensor] = None,
    **kwargs,
):
    """[patch:transformers.models.bart.modeling_bart.eager_attention_forward]"""
    return common_eager_attention_forward(
        module,
        query,
        key,
        value,
        attention_mask=attention_mask,
        scaling=scaling,
        dropout=dropout,
        head_mask=head_mask,
        **kwargs,
    )


def patched_modeling_marian_eager_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: Optional[float] = None,
    dropout: float = 0.0,
    head_mask: Optional[torch.Tensor] = None,
    **kwargs,
):
    """[patch:transformers.models.marian.modeling_marian.eager_attention_forward]"""
    return common_eager_attention_forward(
        module,
        query,
        key,
        value,
        attention_mask=attention_mask,
        scaling=scaling,
        dropout=dropout,
        head_mask=head_mask,
        **kwargs,
    )


class common_RotaryEmbedding(torch.nn.Module):
    # This may cause some issues.
    # @torch.no_grad()
    # PATCHED: the decorator
    @patched_dynamic_rope_update
    def forward(self, x, position_ids):
        inv_freq_expanded = (
            self.inv_freq[None, :, None]
            .float()
            .expand(position_ids.shape[0], -1, 1)
            .to(x.device)
        )
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = (
            x.device.type
            if isinstance(x.device.type, str) and x.device.type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class patched_GemmaRotaryEmbedding(common_RotaryEmbedding):
    _PATCHES_ = ["forward"]
    _PATCHED_CLASS_ = transformers.models.gemma.modeling_gemma.GemmaRotaryEmbedding


if pv.Version(transformers.__version__) >= pv.Version("4.52"):

    class patched_Gemma2RotaryEmbedding(common_RotaryEmbedding):
        _PATCHES_ = ["forward"]
        _PATCHED_CLASS_ = transformers.models.gemma2.modeling_gemma2.Gemma2RotaryEmbedding

    class patched_Gemma3RotaryEmbedding(common_RotaryEmbedding):
        _PATCHES_ = ["forward"]
        _PATCHED_CLASS_ = transformers.models.gemma3.modeling_gemma3.Gemma3RotaryEmbedding


class patched_LlamaRotaryEmbedding(common_RotaryEmbedding):
    _PATCHES_ = ["forward"]
    _PATCHED_CLASS_ = transformers.models.llama.modeling_llama.LlamaRotaryEmbedding


class patched_MistralRotaryEmbedding(common_RotaryEmbedding):
    _PATCHES_ = ["forward"]
    _PATCHED_CLASS_ = transformers.models.mistral.modeling_mistral.MistralRotaryEmbedding


class patched_MixtralRotaryEmbedding(common_RotaryEmbedding):
    _PATCHES_ = ["forward"]
    _PATCHED_CLASS_ = transformers.models.mixtral.modeling_mixtral.MixtralRotaryEmbedding


class patched_PhiRotaryEmbedding(common_RotaryEmbedding):
    _PATCHES_ = ["forward"]
    _PATCHED_CLASS_ = transformers.models.phi.modeling_phi.PhiRotaryEmbedding


if pv.Version(transformers.__version__) >= pv.Version("4.51"):

    class patched_Phi3RotaryEmbedding(common_RotaryEmbedding):
        _PATCHES_ = ["forward"]
        _PATCHED_CLASS_ = transformers.models.phi3.modeling_phi3.Phi3RotaryEmbedding


if pv.Version(transformers.__version__) >= pv.Version("4.52"):

    class patched_Phi4MultimodalRotaryEmbedding(common_RotaryEmbedding):
        _PATCHES_ = ["forward"]
        _PATCHED_CLASS_ = (
            transformers.models.phi4_multimodal.modeling_phi4_multimodal.Phi4MultimodalRotaryEmbedding
        )


if pv.Version(transformers.__version__) >= pv.Version("4.53"):

    class patched_SmolLM3RotaryEmbedding(common_RotaryEmbedding):
        _PATCHES_ = ["forward"]
        _PATCHED_CLASS_ = transformers.models.smollm3.modeling_smollm3.SmolLM3RotaryEmbedding


class patched_IdeficsEmbedding(torch.nn.Module):
    _PATCHES_ = ["forward"]
    _PATCHED_CLASS_ = transformers.models.idefics.modeling_idefics.IdeficsEmbedding

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # if seq_len > self.max_seq_len_cached:
        #    self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        def _set_cos_sin_cache_then(x, inv_freq, seq_len, _cos_cached, _sin_cached):
            t = torch.arange(seq_len, device=x.device, dtype=torch.int64).type_as(inv_freq)
            freqs = torch.einsum("i,j->ij", t, inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            return emb.cos().to(x.dtype), emb.sin().to(x.dtype)

        def _set_cos_sin_cache_else(_x, _inv_freq, _seq_len, cos_cached, sin_cached):
            torch._check(seq_len.item() <= cos_cached.shape[0])
            co = cos_cached[: seq_len.item()].detach().clone()
            torch._check(seq_len.item() <= sin_cached.shape[0])
            si = sin_cached[: seq_len.item()].detach().clone()
            return co.to(dtype=x.dtype), si.to(dtype=x.dtype)

        cos_cached, sin_cached = torch.cond(
            (seq_len > self.max_seq_len_cached).item(),
            _set_cos_sin_cache_then,
            _set_cos_sin_cache_else,
            [x, self.inv_freq, seq_len, self.cos_cached, self.sin_cached],
        )
        return cos_cached, sin_cached


class patched_IdeficsAttention(torch.nn.Module):
    _PATCHES_ = ["forward"]
    _PATCHED_CLASS_ = transformers.models.idefics.modeling_idefics.IdeficsAttention

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # if key_value_states are provided this layer is used as a cross-attention layer
        is_cross_attention = self.is_cross_attention or key_value_states is not None

        bsz, q_len, _ = hidden_states.size()

        query_states = (
            self.q_proj(hidden_states)
            .view(bsz, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        if not is_cross_attention:
            key_states = (
                self.k_proj(hidden_states)
                .view(bsz, q_len, self.num_heads, self.head_dim)
                .transpose(1, 2)
            )
            value_states = (
                self.v_proj(hidden_states)
                .view(bsz, q_len, self.num_heads, self.head_dim)
                .transpose(1, 2)
            )
        else:
            _, kv_len, _ = (
                key_value_states.size()
            )  # Note that, in this case, `kv_len` == `kv_seq_len`
            key_states = (
                self.k_proj(key_value_states)
                .view(bsz, kv_len, self.num_heads, self.head_dim)
                .transpose(1, 2)
            )
            value_states = (
                self.v_proj(key_value_states)
                .view(bsz, kv_len, self.num_heads, self.head_dim)
                .transpose(1, 2)
            )

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += cache_position[0]

        if not is_cross_attention:
            rotary_length = torch.maximum(
                torch.tensor(kv_seq_len, dtype=torch.int64),
                torch.tensor(q_len, dtype=torch.int64),
            )
            cos, sin = self.rotary_emb(value_states, seq_len=rotary_length)
            query_states, key_states = (
                transformers.models.idefics.modeling_idefics.apply_rotary_pos_emb(
                    query_states, key_states, cos, sin, position_ids
                )
            )
        # [bsz, nh, t, hd]

        if past_key_value is not None:
            # sin and cos are specific to RoPE models;
            # cache_position needed for the static cache
            cache_kwargs = {"cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        if self.qk_layer_norms:
            query_states = self.q_layer_norm(query_states)
            key_states = self.k_layer_norm(key_states)

        attention_interface: Callable = (
            transformers.models.idefics.modeling_idefics.eager_attention_forward
        )

        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and output_attentions:
                transformers.models.idefics.modeling_idefics.logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support "
                    "`output_attentions=True`. Falling back to "
                    "eager attention. This warning can be removed using the argument "
                    '`attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = transformers.modeling_utils.ALL_ATTENTION_FUNCTIONS[
                    self.config._attn_implementation
                ]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        if output_attentions:
            attn_weights = None

        if pv.Version(transformers.__version__) < pv.Version("4.53.99"):
            return attn_output, attn_weights, past_key_value
        return attn_output, attn_weights


class patched_SamMaskDecoder(torch.nn.Module):
    _PATCHES_ = ["forward"]
    _PATCHED_CLASS_ = transformers.models.sam.modeling_sam.SamMaskDecoder

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_positional_embeddings: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        output_attentions: Optional[bool] = None,
        attention_similarity: Optional[torch.Tensor] = None,
        target_embedding: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Args:
            image_embeddings (`torch.Tensor`):
                the embeddings from the image encoder
            image_positional_embedding (`torch.Tensor`):
                positional encoding with the shape of image_embeddings
            sparse_prompt_embeddings (`torch.Tensor`):
                The embeddings of the points and boxes
            dense_prompt_embeddings (`torch.Tensor`):
                the embeddings of the mask inputs
            multimask_output (bool):
                Whether to return multiple masks or a single mask.
            output_attentions (bool, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
        """
        batch_size, num_channels, height, width = image_embeddings.shape
        point_batch_size = sparse_prompt_embeddings.shape[1]
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.repeat(batch_size, point_batch_size, 1, 1)

        # torch.cond rewrites the if-else logic to handle empty sparse_prompt_embeddings
        # torch.any is needed to avoid data-dependent control flow
        # with sparse_prompt_embeddings.sum().item() != 0
        def sparse_prompt_embeddings_is_not_empty(output_tokens, sparse_prompt_embeddings):
            return torch.cat((output_tokens, sparse_prompt_embeddings), dim=2)

        def sparse_prompt_embeddings_is_empty(output_tokens, sparse_prompt_embeddings):
            return output_tokens.clone()

        tokens = torch.cond(
            torch.any(sparse_prompt_embeddings != 0),
            sparse_prompt_embeddings_is_not_empty,
            sparse_prompt_embeddings_is_empty,
            [output_tokens, sparse_prompt_embeddings],
        )

        point_embeddings = tokens.to(self.iou_token.weight.dtype)

        # Expand per-image data in batch direction to be per-point
        image_embeddings = image_embeddings + dense_prompt_embeddings
        image_embeddings = image_embeddings.repeat_interleave(point_batch_size, 0)
        image_positional_embeddings = image_positional_embeddings.repeat_interleave(
            point_batch_size, 0
        )

        # Run the transformer, image_positional_embedding are consumed
        torch._check(point_embeddings.shape[0] != 0)
        torch._check(point_embeddings.shape[1] != 0)
        torch._check(point_embeddings.shape[2] != 0)
        torch._check(point_embeddings.shape[3] != 0)
        embeddings_attentions = self.transformer(
            point_embeddings=point_embeddings,
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
            attention_similarity=attention_similarity,
            target_embedding=target_embedding,
            output_attentions=output_attentions,
        )
        point_embedding, image_embeddings = embeddings_attentions[:2]
        iou_token_out = torch.select(point_embedding, dim=2, index=0)
        mask_tokens_out = torch.narrow(
            point_embedding, dim=2, start=1, length=self.num_mask_tokens
        )

        # Upscale mask embeddings and predict masks using the mask tokens
        image_embeddings = image_embeddings.transpose(2, 3).reshape(
            batch_size * point_batch_size, num_channels, height, width
        )

        upscaled_embedding = self.upscale_conv1(image_embeddings)
        upscaled_embedding = self.activation(self.upscale_layer_norm(upscaled_embedding))
        upscaled_embedding = self.activation(self.upscale_conv2(upscaled_embedding))

        hyper_in_list = []
        for i in range(self.num_mask_tokens):
            current_mlp = self.output_hypernetworks_mlps[i]
            hyper_in_list += [current_mlp(mask_tokens_out[:, :, i, :])]
        hyper_in = torch.stack(hyper_in_list, dim=2)

        _, num_channels, height, width = upscaled_embedding.shape
        upscaled_embedding = upscaled_embedding.reshape(
            batch_size, point_batch_size, num_channels, height * width
        )
        masks = (hyper_in @ upscaled_embedding).reshape(
            batch_size, point_batch_size, -1, height, width
        )

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, :, mask_slice, :, :]
        iou_pred = iou_pred[:, :, mask_slice]

        outputs = (masks, iou_pred)

        if len(embeddings_attentions) == 2:
            # transformers==4.54
            return outputs

        if output_attentions and len(embeddings_attentions) > 2:
            outputs = outputs + (embeddings_attentions[2],)  # noqa: RUF005
        else:
            outputs = outputs + (None,)  # noqa: RUF005
        return outputs


def rewrite_loop_for_square_mask(mask: torch.Tensor, seq: torch.Tensor):
    """
    Rewrites the loop in:

    .. code-block:: python

        attention_mask = torch.full(
            [1, seq_length, seq_length], torch.finfo(q.dtype).min, dtype=q.dtype
        )
        for i in range(1, len(seq)):
            attention_mask[..., seq[i - 1] : seq[i], seq[i - 1] : seq[i]] = 0
    """
    r = torch.arange(0, mask.shape[-1], dtype=torch.int64)
    less0 = (r.reshape((-1, 1)) < seq.reshape((1, -1))).to(torch.int64)
    less = less0.sum(axis=-1, keepdim=True) + 1
    sq = less * less.T
    look = (
        torch.max(seq.min() == 0, less != less.max())
        * torch.max(seq.max() == mask.shape[-1], less != less.min())
        * less
    )
    filt = (sq != look**2).to(mask.dtype)
    return mask * filt


class patched_VisionAttention(torch.nn.Module):
    _PATCHES_ = ["forward"]
    _PATCHED_CLASS_ = transformers.models.qwen2_vl.modeling_qwen2_vl.VisionAttention

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = (
            self.qkv(hidden_states)
            .reshape(seq_length, 3, self.num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        if position_embeddings is None:
            transformers.models.qwen2_vl.modeling_qwen2_vl.logger.warning_once(
                "The attention layers in this model are transitioning from "
                " computing the RoPE embeddings internally "
                "through `rotary_pos_emb` (2D tensor of RoPE theta values), "
                "to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin)."
                " In v4.54 `rotary_pos_emb` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        else:
            cos, sin = position_embeddings
        q, k = transformers.models.qwen2_vl.modeling_qwen2_vl.apply_rotary_pos_emb_vision(
            q, k, cos, sin
        )

        attention_mask = torch.full(
            [1, seq_length, seq_length],
            torch.finfo(q.dtype).min,
            device=q.device,
            dtype=q.dtype,
        )
        # for i in range(1, len(cu_seqlens)):
        #     attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i],
        #                         cu_seqlens[i - 1] : cu_seqlens[i]] = 0
        attention_mask = rewrite_loop_for_square_mask(attention_mask, cu_seqlens)

        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        attn_weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.head_dim)
        attn_weights = attn_weights + attention_mask
        attn_weights = torch.nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(0, 1)
        attn_output = attn_output.reshape(seq_length, -1)
        attn_output = self.proj(attn_output)
        return attn_output


try:
    import transformers.models.qwen3_moe

    patch_qwen3 = True
except ImportError:
    patch_qwen3 = False

if patch_qwen3:

    class patched_Qwen3MoeSparseMoeBlock(torch.nn.Module):
        _PATCHES_ = ["forward", "_forward_expert_loop"]
        _PATCHED_CLASS_ = (
            transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeSparseMoeBlock
        )

        def _forward_expert_loop(
            self,
            final_hidden_states,
            expert_mask_idx,
            hidden_states,
            routing_weights,
            expert_idx: int,
        ):
            # idx, top_x = torch.where(expert_mask_idx.squeeze(0))
            idx, top_x = torch.nonzero(expert_mask_idx, as_tuple=True)
            hidden_dim = hidden_states.shape[-1]
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            expert_current_state = self.experts[expert_idx](current_state)
            current_hidden_states = expert_current_state * routing_weights[top_x, idx, None]
            return final_hidden_states.index_add(
                0, top_x, current_hidden_states.to(hidden_states.dtype)
            )

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            """ """
            batch_size, sequence_length, hidden_dim = hidden_states.shape
            hidden_states = hidden_states.view(-1, hidden_dim)
            # router_logits: (batch * sequence_length, n_experts)
            router_logits = self.gate(hidden_states)

            routing_weights = torch.nn.functional.softmax(
                router_logits, dim=1, dtype=torch.float
            )
            routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
            if self.norm_topk_prob:  # only diff with mixtral sparse moe block!
                routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
            # we cast back to the input dtype
            routing_weights = routing_weights.to(hidden_states.dtype)

            final_hidden_states = torch.zeros(
                (batch_size * sequence_length, hidden_dim),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )

            # One hot encode the selected experts to create an expert mask
            # this will be used to easily index which expert is going to be sollicitated
            expert_mask = torch.nn.functional.one_hot(
                selected_experts, num_classes=self.num_experts
            ).permute(2, 1, 0)

            # Loop over all available experts in the model
            # and perform the computation on each expert
            expert_sum = expert_mask.sum(dim=(-1, -2))
            # expert_hit = torch.greater(expert_sum, 0).nonzero()
            # for expert_idx in expert_hit:
            for expert_idx in range(self.num_experts):
                # initial code has a squeeze but it is not possible to do that.
                # expert_mask_idx = expert_mask[expert_idx].squeeze(0)
                expert_mask_idx = expert_mask[expert_idx]
                final_hidden_states = torch.cond(
                    (expert_sum[expert_idx] > 0).item(),
                    lambda final_hidden_states, expert_mask, hidden_states, routing_weights, _i=expert_idx: self._forward_expert_loop(  # noqa: E501
                        final_hidden_states,
                        expert_mask,
                        hidden_states,
                        routing_weights,
                        expert_idx=_i,
                    ),
                    lambda final_hidden_states, *args: final_hidden_states.clone(),
                    [final_hidden_states, expert_mask_idx, hidden_states, routing_weights],
                )

                # if expert_sum[expert_idx] > 0:
                #    idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))

                # Index the correct hidden states and compute the expert hidden state for
                # the current expert. We need to make sure to multiply the output hidden
                # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
                #    current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
                #    current_hidden_states = (
                #        expert_layer(current_state) * routing_weights[top_x, idx, None]
                #    )

                # However `index_add_` only support torch tensors for indexing so we'll use
                # the `top_x` tensor here.
                #    final_hidden_states.index_add_(
                #        0, top_x, current_hidden_states.to(hidden_states.dtype)
                #    )

            final_hidden_states = final_hidden_states.reshape(
                batch_size, sequence_length, hidden_dim
            )
            return final_hidden_states, router_logits


try:
    from transformers.models.gemma3.modeling_gemma3 import Gemma3Model  # noqa: F401

    patch_gemma3 = True
except ImportError:
    patch_gemma3 = False


if patch_gemma3:

    class patched_Gemma3Model(torch.nn.Module):
        _PATCHES_ = ["get_placeholder_mask"]
        _PATCHED_CLASS_ = transformers.models.gemma3.modeling_gemma3.Gemma3Model
        _PATCHED_PR_ = "https://github.com/huggingface/transformers/pull/41319"

        def get_placeholder_mask(
            self,
            input_ids: torch.LongTensor,
            inputs_embeds: torch.FloatTensor,
            image_features: torch.FloatTensor,
        ):
            if input_ids is None:
                special_image_mask = inputs_embeds == self.get_input_embeddings()(
                    torch.tensor(
                        self.config.image_token_id,
                        dtype=torch.long,
                        device=inputs_embeds.device,
                    )
                )
                special_image_mask = special_image_mask.all(-1)
            else:
                special_image_mask = input_ids == self.config.image_token_id

            n_image_tokens = special_image_mask.sum()
            special_image_mask = (
                special_image_mask.unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            n_image_features = image_features.shape[0] * image_features.shape[1]
            # PATCHED: torch._check
            # if inputs_embeds[special_image_mask].numel() != image_features.numel():
            #    raise ValueError( ... )
            torch._check(
                inputs_embeds[special_image_mask].numel() == image_features.numel(),
                lambda: (
                    f"Image features and image tokens do not match: tokens: "
                    f"{n_image_tokens}, features {n_image_features}"
                ),
            )
            return special_image_mask
