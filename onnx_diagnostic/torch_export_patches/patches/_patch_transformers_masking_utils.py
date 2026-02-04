import inspect
from typing import Callable, List, Optional, Tuple
import torch

try:
    import transformers.masking_utils  # noqa: F401

    patch_masking_utils = True
except ImportError:
    patch_masking_utils = False


if patch_masking_utils:
    # Introduced in 4.52
    from transformers.masking_utils import (
        _ignore_causal_mask_sdpa,
        and_masks,
        causal_mask_function,
        padding_mask_function,
        prepare_padding_mask,
    )

    _prepare_padding_mask_kwargs = (
        dict(_slice=False)
        if "_slice" in inspect.signature(prepare_padding_mask).parameters
        else {}
    )

    try:
        # transformers>=5.0
        from transformers.masking_utils import (
            _ignore_bidirectional_mask_sdpa,
            bidirectional_mask_function,
        )
    except ImportError:
        _ignore_bidirectional_mask_sdpa = None
        bidirectional_mask_function = None

    try:
        from transformers.masking_utils import _non_vmap_expansion_sdpa
    except ImportError:

        def _non_vmap_expansion_sdpa(
            batch_indices: torch.Tensor,
            head_indices: torch.Tensor,
            q_indices: torch.Tensor,
            kv_indices: torch.Tensor,
        ):
            """
            https://github.com/huggingface/optimum-onnx/blob/
            c123e8f4fab61b54a8e0e31ce74462bcacca576e/optimum/exporters/onnx/model_patcher.py#L362-L365
            """
            batch_indices = batch_indices[:, None, None, None]
            head_indices = head_indices[None, :, None, None]
            q_indices = q_indices[None, None, :, None]
            kv_indices = kv_indices[None, None, None, :]
            return batch_indices, head_indices, q_indices, kv_indices

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
            # if _is_torchdynamo_exporting():
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
        _ = kwargs.pop("allow_is_bidirectional_skip", None)
        # PATCHED: this line called the patched version of sdpa_mask
        mask = patched_sdpa_mask_recent_torch(
            batch_size=batch_size,
            cache_position=cache_position,
            kv_length=kv_length,
            kv_offset=kv_offset,
            mask_function=mask_function,
            attention_mask=attention_mask,
            allow_is_causal_skip=False,
            allow_is_bidirectional_skip=False,
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
        allow_is_bidirectional_skip: bool = False,
        **kwargs,
    ) -> Optional[torch.Tensor]:
        """manual patch for function ``transformers.masking_utils.sdpa_mask_recent_torch``."""
        q_length = cache_position.shape[0]
        padding_mask = prepare_padding_mask(
            attention_mask, kv_length, kv_offset, **_prepare_padding_mask_kwargs
        )
        if allow_is_causal_skip and _ignore_causal_mask_sdpa(
            padding_mask, q_length, kv_length, kv_offset, local_size
        ):
            return None
        if (
            allow_is_bidirectional_skip
            and _ignore_bidirectional_mask_sdpa
            and _ignore_bidirectional_mask_sdpa(padding_mask, kv_length, kv_offset)
        ):
            return None

        if mask_function is bidirectional_mask_function:
            if padding_mask is not None:
                # used for slicing without data-dependent slicing
                mask_indices = (
                    torch.arange(kv_length, device=cache_position.device) + kv_offset
                )
                return padding_mask[:, None, None, mask_indices].expand(-1, -1, q_length, -1)
            return torch.ones(
                batch_size,
                1,
                q_length,
                kv_length,
                dtype=torch.bool,
                device=cache_position.device,
            )

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

    def patched_sdpa_mask(
        batch_size: int,
        cache_position: torch.Tensor,
        kv_length: int,
        kv_offset: int = 0,
        mask_function: Callable = causal_mask_function,
        attention_mask: torch.Tensor | None = None,
        local_size: int | None = None,
        allow_is_causal_skip: bool = True,
        allow_is_bidirectional_skip: bool = False,
        allow_torch_fix: bool = True,
        use_vmap: bool = False,
        **kwargs,
    ) -> torch.Tensor | None:
        """manual patch for function ``transformers.masking_utils.sdpa_mask``."""
        q_length = cache_position.shape[0]

        # Potentially pad the 2D mask
        padding_mask = prepare_padding_mask(attention_mask, kv_length, kv_offset)

        # Under specific conditions, we can avoid materializing the mask
        #   1. Causal masks can rely on the `is_causal` argument
        #   2. Bidirectional do not need any further processing (no bias)
        if allow_is_causal_skip and _ignore_causal_mask_sdpa(
            padding_mask, q_length, kv_length, kv_offset, local_size
        ):
            return None
        if allow_is_bidirectional_skip and _ignore_bidirectional_mask_sdpa(
            padding_mask, kv_length, local_size
        ):
            return None

        # Potentially add the padding 2D mask
        if padding_mask is not None:
            mask_function = and_masks(mask_function, padding_mask_function(padding_mask))

        batch_arange = torch.arange(batch_size, device=cache_position.device)
        head_arange = torch.arange(1, device=cache_position.device)
        # Similar to `kv_arange = torch.arange(start=kv_offset,
        # end=kv_offset + kv_length, device=cache_position.device)`
        # but without data-dependent slicing (i.e. torch.compile friendly)
        kv_arange = torch.arange(kv_length, device=cache_position.device) + kv_offset

        # Actual mask creation
        # Option 1: Fast non-vmap mask creation (default)
        # PATCHED
        use_vmap = False
        if not use_vmap:
            # Apply mask function element-wise through broadcasting
            attention_mask = mask_function(
                *_non_vmap_expansion_sdpa(batch_arange, head_arange, cache_position, kv_arange)
            )
            # Expand the mask to match batch size
            # and query length if they weren't used in the mask function
            attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)

        # Option 2: Vmap mask creation (torch>=2.6 and custom patterns)
        # elif _is_torch_greater_or_equal_than_2_6:
        # This creates the 4D mask easily.
        # Note that we need this context manager as vmap cannot handle slicing a tensor from
        # scalar tensor (it internally calls `.item()` which vmap does not allow,
        # but this context works around it
        # We don't need to add an offset to the mask_function either,
        # as we vmap directly the correct indices for k and kv indices
        #    with TransformGetItemToIndex():
        #        attention_mask = _vmap_expansion_sdpa(mask_function)(
        #            batch_arange, head_arange, cache_position, kv_arange
        #        )

        # Option 3: Error out since it indicates that the user did something custom,
        # which they shouldn't have (torch<2.6)
        else:
            raise ValueError(
                "The vmap functionality for mask creation "
                "is only supported from torch>=2.6. "
                "Please update your torch version or use "
                "`use_vmap=False` with index-based masks."
            )
        return attention_mask
