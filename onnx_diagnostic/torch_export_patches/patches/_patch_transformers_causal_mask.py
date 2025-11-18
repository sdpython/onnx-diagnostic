from dataclasses import dataclass
from typing import Optional
import torch
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from .patch_helper import _has_transformers


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
    _PATCHES_ = ["_make_causal_mask"] if not _has_transformers("4.48.3") else []
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
