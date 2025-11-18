from typing import Optional, Tuple
import torch

try:
    import transformers.models.qwen2_vl

    patch_qwen2 = True
except ImportError:
    patch_qwen2 = False


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


if patch_qwen2:

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
            attn_weights = torch.matmul(q, k.transpose(1, 2)) / (self.head_dim**0.5)
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            ).to(q.dtype)
            attn_output = torch.matmul(attn_weights, v)
            attn_output = attn_output.transpose(0, 1)
            attn_output = attn_output.reshape(seq_length, -1)
            attn_output = self.proj(attn_output)
            return attn_output
