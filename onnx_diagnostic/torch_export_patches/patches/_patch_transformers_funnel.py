import torch

try:
    import transformers.models.funnel.modeling_funnel

    patch_funnel = True
except ImportError:
    patch_funnel = False

if patch_funnel:
    from transformers.models.funnel.modeling_funnel import _relative_shift_gather

    class patched_FunnelAttentionStructure(torch.nn.Module):
        _PATCHES_ = ["relative_pos"]
        _PATCHED_CLASS_ = transformers.models.funnel.modeling_funnel.FunnelAttentionStructure

        def relative_pos(
            self, pos: torch.Tensor, stride: int, pooled_pos=None, shift: int = 1
        ) -> torch.Tensor:
            if pooled_pos is None:
                pooled_pos = pos
            ref_point = pooled_pos[0] - pos[0]
            # PATCHED
            num_remove = shift * pooled_pos.shape[0]
            max_dist = ref_point + num_remove * stride
            min_dist = pooled_pos[0] - pos[-1]
            return torch.arange(
                max_dist.to(torch.long),
                (min_dist - 1).to(torch.long),
                torch.tensor(-stride, dtype=torch.long),
                dtype=torch.long,
                device=pos.device,
            )

    class patched_FunnelRelMultiheadAttention(torch.nn.Module):
        _PATCHES_ = ["relative_positional_attention"]
        _PATCHED_CLASS_ = (
            transformers.models.funnel.modeling_funnel.FunnelRelMultiheadAttention
        )

        def relative_positional_attention(
            self, position_embeds, q_head, context_len, cls_mask=None
        ):
            """Relative attention score for the positional encodings"""
            # q_head has shape batch_size x sea_len x n_head x d_head
            if self.config.attention_type == "factorized":
                phi, pi, psi, omega = position_embeds
                # Shape n_head x d_head
                u = self.r_r_bias * self.scale
                # Shape d_model x n_head x d_head
                w_r = self.r_kernel

                # Shape batch_size x sea_len x n_head x d_model
                q_r_attention = torch.einsum("binh,dnh->bind", q_head + u, w_r)
                q_r_attention_1 = q_r_attention * phi[:, None]
                q_r_attention_2 = q_r_attention * pi[:, None]

                # Shape batch_size x n_head x seq_len x context_len
                positional_attn = torch.einsum(
                    "bind,jd->bnij", q_r_attention_1, psi
                ) + torch.einsum("bind,jd->bnij", q_r_attention_2, omega)
            else:
                shift = 2 if q_head.shape[1] != context_len else 1
                r = position_embeds[self.block_index][shift - 1]
                # Shape n_head x d_head
                v = self.r_r_bias * self.scale
                # Shape d_model x n_head x d_head
                w_r = self.r_kernel

                # Shape max_rel_len x n_head x d_model
                r_head = torch.einsum("td,dnh->tnh", r, w_r)
                # Shape batch_size x n_head x seq_len x max_rel_len
                positional_attn = torch.einsum("binh,tnh->bnit", q_head + v, r_head)
                # Shape batch_size x n_head x seq_len x context_len
                positional_attn = _relative_shift_gather(positional_attn, context_len, shift)

            if cls_mask is not None:
                # PATCHED
                positional_attn = positional_attn * cls_mask
            return positional_attn
