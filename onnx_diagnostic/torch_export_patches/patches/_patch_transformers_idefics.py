from typing import Callable, Optional, Tuple
import packaging.version as pv
import torch
import transformers


class patched_IdeficsEmbedding(torch.nn.Module):
    _PATCHES_ = ["forward"]
    _PATCHED_CLASS_ = transformers.models.idefics.modeling_idefics.IdeficsEmbedding

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # if seq_len > self.max_seq_len_cached:
        #    self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        def _set_cos_sin_cache_then(x, inv_freq, seq_len, _cos_cached, _sin_cached):
            t = torch.arange(seq_len, device=x.device, dtype=torch.int64).type_as(inv_freq)
            # freqs = torch.einsum("i,j->ij", t, inv_freq)
            freqs = t.reshape((-1, 1)) * inv_freq.reshape((1, -1))
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
