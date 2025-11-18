import os
from typing import Callable, Optional
import torch
import torch.nn.functional as F
from .patch_helper import _is_torchdynamo_exporting
from ._patch_transformers_attention import patched_sdpa_attention_forward

try:
    import transformers.models.qwen2_5_vl
    import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl

    patch_qwen2_5 = True
except ImportError:
    patch_qwen2_5 = False

strategy_for_attention_in_qwen_2_5 = os.environ.get("QWEN25ATTENTION", "BIGMASK")

if patch_qwen2_5:

    class patched_Qwen2_5_VLForConditionalGeneration:
        _PATCHES_ = ["prepare_inputs_for_generation"]
        _PATCHED_CLASS_ = (
            transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration
        )

        def prepare_inputs_for_generation(
            self,
            input_ids,
            past_key_values=None,
            attention_mask=None,
            inputs_embeds=None,
            cache_position=None,
            position_ids=None,
            use_cache=True,
            pixel_values=None,
            pixel_values_videos=None,
            image_grid_thw=None,
            video_grid_thw=None,
            second_per_grid_ts=None,
            **kwargs,
        ):
            # Overwritten -- in specific circumstances we don't want to f
            # forward image inputs to the model
            from transformers.generation import GenerationMixin

            model_inputs = GenerationMixin.prepare_inputs_for_generation(
                self,
                input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
                position_ids=position_ids,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                use_cache=use_cache,
                **kwargs,
            )

            # Qwen2-5-VL position_ids are prepared with rope_deltas
            if position_ids is None:
                # Calculate RoPE index once per generation in the pre-fill stage only.
                # When compiling, we can't check tensor values thus we check only input length
                # It is safe to assume that `length!=1` means we're in pre-fill
                # because compiled models currently cannot do assisted decoding
                if cache_position[0] == 0 or self.model.rope_deltas is None:
                    vision_positions, rope_deltas = self.model.get_rope_index(
                        model_inputs.get("input_ids", None),
                        image_grid_thw=image_grid_thw,
                        video_grid_thw=video_grid_thw,
                        second_per_grid_ts=second_per_grid_ts,
                        attention_mask=attention_mask,
                    )
                    self.model.rope_deltas = rope_deltas
                # then use the prev pre-calculated rope-deltas to get the correct position ids
                elif (
                    "position_ids" in model_inputs and model_inputs["position_ids"] is not None
                ):
                    batch_size, seq_length = model_inputs["position_ids"].shape
                    device = model_inputs["position_ids"].device
                    position_ids = torch.arange(seq_length, device=device)
                    position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
                    delta = cache_position[0] + self.model.rope_deltas
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                    vision_positions = position_ids + delta.expand_as(position_ids)

                # Concatenate "text + vision" positions into [4, bs, seq-len]
                if "position_ids" not in model_inputs or model_inputs["position_ids"] is None:
                    text_positions = torch.arange(input_ids.shape[1], device=input_ids.device)[
                        None, None, :
                    ]
                else:
                    text_positions = model_inputs["position_ids"][None, ...]
                # text_positions = model_inputs["position_ids"][None, ...]
                assert vision_positions is not None, "vision_positions are missing"
                model_inputs["position_ids"] = torch.cat(
                    [text_positions, vision_positions], dim=0
                )

            if cache_position[0] != 0:
                model_inputs["pixel_values"] = None
                model_inputs["pixel_values_videos"] = None

            return model_inputs

    class patched_Qwen2_5_VisionTransformerPretrainedModel:
        _PATCHES_ = ["get_window_index", "forward", "rot_pos_emb"]
        _PATCHED_CLASS_ = (
            transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VisionTransformerPretrainedModel
        )

        def rot_pos_emb(self, grid_thw):
            pos_ids = []
            for thw_ in grid_thw:
                # PATCHED: avoid unbind
                t = thw_[0]
                h = thw_[1]
                w = thw_[2]
                hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
                hpos_ids = hpos_ids.reshape(
                    h // self.spatial_merge_size,
                    self.spatial_merge_size,
                    w // self.spatial_merge_size,
                    self.spatial_merge_size,
                )
                hpos_ids = hpos_ids.permute(0, 2, 1, 3)
                hpos_ids = hpos_ids.flatten()

                wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
                wpos_ids = wpos_ids.reshape(
                    h // self.spatial_merge_size,
                    self.spatial_merge_size,
                    w // self.spatial_merge_size,
                    self.spatial_merge_size,
                )
                wpos_ids = wpos_ids.permute(0, 2, 1, 3)
                wpos_ids = wpos_ids.flatten()
                pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
            pos_ids = torch.cat(pos_ids, dim=0)
            max_grid_size = grid_thw[:, 1:].max()
            rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
            rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
            return rotary_pos_emb

        def get_window_index(self, grid_thw):
            window_index: list = []  # type: ignore[annotation-unchecked]
            # PATCHED
            cu_window_seqlens: list = [torch.tensor([0], dtype=torch.int64)]  # type: ignore[annotation-unchecked]
            window_index_id = 0
            vit_merger_window_size = (
                self.window_size // self.spatial_merge_size // self.patch_size
            )

            for _thw in grid_thw:
                # PATCHED: avoid unbind
                grid_t = _thw[0]
                grid_h = _thw[1]
                grid_w = _thw[2]
                llm_grid_h, llm_grid_w = (
                    grid_h // self.spatial_merge_size,
                    grid_w // self.spatial_merge_size,
                )
                index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(
                    grid_t, llm_grid_h, llm_grid_w
                )
                pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
                pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
                num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
                num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
                index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
                index_padded = index_padded.reshape(
                    grid_t,
                    num_windows_h,
                    vit_merger_window_size,
                    num_windows_w,
                    vit_merger_window_size,
                )
                index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
                    grid_t,
                    num_windows_h * num_windows_w,
                    vit_merger_window_size,
                    vit_merger_window_size,
                )
                seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
                index_padded = index_padded.reshape(-1)
                index_new = index_padded[index_padded != -100]
                window_index.append(index_new + window_index_id)
                cu_seqlens_tmp = (
                    seqlens.cumsum(0) * self.spatial_merge_unit + cu_window_seqlens[-1][-1:]
                )
                # PATCHED
                # cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
                cu_window_seqlens.append(cu_seqlens_tmp)
                window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
            window_index = torch.cat(window_index, dim=0)

            return window_index, torch.cat(cu_window_seqlens, dim=0)

        def forward(
            self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs
        ) -> torch.Tensor:
            """
            Args:
                hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
                    The final hidden states of the model.
                grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                    The temporal, height and width of feature shape of each image in LLM.

            Returns:
                `torch.Tensor`: hidden_states.
            """
            hidden_states = self.patch_embed(hidden_states)
            rotary_pos_emb = self.rot_pos_emb(grid_thw)
            window_index, cu_window_seqlens = self.get_window_index(grid_thw)
            # PATCHED
            # cu_window_seqlens = torch.tensor(
            #    cu_window_seqlens,
            #    device=hidden_states.device,
            #    dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
            # )
            cu_window_seqlens = cu_window_seqlens.to(hidden_states.device).to(grid_thw.dtype)
            cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

            seq_len, _ = hidden_states.size()
            hidden_states = hidden_states.reshape(
                seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1
            )
            hidden_states = hidden_states[window_index, :, :]
            hidden_states = hidden_states.reshape(seq_len, -1)
            rotary_pos_emb = rotary_pos_emb.reshape(
                seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1
            )
            rotary_pos_emb = rotary_pos_emb[window_index, :, :]
            rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            position_embeddings = (emb.cos(), emb.sin())

            cu_seqlens = torch.repeat_interleave(
                grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
            ).cumsum(
                dim=0,
                # Select dtype based on the following factors:
                #  - FA2 requires that cu_seqlens_q must have dtype int32
                #  - torch.onnx.export requires that cu_seqlens_q must have same dtype
                # as grid_thw
                # See https://github.com/huggingface/transformers/pull/34852
                # for more information
                dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
            )
            cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

            for layer_num, blk in enumerate(self.blocks):
                if layer_num in self.fullatt_block_indexes:
                    cu_seqlens_now = cu_seqlens
                else:
                    cu_seqlens_now = cu_window_seqlens

                hidden_states = blk(
                    hidden_states,
                    cu_seqlens=cu_seqlens_now,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )

            hidden_states = self.merger(hidden_states)
            reverse_indices = torch.argsort(window_index)
            hidden_states = hidden_states[reverse_indices, :]
            return hidden_states

    class patched_Qwen2_5_VLVisionAttentionOneIteration(torch.nn.Module):
        def forward(
            self,
            start_end,
            query_states,
            key_states,
            value_states,
            scaling: float = 1.0,
            dropout: float = 0.0,
            **kwargs,
        ):
            a = start_end[0].item()
            b = start_end[1].item()
            q = query_states[:, :, a:b, :]
            k = key_states[:, :, a:b, :]
            v = value_states[:, :, a:b, :]
            return patched_sdpa_attention_forward(
                self,
                q,
                k,
                v,
                attention_mask=None,
                scaling=scaling,
                dropout=dropout,
                is_causal=False,
                **kwargs,
            )[0]

    class patched_Qwen2_5_VLVisionAttention:
        _PATCHES_ = ["forward"]
        _PATCHED_CLASS_ = (
            transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLVisionAttention
        )

        def forward(
            self,
            hidden_states: torch.Tensor,
            cu_seqlens: torch.Tensor,
            rotary_pos_emb: Optional[torch.Tensor] = None,
            position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
            **kwargs,
        ) -> torch.Tensor:
            seq_length = hidden_states.shape[0]
            # PATCHED: avoid the use of unbind
            qkv = (
                self.qkv(hidden_states)
                .reshape(seq_length, 3, self.num_heads, -1)
                .permute(1, 0, 2, 3)
            )

            query_states, key_states, value_states = qkv[0], qkv[1], qkv[2]
            cos, sin = position_embeddings

            # This part should be moved into the loop
            # iteration to enable fusion inside the loop.
            query_states, key_states = (
                transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.apply_rotary_pos_emb_vision(
                    query_states, key_states, cos, sin
                )
            )

            query_states = query_states.transpose(0, 1).unsqueeze(0)
            key_states = key_states.transpose(0, 1).unsqueeze(0)
            value_states = value_states.transpose(0, 1).unsqueeze(0)

            attention_interface: Callable = (
                transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.eager_attention_forward
            )
            if self.config._attn_implementation != "eager":
                # PATCHED
                # attention_interface = ALL_ATTENTION_FUNCTIONS[
                #       self.config._attn_implementation]
                attention_interface = transformers.modeling_utils.ALL_ATTENTION_FUNCTIONS[
                    self.config._attn_implementation
                ]

            if _is_torchdynamo_exporting():
                if self.config._attn_implementation == "flash_attention_2":
                    max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
                    attn_output = torch.onnx.ops.symbolic(
                        "custom::qwen25_flash_attention",
                        (
                            query_states,
                            key_states,
                            value_states,
                            cu_seqlens,
                            cu_seqlens,
                            max_seqlen,
                            max_seqlen,
                            torch.tensor(self.scaling, dtype=torch.float32),
                        ),
                        dtype=query_states.dtype,
                        shape=(
                            query_states.shape[0],  # batch_size
                            query_states.shape[2],  # sequence_length (total patches)
                            query_states.shape[1],  # num_heads
                            query_states.shape[3],  # head_size
                        ),
                        version=1,
                    )
                elif (
                    attention_interface
                    is transformers.integrations.sdpa_attention.sdpa_attention_forward
                    and strategy_for_attention_in_qwen_2_5 == "LOOPMHA"
                ):

                    def _iteration(start_end, query_states, key_states, value_states):
                        return patched_Qwen2_5_VLVisionAttentionOneIteration.forward(
                            self,
                            start_end,
                            query_states,
                            key_states,
                            value_states,
                            scaling=self.scaling,
                            dropout=0.0 if not self.training else self.attention_dropout,
                        )

                    starts = cu_seqlens[:-1]
                    ends = cu_seqlens[1:]
                    # cu_seqlens = [0, 10, 14, 27]
                    # starts: [0, 10, 14]
                    # ends: [10, 14, 17]
                    # starts_ends: [[0, 10], [10, 14], [14, 27]]
                    starts_ends = torch.cat([starts.unsqueeze(1), ends.unsqueeze(1)], dim=1)
                    attn_outputs = [
                        _iteration(start_end, query_states, key_states, value_states)
                        for start_end in starts_ends
                    ]
                    # attn_outputs = torch._higher_order_ops.while_loop(
                    # attn_outputs = torch.ops.higher_order.while_loop(
                    #    (lambda it, starts_ends, *_args: it < starts_ends.shape[0]),
                    #    _iteration,
                    #    (torch.tensor(0),
                    #       starts_ends, query_states, key_states, value_states), tuple(),
                    # )
                    attn_output = torch.cat(attn_outputs, dim=1)
                elif (
                    attention_interface
                    is transformers.integrations.sdpa_attention.sdpa_attention_forward
                    and strategy_for_attention_in_qwen_2_5 == "BIGMASK"
                ):
                    # make square mask
                    indices = torch.arange(
                        cu_seqlens.max(), dtype=cu_seqlens.dtype, device=cu_seqlens.device
                    )
                    dot = (cu_seqlens.unsqueeze(1) <= indices.unsqueeze(0)).to(
                        cu_seqlens.dtype
                    )
                    dot = dot.sum(dim=0)
                    mask = dot.unsqueeze(1) - dot.unsqueeze(0)
                    bool_mask = mask == 0
                    bool_mask = bool_mask.unsqueeze(0).unsqueeze(0)

                    torch._check(bool_mask.shape[2] == key_states.shape[2])
                    torch._check(bool_mask.shape[3] == key_states.shape[2])

                    attn_output, _ = attention_interface(
                        self,
                        query_states,
                        key_states,
                        value_states,
                        attention_mask=bool_mask,
                        scaling=self.scaling,
                        dropout=0.0 if not self.training else self.attention_dropout,
                        is_causal=False,
                        **kwargs,
                    )
                elif (
                    attention_interface
                    is transformers.integrations.sdpa_attention.sdpa_attention_forward
                    and strategy_for_attention_in_qwen_2_5 == "PACKED"
                ):
                    max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
                    attn_output = torch.onnx.ops.symbolic(
                        "custom::qwen25_packed_attention",
                        (
                            query_states,
                            key_states,
                            value_states,
                            cu_seqlens,
                            cu_seqlens,
                            max_seqlen,
                            max_seqlen,
                            torch.tensor(self.scaling, dtype=torch.float32),
                        ),
                        dtype=query_states.dtype,
                        shape=(
                            query_states.shape[0],  # batch_size
                            query_states.shape[2],  # sequence_length (total patches)
                            query_states.shape[1],  # num_heads
                            query_states.shape[3],  # head_size
                        ),
                        version=1,
                    )
                else:
                    raise NotImplementedError(
                        f"Not export strategy for strategy_for_attention_in_qwen_2_5="
                        f"{strategy_for_attention_in_qwen_2_5!r}, "
                        f"(use QWEN25ATTENTION to change it), and attention_interface="
                        f"{attention_interface!r} (use sdpa)"
                    )
            elif self.config._attn_implementation == "flash_attention_2":
                # Flash Attention 2: Use cu_seqlens for variable length attention
                max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
                attn_output, _ = attention_interface(
                    self,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask=None,
                    scaling=self.scaling,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    cu_seq_lens_q=cu_seqlens,
                    cu_seq_lens_k=cu_seqlens,
                    max_length_q=max_seqlen,
                    max_length_k=max_seqlen,
                    is_causal=False,
                    **kwargs,
                )
            else:
                # Other implementations: Process each chunk separately
                lengths = cu_seqlens[1:] - cu_seqlens[:-1]
                splits = [
                    torch.split(tensor, lengths.tolist(), dim=2)
                    for tensor in (query_states, key_states, value_states)
                ]

                attn_outputs = [
                    attention_interface(
                        self,
                        q,
                        k,
                        v,
                        attention_mask=None,
                        scaling=self.scaling,
                        dropout=0.0 if not self.training else self.attention_dropout,
                        is_causal=False,
                        **kwargs,
                    )[0]
                    for q, k, v in zip(*splits)
                ]
                attn_output = torch.cat(attn_outputs, dim=1)

            attn_output = attn_output.reshape(seq_length, -1).contiguous()
            attn_output = self.proj(attn_output)
            return attn_output
