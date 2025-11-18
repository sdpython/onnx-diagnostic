import torch

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
