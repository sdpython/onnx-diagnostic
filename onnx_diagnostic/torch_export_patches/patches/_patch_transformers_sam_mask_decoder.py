from typing import Optional
import torch
import transformers


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
