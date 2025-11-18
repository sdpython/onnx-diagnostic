import torch
import transformers

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
