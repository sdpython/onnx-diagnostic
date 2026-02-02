r"""
Export visual and embedding parts of microsoft/Phi-4-multimodal-instruct
========================================================================

Requirements
++++++++++++

::

    git+https://github.com/sdpython/experimental-experiment.git  # optional
    backoff
    huggingface_hub
    onnx-diagnostic>=0.8.6
    onnxruntime>=1.23
    peft==0.17.1
    Pillow
    requests
    torch>=2.10  # weekly is better
    tqdm
    transformers==4.48.3

.. note::

    ``flash_attn`` must be removed to export if it was installed.

Examples
++++++++

.. code-block:: bash

    python -m onnx_diagnostic.ci_models.export_phi4_mm \
        -m microsoft/Phi-4-multimodal-instruct --device cuda --dtype float16 \
        --exporter custom --pretrained --second-input --part vision
"""

import os
import pprint
import sys
import textwrap
import time
from typing import Dict, List, Optional, Tuple, Union

from .ci_helpers import (
    check_for_discrepancies_and_log_everything_into_a_json_file,
    compute_expected_outputs,
    get_parser,
    get_torch_dtype_from_command_line_args,
    simplify_model_id_for_a_filename,
    zip_model_and_data_into_a_single_file,
)


def get_patches_transformers():
    import re
    from itertools import cycle
    import torch
    import transformers

    class patched_PreTrainedModel(torch.nn.Module):
        _PATCHES_ = ["get_expanded_tied_weights_keys"]
        _PATCHED_CLASS_ = transformers.modeling_utils.PreTrainedModel

        def get_expanded_tied_weights_keys(self, all_submodels: bool = False) -> dict:
            if all_submodels:
                expanded_tied_weights = {}
                for prefix, submodule in self.named_modules(remove_duplicate=False):
                    if isinstance(submodule, transformers.modeling_utils.PreTrainedModel):
                        submodel_tied_weights = submodule.get_expanded_tied_weights_keys(
                            all_submodels=False
                        )
                        if prefix != "":
                            submodel_tied_weights = {
                                f"{prefix}.{k}": f"{prefix}.{v}"
                                for k, v in submodel_tied_weights.items()
                            }
                        expanded_tied_weights.update(submodel_tied_weights)
                return expanded_tied_weights

            tied_mapping = self._tied_weights_keys
            if not self.config.tie_word_embeddings and not self.config.tie_encoder_decoder:
                return {}
            elif tied_mapping is None:
                return {}
            common_case_regex = re.compile(r"^[A-Za-z0-9_\.]+(weight)|(bias)$")
            # PATCHED
            if tied_mapping == ["lm_head.weight"]:
                tied_mapping = {"lm_head.weight": "model.embed_tokens.weight"}
            if all(
                common_case_regex.match(k) for k in tied_mapping.keys() | tied_mapping.values()
            ):
                return tied_mapping.copy()

            expanded_tied_weights = {}
            all_param_names = {k for k, _ in self.named_parameters(remove_duplicate=False)} | {
                k for k, _ in self.named_buffers(remove_duplicate=False)
            }
            for target_name, source_name in tied_mapping.items():
                target_name = "^" + target_name
                source_name = "^" + source_name

                source_params = sorted(
                    filter(lambda x: re.search(source_name, x), all_param_names)
                )
                target_params = sorted(
                    filter(lambda x: re.search(target_name, x), all_param_names)
                )
                if (
                    not len(source_params) > 0
                    or not len(target_params) > 0
                    or len(target_params) % len(source_params) != 0
                ):
                    raise ValueError(
                        f"There is an issue with your definition of "
                        f"`tie_weights_keys` for {source_name}:{target_name}. "
                        f"We found {source_params} to tie into {target_params}"
                    )
                for target_n, source_n in zip(target_params, cycle(source_params)):
                    if source_n in expanded_tied_weights.keys():
                        expanded_tied_weights[target_n] = expanded_tied_weights[source_n]
                    else:
                        expanded_tied_weights[target_n] = source_n

            return expanded_tied_weights

    return [patched_PreTrainedModel]


def get_patches(mod, mod_siglip):
    import torch
    from transformers.modeling_outputs import BaseModelOutputWithPooling
    from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
    from ..export.cf_simple_loop_for import simple_loop_for

    _IMAGE_SPECIAL_TOKEN_ID = mod._IMAGE_SPECIAL_TOKEN_ID

    class patched_SiglipVisionEmbeddings(torch.nn.Module):
        _PATCHES_ = ["forward"]
        _PATCHED_CLASS_ = mod_siglip.SiglipVisionEmbeddings

        def forward(
            self, pixel_values: torch.FloatTensor, patch_attention_mask: torch.BoolTensor
        ) -> torch.Tensor:
            batch_size = pixel_values.size(0)

            patch_embeds = self.patch_embedding(pixel_values)
            embeddings = patch_embeds.flatten(2).transpose(1, 2)

            max_im_h, max_im_w = pixel_values.size(2), pixel_values.size(3)
            max_nb_patches_h, max_nb_patches_w = (
                max_im_h // self.patch_size,
                max_im_w // self.patch_size,
            )
            boundaries = torch.arange(
                torch.tensor(1 / self.num_patches_per_side, dtype=pixel_values.dtype),
                torch.tensor(1.0, dtype=pixel_values.dtype),
                torch.tensor(1 / self.num_patches_per_side, dtype=pixel_values.dtype),
            )
            position_ids = torch.full(
                size=(batch_size, max_nb_patches_h * max_nb_patches_w), fill_value=0
            )

            # PATHED: a loop replace with scan.

            def body(p_attn_mask, position_ids_row, boundaries):
                h_len = torch.tensor(1, dtype=boundaries.dtype) / p_attn_mask[:, 0].sum()
                w_len = torch.tensor(1, dtype=boundaries.dtype) / p_attn_mask[0].sum()
                torch._check(h_len.item() > 0)
                fractional_coords_h = torch.arange(
                    torch.tensor(0.0, dtype=boundaries.dtype),
                    torch.tensor(1 - 1e-6, dtype=boundaries.dtype),
                    h_len,
                )
                torch._check(w_len.item() > 0)
                fractional_coords_w = torch.arange(
                    torch.tensor(0.0, dtype=boundaries.dtype),
                    torch.tensor(1 - 1e-6, dtype=boundaries.dtype),
                    w_len,
                )

                bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
                bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)

                pos_ids = (
                    bucket_coords_h[:, None] * self.num_patches_per_side + bucket_coords_w
                ).flatten()

                row = position_ids_row.clone()
                row[p_attn_mask.view(-1)] = pos_ids
                return [row]

            position_ids = torch.ops.higher_order.scan(
                body, [], [patch_attention_mask, position_ids], additional_inputs=[boundaries]
            )[0]

            position_ids = position_ids.to(self.position_embedding.weight.device)
            embeddings = embeddings + self.position_embedding(position_ids)
            return embeddings

    class patched_SiglipVisionTransformer(torch.nn.Module):
        _PATCHES_ = ["forward"]
        _PATCHED_CLASS_ = mod_siglip.SiglipVisionTransformer

        def forward(
            self,
            pixel_values,
            patch_attention_mask: Optional[torch.BoolTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
        ) -> Union[Tuple, BaseModelOutputWithPooling]:
            output_attentions = (
                output_attentions
                if output_attentions is not None
                else self.config.output_attentions
            )
            output_hidden_states = (
                output_hidden_states
                if output_hidden_states is not None
                else self.config.output_hidden_states
            )
            return_dict = (
                return_dict if return_dict is not None else self.config.use_return_dict
            )

            batch_size = pixel_values.size(0)
            if patch_attention_mask is None:
                patch_attention_mask = torch.ones(
                    size=(
                        batch_size,
                        pixel_values.size(2) // self.config.patch_size,
                        pixel_values.size(3) // self.config.patch_size,
                    ),
                    dtype=torch.bool,
                    device=pixel_values.device,
                )

            hidden_states = self.embeddings(
                pixel_values=pixel_values, patch_attention_mask=patch_attention_mask
            )

            patch_attention_mask = patch_attention_mask.view(batch_size, -1)
            # PATCHED: skip the test
            # if not torch.any(~patch_attention_mask):
            #    attention_mask = None
            # else:
            #    attention_mask = (
            #        _prepare_4d_attention_mask(patch_attention_mask, hidden_states.dtype)
            ##        if not self.config._flash_attn_2_enabled
            #        else patch_attention_mask
            #    )
            attention_mask = _prepare_4d_attention_mask(
                patch_attention_mask, hidden_states.dtype
            )

            encoder_outputs = self.encoder(
                inputs_embeds=hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            last_hidden_state = encoder_outputs[0]
            last_hidden_state = self.post_layernorm(last_hidden_state)

            pooled_output = self.head(
                hidden_state=last_hidden_state,
                attention_mask=patch_attention_mask,
            )

            if not return_dict:
                return (last_hidden_state, pooled_output, *encoder_outputs[1:])

            return BaseModelOutputWithPooling(
                last_hidden_state=last_hidden_state,
                pooler_output=pooled_output,
                hidden_states=encoder_outputs.hidden_states,
                attentions=encoder_outputs.attentions,
            )

    class patched_Phi4MMImageEmbedding(torch.nn.Module):
        _PATCHES_ = ["forward"]
        _PATCHED_CLASS_ = mod.Phi4MMImageEmbedding

        def forward(
            self,
            input_ids: torch.LongTensor,
            input_embeds: torch.FloatTensor,
            image_sizes=None,
            **kwargs,
        ) -> torch.FloatTensor:

            if isinstance(input_ids, tuple):
                input_ids, input_embeds = input_ids

            img_embeds = input_embeds
            if image_sizes is None and "image_sizes" in kwargs:
                image_sizes = kwargs["image_sizes"]
            img_sizes = image_sizes

            if self.img_features is not None:
                img_embeds = self.img_features.clone()
                self.img_features = None

            if self.img_sizes is not None:
                img_sizes = self.img_sizes

            dtype = self.img_processor.embeddings.patch_embedding.weight.dtype
            if img_embeds is not None:
                img_embeds = img_embeds.to(dtype)

            if self.image_attention_mask is not None:
                image_attention_mask = self.image_attention_mask.clone()
                self.image_attention_mask = None
            elif "image_attention_mask" in kwargs:
                image_attention_mask = kwargs["image_attention_mask"]
            else:
                image_attention_mask = None
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])

            with torch.no_grad():
                positions_tuple = torch.nonzero(
                    input_ids == _IMAGE_SPECIAL_TOKEN_ID, as_tuple=True
                )

            select = False
            hd_transform = False

            if isinstance(self.img_projection, torch.nn.Sequential):
                target_device = self.img_projection[0].bias.device
            else:
                target_device = self.img_projection.bias.device

            # PATCHED: Let's assume it is always true.
            if True:  # len(positions.tolist()) > 0:
                if self.use_hd_transform and img_sizes is not None:
                    hd_transform = True
                    bs = img_embeds.shape[0]
                    if image_attention_mask is not None:
                        img_features = self.get_img_features(
                            img_embeds.flatten(0, 1),
                            attention_mask=image_attention_mask.type(torch.BoolTensor)
                            .flatten(0, 1)
                            .to(target_device),
                        )
                    else:
                        img_features = self.get_img_features(img_embeds.flatten(0, 1))

                    base_resolution = self.crop_size
                    base_feat_height_reduction = self.base_feat_height_reduction

                    base_feat_height = base_feat_width = torch.sym_int(
                        img_features.shape[1] ** 0.5
                    )
                    img_features = img_features.view(
                        bs, -1, base_feat_height * base_feat_width, self.image_dim_out
                    )
                    C = self.image_dim_out
                    H = base_feat_height

                    if isinstance(img_sizes, torch.Tensor):
                        img_sizes = img_sizes.view(-1, 2)
                else:
                    raise NotImplementedError
                select = True

            hidden_states = kwargs["wte"](input_ids)

            assert select
            if hd_transform:

                def body_fn(
                    _bs,
                    img_features,
                    img_sizes,
                    image_attention_mask,
                    cst_shape_CH,
                    glb_GN,
                    sub_GN,
                    proj_0_weight,
                    proj_0_bias,
                    proj_1_weight,
                    proj_1_bias,
                    base_resolution=None,
                    base_feat_height_reduction=None,
                    base_feat_height=None,
                    base_feat_width=None,
                ):
                    # oddly, it seems impossible to write img_sizes[_bs.item()]
                    # it needs img_sizes[_bs.item() : (_bs + 1).item()][0]
                    row = img_sizes[_bs.item() : (_bs + 1).item()]
                    row = row[0]
                    h, w = row[0], row[1]
                    h = h // base_resolution
                    w = w // base_resolution
                    B_ = h * w
                    C, H = cst_shape_CH.shape

                    # 1 x (24x24) x 1024
                    global_img_feature = img_features[_bs.item() : (_bs + 1).item(), :1][0]

                    # 1 x 12 x 12 x 4096
                    glb_img = (
                        global_img_feature.reshape(1, H, H, C)
                        .reshape(
                            1,
                            H // base_feat_height_reduction,
                            base_feat_height_reduction,
                            H // base_feat_height_reduction,
                            base_feat_height_reduction,
                            C,
                        )
                        .contiguous()
                        .permute(0, 1, 3, 2, 4, 5)
                        .reshape(
                            1,
                            H // base_feat_height_reduction,
                            H // base_feat_height_reduction,
                            base_feat_height_reduction * base_feat_height_reduction * C,
                        )
                        .contiguous()
                    )
                    temp_glb_GN = sub_GN.repeat(1, H // base_feat_height_reduction, 1, 1)

                    # 1 x 156 x 4096
                    glb_img = torch.cat([glb_img, temp_glb_GN], dim=2).reshape(
                        1, -1, base_feat_height_reduction * base_feat_height_reduction * C
                    )

                    # (max_num_crops-1) x (12x12) x C
                    sub_img = img_features[_bs.item() : (_bs + 1).item(), 1:][0]
                    # 16x574x1024
                    # get rid of padding sub_img
                    sub_img = sub_img[: B_.item()]

                    # (num_crops, 12, 2, 12, 2, 1024) -> (num_crops, 12, 12, 2, 2, 1024)
                    # -> (num_crops, 12*12, 4*1024)
                    sub_img = (
                        sub_img.reshape(B_.item(), H, H, C)
                        .reshape(
                            B_.item(),
                            H // base_feat_height_reduction,
                            base_feat_height_reduction,
                            H // base_feat_height_reduction,
                            base_feat_height_reduction,
                            C,
                        )
                        .contiguous()
                        .permute(0, 1, 3, 2, 4, 5)
                        .reshape(
                            B_.item(),
                            -1,
                            base_feat_height_reduction * base_feat_height_reduction * C,
                        )
                        .contiguous()
                    )
                    sub_img = (
                        sub_img.reshape(
                            1,
                            h.item(),
                            w.item(),
                            base_feat_height // base_feat_height_reduction,
                            base_feat_width // base_feat_height_reduction,
                            -1,
                        )
                        .permute(0, 1, 3, 2, 4, 5)
                        .reshape(
                            1,
                            (h * base_feat_height // base_feat_height_reduction).item(),
                            (w * base_feat_width // base_feat_height_reduction).item(),
                            base_feat_height_reduction * base_feat_height_reduction * C,
                        )
                    )

                    reshaped_image_attention_mask = (
                        image_attention_mask[
                            _bs.item() : (_bs + 1).item(), 1 : (B_ + 1).item(), 0::2, 0::2
                        ][0]
                        .reshape(
                            1,
                            h.item(),
                            w.item(),
                            base_feat_height // base_feat_height_reduction,
                            base_feat_width // base_feat_height_reduction,
                        )
                        .permute(0, 1, 3, 2, 4)
                        .reshape(
                            1,
                            (h * base_feat_height // base_feat_height_reduction).item(),
                            (w * base_feat_width // base_feat_height_reduction).item(),
                        )
                    )
                    useful_height = (
                        reshaped_image_attention_mask[0, :, 0].sum().to(torch.int64).item()
                    )
                    useful_width = (
                        reshaped_image_attention_mask[0, 0, :].sum().to(torch.int64).item()
                    )
                    # the module cannot be extracted from here
                    sub_img = sub_img[:, :useful_height, :useful_width]
                    temp_sub_GN = sub_GN.repeat(1, useful_height, 1, 1)
                    # temp_len = (
                    #    image_attention_mask[_bs, : B_ + 1, 0::2, 0::2]
                    #    .sum()
                    #    .to(torch.int64)
                    #    .item()
                    #    + (useful_height + 1)
                    #    + base_feat_height // base_feat_height_reduction
                    # )

                    sub_img = torch.cat([sub_img, temp_sub_GN], dim=2).reshape(
                        1, -1, base_feat_height_reduction * base_feat_height_reduction * C
                    )
                    # (1, num_img_tokens, 1024*4)

                    # glb + sub
                    # glb_sub
                    # output_imgs.append(torch.cat([glb_img, self.glb_GN, sub_img], dim=1))
                    # sub_glb
                    _output_img = torch.cat([sub_img, glb_GN, glb_img], dim=1)
                    # output_len.append(temp_len)
                    proj = torch.nn.functional.linear(_output_img, proj_0_weight, proj_0_bias)
                    proj = torch.nn.functional.gelu(proj)
                    proj = torch.nn.functional.linear(proj, proj_1_weight, proj_1_bias)
                    return (proj,)

                def local_body_fn(
                    n_iter,
                    img_features,
                    img_sizes,
                    image_attention_mask,
                    cst_shape_CH,
                    glb_GN,
                    sub_GN,
                    proj_0_weight,
                    proj_0_bias,
                    proj_1_weight,
                    proj_1_bias,
                ):
                    return body_fn(
                        n_iter,
                        img_features,
                        img_sizes,
                        image_attention_mask,
                        cst_shape_CH,
                        glb_GN,
                        sub_GN,
                        proj_0_weight,
                        proj_0_bias,
                        proj_1_weight,
                        proj_1_bias,
                        base_resolution=base_resolution,
                        base_feat_height_reduction=base_feat_height_reduction,
                        base_feat_height=base_feat_height,
                        base_feat_width=base_feat_width,
                    )

                tmp = torch.arange(bs + 1).max()
                glb_GN = self.glb_GN
                sub_GN = self.sub_GN
                cst_shape_CH = torch.zeros((C, H), dtype=torch.int32)

                merged_img_set_tensor = simple_loop_for(
                    tmp,
                    local_body_fn,
                    (
                        img_features,
                        img_sizes,
                        image_attention_mask,
                        cst_shape_CH,
                        glb_GN,
                        sub_GN,
                        self.img_projection[0].weight,
                        self.img_projection[0].bias,
                        # self.img_projection[1] is GELU
                        self.img_projection[2].weight,
                        self.img_projection[2].bias,
                    ),
                    [1],
                )
                torch._check(isinstance(merged_img_set_tensor, torch.Tensor))
                merged_img_set_tensor = merged_img_set_tensor.squeeze(0)

                # merged_img_set_tensor = torch.cat(img_set_tensor, dim=1).squeeze(0)
                merged_img_set_tensor = merged_img_set_tensor.to(hidden_states.dtype).to(
                    hidden_states.device
                )
                with torch.autocast(device_type=hidden_states.device.type, enabled=False):
                    new_hidden_states = hidden_states.index_put(
                        indices=positions_tuple,
                        values=merged_img_set_tensor,
                        accumulate=False,
                    )
                hidden_states = new_hidden_states
            else:
                raise NotImplementedError

            if self.drop is not None:
                hidden_states = self.drop(hidden_states)

            return hidden_states

    return [
        *get_patches_transformers(),
        patched_Phi4MMImageEmbedding,
        patched_SiglipVisionEmbeddings,
        patched_SiglipVisionTransformer,
    ]


def get_inputs_for_part(
    model_id: str,
    part: str,
    torch_dtype: "torch.dtype",  # noqa: F821
    device: str,
    second_input: bool,
) -> Tuple[Dict[str, "torch.Tensor"], List[Dict[str, "torch.Tensor"]]]:  # noqa: F821
    if part == "vision":
        import requests
        from PIL import Image
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        user_prompt = "<|user|>\n"
        assistant_prompt = "<|assistant|>\n"
        prompt_suffix = "<|end|>\n"
        prompt = (
            f"{user_prompt}<|image_1|>\n<|image_2|>\n"
            f"What is shown in these four images?{prompt_suffix}{assistant_prompt}"
        )

        root = os.path.join(os.path.dirname(__file__), "..", "..", "_small_data")
        # "https://www.ilankelman.org/stopsigns/australia.jpg"
        url = os.path.join(root, "American_Flamingo_JG.jpg")
        image_1 = (
            Image.open(requests.get(url, stream=True).raw)
            if url.startswith("https")
            else Image.open(url)
        )
        # "https://wallpaper.dog/large/10809054.jpg"
        url = os.path.join(root, "RedcrestedTuraco.jpg")
        image_4 = (
            Image.open(requests.get(url, stream=True).raw)
            if url.startswith("https")
            else Image.open(url)
        )

        images = [image_1, image_4]
        inputs = processor(prompt, images=images, return_tensors="pt").to(device)
        export_inputs = dict(
            input_ids=inputs["input_ids"].to(device),
            input_image_embeds=inputs["input_image_embeds"].to(torch_dtype).to(device),
            image_attention_mask=inputs["image_attention_mask"].to(torch_dtype).to(device),
            image_sizes=inputs["image_sizes"].to(device),
        )
        assert (
            export_inputs["input_image_embeds"].shape[-2] >= 28
            and export_inputs["input_image_embeds"].shape[-1] >= 28
        ), (
            f"required by the exported program but shape is "
            f"{export_inputs['input_image_embeds'].shape}"
        )

        other_inputs = []
        if second_input:
            prompt = (
                f"{user_prompt}<|image_1|>\n<|image_2|>\n<|image_3|>\n<|image_4|>\n"
                f"What is shown in these four images?{prompt_suffix}{assistant_prompt}"
            )
            image_2_path = os.path.join(
                os.path.dirname(__file__), "data", "Blanca_Lake_Hudak.jpg"
            )
            image_2 = Image.open(image_2_path)
            url = (
                "https://th.bing.com/th/id/OIP.gCvQ1vmPVJmrq1nnzM3ZHQHaEo?rs=1&pid=ImgDetMain"
            )
            image_3_path = os.path.join(
                os.path.dirname(__file__), "data", "Ice_worm_glacier.jpg"
            )
            image_3 = Image.open(image_3_path)

            images = [image_1, image_2, image_3, image_4]
            inputs = processor(prompt, images=images, return_tensors="pt").to(device)
            other_inputs = [
                dict(
                    input_ids=inputs["input_ids"].to(device),
                    input_image_embeds=inputs["input_image_embeds"].to(torch_dtype).to(device),
                    image_attention_mask=inputs["image_attention_mask"]
                    .to(torch_dtype)
                    .to(device),
                    image_sizes=inputs["image_sizes"].to(device),
                )
            ]
        return export_inputs, other_inputs

    raise NotImplementedError(f"No inputs yet implement for part={part!r}")


def main(
    model_id: str = "microsoft/Phi-4-multimodal-instruct",
    device: str = "cpu",
    dtype: str = "float32",
    exporter: str = "onnx-dynamo",
    pretrained: bool = True,
    second_input: bool = True,
    make_zip: bool = False,
    output_folder: str = "dump_models",
    existing_onnx: str | None = None,
    part: str = "vision",
    atol: float = 2,
    mismatch01: float = 0.01,
    profile_exporter: bool = False,
):
    """
    Exports model Qwen/Qwen2.5-VL-7B-Instruct or pieces of it.
    The script applies as well to other models based on the same architecture.

    The function saves everything on disk. It does not generate new inputs
    on the second run but reuses the saved ones. Same goes for the expected
    outputs with are also saved on disk.

    :param model_id: model id
    :param device: device
    :param dtype: dtype
    :param exporter: exportor to use
    :param pretrained: pretrained=False is usually used to test
    :param second_input: checks discrepancies on more examples
    :param make_zip: creates a zip at the end
    :param output_folder: output folder
    :param part: "" to export the whole model, ``"vision"`` for vision part,
        ...
    :param atol: raises an exception if tolerance is above that threshold
    :param mismatch01: raises an exception if the ratio of mismatches
        is above that threshold
    :param profile_exporter: profiles the exporter
    """
    prefix = simplify_model_id_for_a_filename(model_id)
    basename = os.path.join(
        output_folder, f"model.{prefix}.{part}.{device}.{dtype}.{exporter}"
    )
    filename = f"{basename}.onnx"
    stat_file = f"{basename}.stats"

    print("------------------------------------------------------------------")
    print(f"-- model_id={model_id}")
    print(f"-- part={part}")
    print(f"-- device={device}")
    print(f"-- dtype={dtype}")
    print(f"-- exporter={exporter}")
    print(f"-- pretrained={pretrained}")
    print(f"-- second_input={second_input}")
    print(f"-- make_zip={make_zip}")
    print(f"-- output_folder={output_folder}")
    print(f"-- atol={atol}")
    print(f"-- mismatch01={mismatch01}")
    print(f"-- profile_exporter={profile_exporter}")
    print("------------------------------------------------------------------")
    print(f"-- prefix={prefix}")
    print(f"-- export in {filename!r}")
    print("------------------------------------------------------------------")

    if os.path.exists(stat_file) and not existing_onnx:
        print(f"-- skipping because {stat_file!r} already exists")
        return

    print("-- import torch and others")
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    from ..helpers import string_type, string_diff, max_diff
    from ..torch_export_patches import torch_export_patches
    from ..torch_export_patches.patch_details import PatchDetails
    from ..torch_export_patches.patch_inputs import use_dyn_not_str
    from ..export.api import to_onnx

    if output_folder and output_folder != ".":
        os.makedirs(output_folder, exist_ok=True)

    print(f"-- create model {model_id!r}")
    print(
        f"-- device={device!r}, dtype={dtype!r}, exporter={exporter!r}, "
        f"pretrained={pretrained!r}"
    )
    torch_dtype = get_torch_dtype_from_command_line_args(dtype)

    if pretrained:
        print("-- pretrained model")
        config = AutoConfig.from_pretrained(
            model_id, trust_remote_code=True, attn_implementation="sdpa"
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map=device,
            attn_implementation="sdpa",
        ).eval()
        data = dict(model=model)
    else:
        print("-- random model")
        config = AutoConfig.from_pretrained(
            model_id, trust_remote_code=True, attn_implementation="sdpa"
        )
        config.attn_implementation = "sdpa"
        config._attn_implementation = "sdpa"
        config.num_hidden_layers = 2
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        data = dict(model=model)

    main_mod_name = model.__module__
    assert (
        main_mod_name in sys.modules
    ), f"Unable to find {main_mod_name!r} in {pprint.pformat(list(sys.modules))}"
    main_mod = sys.modules[main_mod_name]
    model = model.to(device).to(getattr(torch, dtype))
    mod_siglip_name = model.model.embed_tokens_extend.image_embed.img_processor.__module__
    assert (
        mod_siglip_name in sys.modules
    ), f"Unable to find {mod_siglip_name!r} in {pprint.pformat(list(sys.modules))}"
    mod_siglip = sys.modules[mod_siglip_name]

    print(f"-- config._attn_implementation={model.config._attn_implementation}")
    print(f"-- model.dtype={model.dtype}")
    print(f"-- model.device={model.device}")

    export_inputs, other_inputs = None, None
    if not part:
        # used to unit test
        from ..helpers.torch_helper import to_any

        assert "inputs" in data, f"key 'inputs' is missing from data (available {set(data)})"
        model_to_export = data["model"]
        model_to_export.eval()
        export_inputs = to_any(to_any(data["inputs"], device), torch_dtype)
        other_inputs = [
            v for k, v in data.items() if k.startswith("inputs_") if k != "inputs_prompt"
        ]
        dynamic_shapes = data["dynamic_shapes"]
        assert other_inputs, f"No other inputs was found from data (available {set(data)})"

    elif part == "vision":

        class VisionPart(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(
                self, input_ids, input_image_embeds, image_attention_mask, image_sizes
            ):
                torch._check(input_image_embeds.shape[-2] >= 28)
                torch._check(input_image_embeds.shape[-1] >= 28)
                return model.model.embed_tokens_extend.image_embed(
                    input_ids=input_ids,
                    input_embeds=input_image_embeds,
                    image_attention_mask=image_attention_mask,
                    image_sizes=image_sizes,
                    wte=model.model.embed_tokens,
                )

        model_to_export = VisionPart(model)
        model_to_export.eval()

        dynamic_shapes = {
            "input_ids": {1: "seq_length"},
            "input_image_embeds": {
                0: "num_images",
                1: "max_num_crops",
                3: "height",
                4: "width",
            },
            "image_attention_mask": {0: "num_images", 1: "max_num_crops"},
            "image_sizes": {0: "num_images"},
        }

    else:
        raise NotImplementedError(f"no export yet for part={part!r}")

    print(f"-- part={part!r}")
    print(f"-- model_to_export={type(model_to_export)}")
    print(f"-- dynamic_shapes={dynamic_shapes}")
    print("-- ############")
    print("-- INPUT/OUTPUT")
    print("-- ############")

    input_filename = os.path.join(output_folder, f"inputs.{prefix}.{part}.{device}.{dtype}.pt")
    if os.path.exists(input_filename):
        print(f"-- restore inputs from {input_filename!r}")
        data = torch.load(input_filename, weights_only=False)
        export_inputs = data["export_inputs"]
        other_inputs = data["other_inputs"]
        dynamic_shapes = data["dynamic_shapes"]
    elif export_inputs is not None:
        data = dict(
            export_inputs=export_inputs,
            other_inputs=other_inputs,
            dynamic_shapes=dynamic_shapes,
        )
        print(f"-- dump inputs into {input_filename!r}")
        torch.save(data, input_filename)
    else:
        export_inputs, other_inputs = get_inputs_for_part(
            model_id,
            part,
            torch_dtype,
            device,
            second_input,
        )
        data = dict(
            export_inputs=export_inputs,
            other_inputs=other_inputs,
            dynamic_shapes=dynamic_shapes,
        )
        print(f"-- dump inputs into {input_filename!r}")
        torch.save(data, input_filename)

    print(f"-- export_inputs={string_type(export_inputs, with_shape=True, with_device=True)}")
    print(f"-- other_inputs={string_type(other_inputs, with_shape=True, with_device=True)}")
    print(f"-- dynamic_shapes={dynamic_shapes}")
    output_filename = os.path.join(
        output_folder, f"expected.{prefix}.visual.{device}.{dtype}.pt"
    )

    print("-- ##################")
    print("-- # EXPECTED_OUTPUTS")
    print("-- ##################")

    export_expected, *_ = compute_expected_outputs(
        output_filename, model_to_export, input_filename
    )

    if existing_onnx and os.path.exists(existing_onnx):
        print("-- ######")
        print(f"-- USING EXISTING ONNX {existing_onnx!r}")
        print("-- ######")

        exporter = existing_onnx
        filename = existing_onnx
        target_opset = None
    else:
        print("-- ######")
        print("-- EXPORT")
        print("-- ######")

        additional_patches = get_patches(main_mod, mod_siglip)

        begin = time.perf_counter()

        target_opset = 22

        details = PatchDetails()
        with torch_export_patches(
            patch_torch=True,  # needed for DynamicDimConstraintPrinter
            patch_sympy=False,
            patch_transformers=True,
            verbose=1,
            stop_if_static=0,
            profile=(f"{basename}.profile.html" if profile_exporter else None),
            custom_patches=additional_patches,
            patch_details=details,
        ):
            # let's again the patched code runs
            patched_expected = model_to_export(**export_inputs)
            diff = max_diff(export_expected, patched_expected, hist=[0.1, 0.01])
            print(f"-- discrepancies PATCHED/ORIGINAL {string_diff(diff)}")
            assert diff["abs"] < atol, (
                f"Patches do not output the same values\n"
                f"\nexpected={string_type(export_expected, with_shape=True)}"
                f"\n patched={string_type(patched_expected, with_shape=True)}"
                f"\ndiff={string_diff(diff)}"
            )
            if details and not os.path.exists(f"{basename}.patches_details.rst"):
                print("-- builds patch details")
                ep = torch.export.export(
                    model_to_export,
                    (),
                    kwargs=export_inputs,
                    dynamic_shapes=use_dyn_not_str(dynamic_shapes),
                )
                patches = details.patches_involved_in_graph(ep.graph)
                report = details.make_report(patches, format="rst")
                with open(f"{basename}.patches_details.rst", "w") as f:
                    f.write(report)
                with open(f"{basename}.ep", "w") as f:
                    f.write(str(ep))
                with open(f"{basename}.graph", "w") as f:
                    f.write(str(ep.graph))
                print("-- done writing patch details")

            to_onnx(
                model_to_export,
                kwargs=export_inputs,
                dynamic_shapes=dynamic_shapes,
                filename=filename,
                exporter=exporter,
                verbose=1,
                save_ep=None,
                target_opset=target_opset,
                optimize=True,
            )
        export_duration = time.perf_counter() - begin

    print("-- ###############")
    print("-- # DISCREPANCIES")
    print("-- ###############")

    info = {
        "model_id": model_id,
        "part": part,
        "device": device,
        "dtype": dtype,
        "exporter": exporter,
        "pretrained": pretrained,
        "attention": os.environ.get("QWEN25ATTENTION", "default"),
    }

    check_for_discrepancies_and_log_everything_into_a_json_file(
        agg_stat_file=os.path.join(output_folder, "collection_statistics.js"),
        stat_file=stat_file,
        export_duration=export_duration,
        device=device,
        model_file=filename,
        cached_inputs=input_filename,
        cached_expected_outputs=output_filename,
        main_info=info,
        atol=atol,
        mismatch01=mismatch01,
    )

    if make_zip:
        print("-- #####")
        print("-- # ZIP")
        print("-- #####")
        zip_model_and_data_into_a_single_file(f"{basename}.zip", filename)


if __name__ == "__main__":
    parser = get_parser(
        "qwen25",
        epilog=textwrap.dedent(r"""
            Tested command lines::

                python -m onnx_diagnostic.ci_models.export_phi4_mm \
                    -m microsoft/Phi-4-multimodal-instruct \
                    --device cuda --dtype float16 --exporter custom \
                    --pretrained --second-input --part vision
            """),
    )
    args = parser.parse_args(sys.argv[1:])
    main(
        model_id=args.mid,
        device=args.device,
        dtype=args.dtype,
        exporter=args.exporter,
        pretrained=args.pretrained,
        second_input=args.second_input,
        make_zip=args.zip,
        output_folder=args.output_folder,
        existing_onnx=args.existing_onnx,
        part=args.part,
        atol=args.atol,
        mismatch01=args.mismatch01,
        profile_exporter=args.profile_exporter,
    )
