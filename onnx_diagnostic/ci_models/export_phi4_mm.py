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
import time
from typing import Any, Dict, List, Tuple
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
                1 / self.num_patches_per_side, 1.0, 1 / self.num_patches_per_side
            )
            position_ids = torch.full(
                size=(
                    batch_size,
                    max_nb_patches_h * max_nb_patches_w,
                ),
                fill_value=0,
            )

            for batch_idx, p_attn_mask in enumerate(patch_attention_mask):
                nb_patches_h = p_attn_mask[:, 0].sum()
                nb_patches_w = p_attn_mask[0].sum()

                # PATCHED: add checks
                torch._check(nb_patches_h.item() > 0)
                torch._check(nb_patches_w.item() > 0)
                fractional_coords_h = torch.arange(0, 1 - 1e-6, 1 / nb_patches_h)
                fractional_coords_w = torch.arange(0, 1 - 1e-6, 1 / nb_patches_w)

                bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
                bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)

                pos_ids = (
                    bucket_coords_h[:, None] * self.num_patches_per_side + bucket_coords_w
                ).flatten()
                position_ids[batch_idx][p_attn_mask.view(-1).cpu()] = pos_ids

            position_ids = position_ids.to(self.position_embedding.weight.device)

            embeddings = embeddings + self.position_embedding(position_ids)
            return embeddings

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
                # # pipeline parallel
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
                # convert to bf16
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
                # positions = torch.nonzero(
                #   input_ids == _IMAGE_SPECIAL_TOKEN_ID, as_tuple=False)
                positions_tuple = torch.nonzero(
                    input_ids == _IMAGE_SPECIAL_TOKEN_ID, as_tuple=True
                )

            # logger.info(f'position size: {positions.size()} ...')
            fake_image_forward = False
            select = False
            hd_transform = False

            if isinstance(self.img_projection, torch.nn.Sequential):
                target_device = self.img_projection[0].bias.device
                target_dtype = self.img_projection[0].bias.dtype
            else:  # It's a single nn.Linear layer
                target_device = self.img_projection.bias.device
                target_dtype = self.img_projection.bias.dtype

            # Let's assume it is always true.
            if True:  # len(positions.tolist()) > 0:
                if self.use_hd_transform and img_sizes is not None and len(img_sizes):
                    hd_transform = True
                    # img_embeds: (num_images, max_num_crops, 3, H, W)
                    # img_sizes: (num_images, 2).view(1, -1)

                    bs = img_embeds.shape[0]
                    # Nx(HW)xC
                    if image_attention_mask is not None and len(image_attention_mask) > 0:
                        img_features = self.get_img_features(
                            img_embeds.flatten(0, 1),
                            attention_mask=image_attention_mask.type(torch.BoolTensor)
                            .flatten(0, 1)
                            .to(target_device),
                        )
                    else:
                        img_features = self.get_img_features(img_embeds.flatten(0, 1))

                    # base_feat_height_target = self.base_feat_height_target
                    base_resolution = self.crop_size
                    base_feat_height_reduction = self.base_feat_height_reduction

                    base_feat_height = base_feat_width = torch.sym_int(
                        img_features.shape[1] ** 0.5
                    )

                    # bs x max_num_crops x (24x24) x C
                    img_features = img_features.view(
                        bs, -1, base_feat_height * base_feat_width, self.image_dim_out
                    )
                    C = self.image_dim_out
                    H = base_feat_height

                    output_imgs = []
                    output_len = []

                    if isinstance(img_sizes, torch.Tensor):
                        img_sizes = img_sizes.view(-1, 2)
                    for _bs in range(bs):
                        h, w = img_sizes[_bs]
                        h = h // base_resolution
                        w = w // base_resolution
                        B_ = h * w

                        global_img_feature = img_features[_bs, :1]

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
                        temp_glb_GN = self.sub_GN.repeat(
                            1, H // base_feat_height_reduction, 1, 1
                        )

                        glb_img = torch.cat([glb_img, temp_glb_GN], dim=2).reshape(
                            1, -1, base_feat_height_reduction * base_feat_height_reduction * C
                        )

                        sub_img = img_features[_bs, 1:]
                        sub_img = sub_img[:B_]

                        sub_img = (
                            sub_img.reshape(B_, H, H, C)
                            .reshape(
                                B_,
                                H // base_feat_height_reduction,
                                base_feat_height_reduction,
                                H // base_feat_height_reduction,
                                base_feat_height_reduction,
                                C,
                            )
                            .contiguous()
                            .permute(0, 1, 3, 2, 4, 5)
                            .reshape(
                                B_,
                                -1,
                                base_feat_height_reduction * base_feat_height_reduction * C,
                            )
                            .contiguous()
                        )
                        sub_img = (
                            sub_img.reshape(
                                1,
                                h,
                                w,
                                base_feat_height // base_feat_height_reduction,
                                base_feat_width // base_feat_height_reduction,
                                -1,
                            )
                            .permute(0, 1, 3, 2, 4, 5)
                            .reshape(
                                1,
                                h * base_feat_height // base_feat_height_reduction,
                                w * base_feat_width // base_feat_height_reduction,
                                base_feat_height_reduction * base_feat_height_reduction * C,
                            )
                        )

                        if image_attention_mask is not None and len(image_attention_mask) > 0:
                            reshaped_image_attention_mask = (
                                image_attention_mask[_bs, 1 : B_ + 1, 0::2, 0::2]
                                .reshape(
                                    1,
                                    h,
                                    w,
                                    base_feat_height // base_feat_height_reduction,
                                    base_feat_width // base_feat_height_reduction,
                                )
                                .permute(0, 1, 3, 2, 4)
                                .reshape(
                                    1,
                                    h * base_feat_height // base_feat_height_reduction,
                                    w * base_feat_width // base_feat_height_reduction,
                                )
                            )
                            useful_height = torch.sym_int(
                                reshaped_image_attention_mask[0, :, 0].sum().item()
                            )
                            useful_width = torch.sym_int(
                                reshaped_image_attention_mask[0, 0, :].sum().item()
                            )
                            sub_img = sub_img[:, :useful_height, :useful_width]
                            temp_sub_GN = self.sub_GN.repeat(1, useful_height, 1, 1)
                            temp_len = (
                                torch.sym_int(
                                    image_attention_mask[_bs, : B_ + 1, 0::2, 0::2]
                                    .sum()
                                    .item()
                                )
                                + (useful_height + 1)
                                + base_feat_height // base_feat_height_reduction
                            )
                        else:
                            temp_sub_GN = self.sub_GN.repeat(
                                1, h * base_feat_height // base_feat_height_reduction, 1, 1
                            )
                            temp_len = torch.sym_int(
                                (h * w + 1) * self.num_img_tokens
                                + 1
                                + (h + 1) * base_feat_height // base_feat_height_reduction
                            )

                        sub_img = torch.cat([sub_img, temp_sub_GN], dim=2).reshape(
                            1, -1, base_feat_height_reduction * base_feat_height_reduction * C
                        )
                        # (1, num_img_tokens, 1024*4)

                        # glb + sub
                        if self.hd_transform_order == "glb_sub":
                            output_imgs.append(
                                torch.cat([glb_img, self.glb_GN, sub_img], dim=1)
                            )
                        elif self.hd_transform_order == "sub_glb":
                            output_imgs.append(
                                torch.cat([sub_img, self.glb_GN, glb_img], dim=1)
                            )
                        else:
                            raise NotImplementedError(
                                f"hd_transform_order = {self.hd_transform_order}, "
                                f"not implemented"
                            )

                        output_len.append(temp_len)

                    img_set_tensor = []
                    for _output_img in output_imgs:
                        img_feature_proj = self.img_projection(
                            _output_img.to(target_device).to(target_dtype)
                        )
                        img_set_tensor.append(img_feature_proj)

                else:
                    raise NotImplementedError
                select = True
            else:
                # # create a fake image tensor
                # # TODO: need define image size for different vision model
                if self.training:
                    img_embeds = torch.zeros(
                        1,
                        3,
                        self.crop_size,
                        self.crop_size,
                        dtype=target_dtype,
                        device=input_ids.device,
                    )

                    tt = (
                        self.get_img_features(img_embeds)
                        .to(target_device)
                        .to(target_dtype)
                        .reshape(-1, 1024)
                    )
                    if self.use_hd_transform:
                        img_set_tensor = self.img_projection(
                            tt.reshape(
                                -1, self.image_dim_out * self.base_feat_height_reduction**2
                            )
                            * self.glb_GN[0]
                            * self.sub_GN[0, 0]
                        )
                    else:
                        img_set_tensor = self.img_projection(tt)  # adapted visual features.
                    fake_image_forward = True

            hidden_states = kwargs["wte"](input_ids)

            if select:
                if hd_transform:
                    merged_img_set_tensor = torch.cat(img_set_tensor, dim=1).squeeze(0)
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

            if fake_image_forward and self.training:
                hidden_states = (
                    hidden_states
                    + (
                        0 * img_set_tensor[0].to(hidden_states.dtype).to(hidden_states.device)
                    ).sum()
                )

            if self.drop is not None:
                hidden_states = self.drop(hidden_states)

            return hidden_states

    return [
        *get_patches_transformers(),
        patched_Phi4MMImageEmbedding,
        patched_SiglipVisionEmbeddings,
    ]


def get_untrained_model(model_id: str, second_input: bool, verbose: int) -> Dict[str, Any]:
    """
    Returns an untrained model.

    :param model_id: model id
    :param second_input: second input set
    :param verbose: verbosity
    :return: model and data
    """
    from ..torch_models.hghub.model_inputs import get_untrained_model_with_inputs

    if model_id == "arnir0/Tiny-LLM":
        # used to run a unit test
        _config_reduction = None
    else:

        def _config_reduction(config, task):
            return {
                # "num_hidden_layers": 2,
                # "_attn_implementation": "flash_attention_2",
                "_attn_implementation": "sdpa",
            }

    config_reduction = _config_reduction
    data = get_untrained_model_with_inputs(
        model_id,
        verbose=verbose,
        add_second_input=second_input,
        config_reduction=config_reduction,
    )
    return data


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

        url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        image_1 = Image.open(requests.get(url, stream=True).raw)
        url = "https://wallpaper.dog/large/10809054.jpg"
        image_4 = Image.open(requests.get(url, stream=True).raw)

        images = [image_1, image_4]
        inputs = processor(prompt, images=images, return_tensors="pt").to(device)
        export_inputs = dict(
            input_ids=inputs["input_ids"].to(device),
            input_image_embeds=inputs["input_image_embeds"].to(torch_dtype).to(device),
            image_attention_mask=inputs["image_attention_mask"].to(torch_dtype).to(device),
            image_sizes=inputs["image_sizes"].to(device),
        )

        other_inputs = []
        if second_input:
            prompt = (
                f"{user_prompt}<|image_1|>\n<|image_2|>\n<|image_3|>\n<|image_4|>\n"
                f"What is shown in these four images?{prompt_suffix}{assistant_prompt}"
            )
            url = "https://img.freepik.com/free-photo/painting-mountain-lake-with-mountain-background_188544-9126.jpg?w=2000"
            image_2 = Image.open(requests.get(url, stream=True).raw)
            url = (
                "https://th.bing.com/th/id/OIP.gCvQ1vmPVJmrq1nnzM3ZHQHaEo?rs=1&pid=ImgDetMain"
            )
            image_3 = Image.open(requests.get(url, stream=True).raw)

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
    atol: float = 0.01,
    mismatch01: float = 0.1,
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
    from ..export.api import to_onnx

    if output_folder and output_folder != ".":
        os.makedirs(output_folder, exist_ok=True)

    print(f"-- create model {model_id!r}")
    print(
        f"-- device={device!r}, dtype={dtype!r}, exporter={exporter!r}, "
        f"pretrained={pretrained!r}"
    )
    torch_dtype = get_torch_dtype_from_command_line_args(dtype)

    with torch_export_patches(
        patch_torch=False,
        patch_sympy=False,
        patch_transformers=True,
        verbose=1,
        stop_if_static=2,
        profile=(f"{basename}.profile.html" if profile_exporter else None),
        custom_patches=get_patches_transformers(),
    ):
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
            data = get_untrained_model(model_id, second_input=second_input, verbose=1)
            model = data["model"]
            _config = data["configuration"]

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
                return model.model.embed_tokens_extend.image_embed(
                    input_ids=input_ids,
                    input_embeds=input_image_embeds,
                    image_attention_mask=image_attention_mask,
                    image_sizes=image_sizes,
                    wte=model.model.embed_tokens,
                )

        model_to_export = VisionPart(model)

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

        with torch_export_patches(
            patch_torch=False,
            patch_sympy=False,
            patch_transformers=True,
            verbose=1,
            stop_if_static=2,
            profile=(f"{basename}.profile.html" if profile_exporter else None),
            custom_patches=additional_patches,
        ):
            # let's again the patched code runs
            patched_expected = model_to_export(**export_inputs)
            diff = max_diff(export_expected, patched_expected)
            assert diff["abs"] < atol, (
                f"Patches do not output the same values\n"
                f"\nexpected={string_type(export_expected, with_shape=True)}"
                f"\n patched={string_type(patched_expected, with_shape=True)}"
                f"\ndiff={string_diff(diff)}"
            )
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
    parser = get_parser("qwen25")
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
