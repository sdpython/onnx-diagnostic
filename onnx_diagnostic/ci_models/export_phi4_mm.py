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


def get_patches(mod):
    import torch

    class patched_Phi4MMImageEmbedding(torch.nn.Module):
        _PATCHES_ = ["forward", "get_image_embeddings"]
        _PATCHED_CLASS_ = mod.Phi4MMImageEmbedding

        @classmethod
        def get_image_embeddings(
            cls,
            image_features,  # [bs, max_num_crops, base_feat_height *
            #                   base_feat_width (24*24), C]
            attention_mask,  # 4D image attention mask
            image_sizes,  # [num_images, 2]
            sub_GN,  # [1, 1, 1, image_dim_out * base_feat_height_reduction**2]
            glb_GN,  # [1, 1, image_dim_out * base_feat_height_reduction**2]
            bfht: int,  # base_feat_height_target
            crop_size: int,  # base_resolution
            bfhr: int,  # base_feat_height_reduction
            bfh: int,  # base_feat_height
            bfw: int,  # base_feat_width
            C: int,  # Channels
            H: int,  # Height
            device: torch.device,  # Target device
            dtype: torch.dtype,  # Target dtype
        ):
            """Compute HD feature transformation."""
            # Compute common constants used frequently in for-loop
            H_bfhr = H // bfhr
            bfhr_2_C = bfhr * bfhr * C
            bfh_bfhr = bfh // bfhr
            bfw_bfhr = bfw // bfhr

            all_image_embeddings = torch.empty(0, 1152).to(device)
            for i, img_size in enumerate(image_sizes):
                h, w = img_size[0], img_size[1]
                h = torch.tensor(h // crop_size, dtype=torch.int64)
                w = torch.tensor(w // crop_size, dtype=torch.int64)
                B_ = h * w

                # Compute common constants used frequently
                # that are dependent on values in for-loop
                h_bfh_bfhr = h * bfh // bfhr  # h * bfh_bfhr
                w_bfw_bfhr = w * bfw // bfhr  # w * bfw_bfhr

                # 1 x (24x24) x 1024
                global_img_feature = image_features[i, :1]

                # 1 x 12 x 12 x 4096
                glb_img = (
                    global_img_feature.reshape(1, H, H, C)
                    .reshape(1, H_bfhr, bfhr, H_bfhr, bfhr, C)
                    .contiguous()
                    .permute(0, 1, 3, 2, 4, 5)
                    .reshape(1, H_bfhr, H_bfhr, bfhr_2_C)
                    .contiguous()
                )
                temp_glb_GN = sub_GN.repeat(1, H_bfhr, 1, 1)

                # 1 x 156 x 4096
                glb_img = torch.cat([glb_img, temp_glb_GN], dim=2).reshape(1, -1, bfhr_2_C)

                # (max_num_crops-1) x (12x12) x C
                sub_img = image_features[i, 1:]
                # 16x574x1024
                # get rid of padding sub_img
                sub_img = sub_img[:B_]

                # (num_crops, 12, 2, 12, 2, 1024) -> (num_crops, 12, 12, 2, 2, 1024)
                #                                 -> (num_crops, 12*12, 4*1024)
                sub_img = (
                    sub_img.reshape(B_, H, H, C)
                    .reshape(B_, H_bfhr, bfhr, H_bfhr, bfhr, C)
                    .contiguous()
                    .permute(0, 1, 3, 2, 4, 5)
                    .reshape(B_, -1, bfhr_2_C)
                    .contiguous()
                )
                sub_img = (
                    sub_img.reshape(1, h, w, bfh_bfhr, bfw_bfhr, -1)
                    .permute(0, 1, 3, 2, 4, 5)
                    .reshape(1, h_bfh_bfhr, w_bfw_bfhr, bfhr_2_C)
                )

                reshaped_attention_mask = (
                    attention_mask[i, 1 : B_ + 1, 0::2, 0::2]
                    .reshape(1, h, w, bfh_bfhr, bfw_bfhr)
                    .permute(0, 1, 3, 2, 4)
                    .reshape(1, h_bfh_bfhr, w_bfw_bfhr)
                )
                useful_height = int(
                    reshaped_attention_mask[0, :, 0].sum().to(torch.int64).item()
                )
                useful_width = int(
                    reshaped_attention_mask[0, 0, :].sum().to(torch.int64).item()
                )
                sub_img = sub_img[:, :useful_height, :useful_width]
                temp_sub_GN = sub_GN.repeat(1, useful_height, 1, 1)

                sub_img = torch.cat([sub_img, temp_sub_GN], dim=2).reshape(1, -1, bfhr_2_C)
                # (1, num_img_tokens, 1024*4)

                # Apply hd_transform_order = 'sub_glb'
                all_image_embeddings = torch.cat(
                    [
                        all_image_embeddings,
                        sub_img.view(-1, 1152),
                        glb_GN.view(-1, 1152),
                        glb_img.view(-1, 1152),
                    ]
                )
            return all_image_embeddings

        def forward(
            self,
            input_ids: torch.LongTensor,
            input_embeds: torch.FloatTensor,
            image_attention_mask: torch.FloatTensor,
            image_sizes: torch.LongTensor,
            wte=None,
        ) -> torch.FloatTensor:
            # pixel_values: (num_images, max_num_crops, 3, H, W)
            # image_sizes: (num_images, 2).view(1, -1)
            pixel_values = input_embeds
            attention_mask = image_attention_mask

            # input_shape = input_ids.size()
            # input_ids = input_ids.view(-1, input_shape[-1])
            # positions = torch.nonzero(
            #   input_ids == mod._IMAGE_SPECIAL_TOKEN_ID, as_tuple=False)

            # Note:
            # This function is executed if positions is not empty.
            # We assume it is always verified and the test is skipped.

            if isinstance(self.img_projection, torch.nn.Sequential):
                target_device = self.img_projection[0].bias.device
                target_dtype = self.img_projection[0].bias.dtype
            else:  # It's a single nn.Linear layer
                target_device = self.img_projection.bias.device
                target_dtype = self.img_projection.bias.dtype
            assert pixel_values.ndim == 5, (
                f"(branch 1) pixel_values size: {pixel_values.size()}, "
                f"expect 5D tensor for hd transform"
            )

            # Compute image features: Nx(HW)xC
            image_features = self.get_img_features(
                pixel_values.flatten(0, 1),
                attention_mask=attention_mask.type(torch.BoolTensor)
                .flatten(0, 1)
                .to(target_device),
            )

            # Calculate height and width of base feature
            base_feat_height = base_feat_width = torch.sym_int(image_features.shape[1] ** 0.5)
            torch._check(
                base_feat_height == self.base_feat_height_target
                and base_feat_width == self.base_feat_height_target,
                lambda: (
                    f"base_feat_height: {base_feat_height}, "
                    f"base_feat_width: {base_feat_width}, "
                    f"expect {self.base_feat_height_target} features for hd transform"
                ),
            )

            # bs x max_num_crops x (bfh*bfw) x C
            bs = pixel_values.shape[0]
            image_features = image_features.view(
                bs, -1, base_feat_height * base_feat_width, self.image_dim_out
            )

            # all_image_embeddings: (num_img_tokens, 1152)
            all_image_embeddings = self.get_image_embeddings(
                image_features,
                attention_mask,
                image_sizes,
                self.sub_GN,
                self.glb_GN,
                bfht=self.base_feat_height_target,
                crop_size=self.crop_size,
                bfhr=self.base_feat_height_reduction,
                bfh=base_feat_height,
                bfw=base_feat_width,
                C=self.image_dim_out,
                H=base_feat_height,
                device=target_device,
                dtype=target_dtype,
            )

            # image_features_proj: (num_img_tokens, 3072)
            image_features_proj = self.img_projection(
                all_image_embeddings.unsqueeze(0).to(device=target_device, dtype=target_dtype)
            )
            return image_features_proj.squeeze()

    return [*get_patches_transformers(), patched_Phi4MMImageEmbedding]


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

        additional_patches = get_patches(main_mod)

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
