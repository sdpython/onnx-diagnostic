r"""
Export visual and embedding parts of Qwen/Qwen2.5-VL-7B-Instruct
================================================================

Requirements
++++++++++++

::

    git+https://github.com/sdpython/experimental-experiment.git  # optional
    huggingface_hub>=1.2.1
    onnx-diagnostic>=0.8.6
    onnxruntime>=1.23
    torch>=2.9  # weekly is better
    tqdm
    transformers>=4.57

Examples
++++++++

.. code-block:: bash

    python -m onnx_diagnostic.ci_models.export_qwen25_vl \
        -m Qwen/Qwen2.5-VL-7B-Instruct \
        --device cpu --dtype float32 --exporter onnx-dynamo --pretrained --second-input --zip

To choose a specific Attention schema:

.. code-block:: bash

    QWEN25ATTENTION=LOOPMHA python -m onnx_diagnostic.ci_models.export_qwen25_vl \
        -m Qwen/Qwen2.5-VL-7B-Instruct \
        --device cpu --dtype float32 --exporter onnx-dynamo --pretrained --second-input --zip

Cheat sheet for tar commands. To make a tar:
``tar -czvf model.tar.gz model.onnx model.data``
And to untar:
``tar -xzvf model.tar.gz``.

Rewritings
++++++++++

* `overview <https://sdpython.github.io/doc/onnx-diagnostic/dev/status/patches_diff.html#auto-patch-transformers-qwen2-5-vlforconditionalgeneration-prepare-inputs-for-generation-patched-qwen2-5-vlforconditionalgeneration-prepare-inputs-for-generation>`_
* code: `_patch_transformers_qwen2_5.py <https://github.com/sdpython/onnx-diagnostic/blob/main/onnx_diagnostic/torch_export_patches/patches/_patch_transformers_qwen2_5.py>`_

Attention
+++++++++

The attention is either implemented with ``MultiHeadAttention`` in a loop,
either with ``PackedMultiHeadAttention``. The choice is made based on the device.
It is possible to overwrite this by by setting environment variable
``QWEN25ATTENTION`` to:

* ``PACKED``: PackedMultiHeadAttention
* ``LOOPMHA``: Loop over MultiHeadAttention
* ``LOOPA23``: Loop over Attention(23), needs opset 23+.
"""

import os
import sys
import time
from typing import Dict, List, Tuple
from .ci_helpers import (
    check_for_discrepancies_and_log_everything_into_a_json_file,
    compute_expected_outputs,
    get_parser,
    get_torch_dtype_from_command_line_args,
    remove_inplace_body_last_input_output_type_for_loop_because_they_might_be_sequences,
    simplify_model_id_for_a_filename,
    zip_model_and_data_into_a_single_file,
)


def get_inputs_for_visual_part(
    part: str, torch_dtype: "torch.dtype", device: str, second_input: bool  # noqa: F821
) -> Tuple[Dict[str, "torch.Tensor"], List[Dict[str, "torch.Tensor"]]]:  # noqa: F821
    import torch

    if part == "visual":

        export_inputs = dict(
            pixel_values=torch.randn((1292, 1176), dtype=torch_dtype).to(device),
            image_grid_thw=torch.tensor([[1, 34, 38]], dtype=torch.int64).to(device),
        )
        other_inputs = []
        if second_input:
            other_inputs = [
                dict(
                    pixel_values=torch.randn((1292, 1176), dtype=torch_dtype).to(device),
                    image_grid_thw=torch.tensor([[1, 34, 38]], dtype=torch.int64).to(device),
                ),
                dict(
                    pixel_values=torch.rand((1292, 1176), dtype=torch_dtype).to(device),
                    image_grid_thw=torch.tensor([[1, 34, 38]], dtype=torch.int64).to(device),
                ),
                dict(
                    pixel_values=torch.randn((14308, 1176), dtype=torch_dtype).to(device),
                    image_grid_thw=torch.tensor([[1, 98, 146]], dtype=torch.int64).to(device),
                ),
                dict(
                    pixel_values=torch.rand((14308, 1176), dtype=torch_dtype).to(device),
                    image_grid_thw=torch.tensor([[1, 98, 146]], dtype=torch.int64).to(device),
                ),
            ]
        return export_inputs, other_inputs

    raise NotImplementedError(f"No inputs yet implement for part={part!r}")


def main(
    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    device: str = "cpu",
    dtype: str = "float32",
    exporter: str = "onnx-dynamo",
    pretrained: bool = True,
    second_input: bool = True,
    make_zip: bool = False,
    output_folder: str = "dump_models",
    existing_onnx: str | None = None,
    part: str = "visual",
    atol: float = 0.01,
    mismatch01: float = 0.1,
):
    prefix = simplify_model_id_for_a_filename(model_id)
    if "QWEN25ATTENTION" in os.environ:
        prefix = f"{prefix}.{os.environ['QWEN25ATTENTION']}"
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
    print("------------------------------------------------------------------")
    print(f"-- prefix={prefix}")
    print(f"-- export in {filename!r}")
    print("------------------------------------------------------------------")

    if os.path.exists(stat_file) and not existing_onnx:
        print(f"-- skipping because {stat_file!r} already exists")
        return

    print("-- import torch and others")
    import torch
    from transformers import AutoModel, AutoProcessor
    from ..helpers import string_type
    from ..torch_export_patches.patches._patch_transformers_qwen2_5 import (
        PLUGS,
    )
    from ..torch_export_patches import torch_export_patches
    from ..torch_models.hghub.model_inputs import get_untrained_model_with_inputs
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
        model = AutoModel.from_pretrained(
            model_id, device_map=device, dtype=torch_dtype, attn_implementation="sdpa"
        ).eval()
    else:
        print("-- random model")

        def _config_reduction(config, task):
            return {
                # "num_hidden_layers": 2,
                "text_config": {
                    "num_hidden_layers": 2,
                    "layer_types": ["full_attention", "full_attention"],
                },
                # "_attn_implementation": "flash_attention_2",
                "_attn_implementation": "sdpa",
                "dtype": "float16",
            }

        config_reduction = _config_reduction
        data = get_untrained_model_with_inputs(
            model_id, verbose=1, add_second_input=False, config_reduction=config_reduction
        )
        model = data["model"]

    model = model.to(device).to(getattr(torch, dtype))

    print(f"-- config._attn_implementation={model.config._attn_implementation}")
    print(f"-- model.dtype={model.dtype}")
    print(f"-- model.device={model.device}")
    processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
    print(f"-- processor={type(processor)}")

    if part == "visual":

        class VisualPart(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, pixel_values, image_grid_thw):
                return model.get_image_features(pixel_values, image_grid_thw)

        assert hasattr(
            model, "get_image_features"
        ), f"get_image_features not found in class {type(model)}"
        model_to_export = VisualPart(model)

        print(f"-- part={part!r}")
        print(f"-- model_to_export={type(model_to_export)}")
    else:
        raise NotImplementedError(f"no export yet for part={part!r}")

    print("-- ############")
    print("-- INPUT/OUTPUT")
    print("-- ############")

    input_filename = os.path.join(output_folder, f"inputs.{prefix}.{part}.{device}.{dtype}.pt")
    if os.path.exists(input_filename):
        print(f"-- restore inputs from {input_filename!r}")
        data = torch.load(input_filename)
        export_inputs = data["export_inputs"]
        other_inputs = data["other_inputs"]
    else:
        export_inputs, other_inputs = get_inputs_for_visual_part(
            part, torch_dtype, device, second_input
        )
        data = dict(export_inputs=export_inputs, other_inputs=other_inputs)
        print(f"-- dump inputs into {input_filename!r}")
        torch.save(data, input_filename)

    print(f"-- export_inputs={string_type(export_inputs, with_shape=True, with_device=True)}")
    print(f"-- other_inputs={string_type(other_inputs, with_shape=True, with_device=True)}")
    output_filename = os.path.join(
        output_folder, f"expected.{prefix}.visual.{device}.{dtype}.pt"
    )

    print("-- ##################")
    print("-- # EXPECTED_OUTPUTS")
    print("-- ##################")

    compute_expected_outputs(output_filename, model_to_export, input_filename)

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

        dynamic_shapes = dict(
            pixel_values={0: "hidden_width", 1: "hidden_height"},
            image_grid_thw={},  # {0: "n_images"}, # TODO: fix
        )

        begin = time.perf_counter()

        target_opset = 22
        if (
            exporter == "onnx-dynamo"
            and device == "cuda"
            and "QWEN25ATTENTION" not in os.environ
        ):
            os.environ["QWEN25ATTENTION"] = "PACKED"
        elif "QWEN25ATTENTION" in os.environ and os.environ["QWEN25ATTENTION"] == "LOOPA23":
            target_opset = 23

        with torch_export_patches(
            patch_torch=False,
            patch_sympy=False,
            patch_transformers=True,
            verbose=1,
            stop_if_static=2,
        ):
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
                onnx_plugs=PLUGS,
            )
        export_duration = time.perf_counter() - begin

        if exporter == "onnx-dynamo":
            # onnx-dynamo fails at producing function body with sequences as input / output.
            # They are replaced by tensor type one step in the model.
            print("-- remove_body_last_input_output_for_loop")
            remove_inplace_body_last_input_output_type_for_loop_because_they_might_be_sequences(
                filename
            )
            print("-- done.")

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

    print("-- #####")
    print("-- # ZIP")
    print("-- #####")

    if make_zip:
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
    )
