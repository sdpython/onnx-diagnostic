"""
export visual embedding of Qwen/Qwen2.5-VL-7B-Instruct
======================================================

requirements
++++++++++++

git+https://github.com/sdpython/experimental-experiment.git
huggingface_hub>=1.2.1
onnx-diagnostic>=0.8.4
onnxruntime>=1.23
torch>=2.9  # weekly is better
transformers>=4.57

example
+++++++

.. code-block:: bash

    python export_qwen25_vl_visual.py -m Qwen/Qwen2.5-VL-7B-Instruct --device cpu --dtype float32 --exporter custom --pretrained --second-input
"""

import os
import sys
import time
from argparse import ArgumentParser, BooleanOptionalAction


def remove_inplace_body_last_input_output_type_for_loop(filename: str):
    import onnx

    model = onnx.load(filename, load_external_data=False)
    for node in model.graph.node:
        if node.op_type == "Loop":
            g = node.attribute[0].g
            g.input[-1].type.CopyFrom(onnx.TypeProto())
            g.output[-1].type.CopyFrom(onnx.TypeProto())
    onnx.save(model, filename, save_as_external_data=False)


def main(
    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    device: str = "cpu",
    dtype: str = "float32",
    exporter: str = "onnx-dynamo",
    pretrained: bool = True,
    second_input: bool = True,
):
    print("-- import torch")
    import torch

    print("-- import onnxruntime")
    import onnxruntime

    print("-- import transformers")
    from transformers import AutoModel, AutoProcessor

    print("-- import onnx_diagnostic")
    from onnx_diagnostic.helpers import string_type, max_diff
    from onnx_diagnostic.torch_export_patches.patches._patch_transformers_qwen2_5 import (
        PLUGS,
    )
    from onnx_diagnostic.torch_export_patches import torch_export_patches
    from onnx_diagnostic.torch_models.hghub.model_inputs import get_untrained_model_with_inputs
    from onnx_diagnostic.export.api import to_onnx

    print(f"-- creating model {model_id!r}")
    print(
        f"-- device={device!r}, dtype={dtype!r}, exporter={exporter!r}, "
        f"pretrained={pretrained!r}"
    )
    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype]

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

    inputs = dict(
        hidden_states=torch.rand((1292, 1176), dtype=torch_dtype).to(device),
        grid_thw=torch.tensor([[1, 34, 38]], dtype=torch.int64).to(device),
    )
    big_inputs = (
        dict(
            hidden_states=torch.rand((14308, 1176), dtype=torch_dtype).to(device),
            grid_thw=torch.tensor([[1, 98, 146]], dtype=torch.int64).to(device),
        )
        if second_input
        else None
    )

    model_to_export = model.visual if hasattr(model, "visual") else model.model.visual
    if not os.environ.get("STOPAT", ""):
        print(f"-- compute with inputs: {string_type(inputs, with_shape=True)}")
        expected = model_to_export(**inputs)
        print(f"-- got: {string_type(expected, with_shape=True)}")
        print(f"-- compute with inputs: {string_type(big_inputs, with_shape=True)}")
        expected_big = None if big_inputs is None else model_to_export(**big_inputs)
        print(f"-- got: {string_type(expected_big, with_shape=True)}")
    else:
        expected = None
        expected_big = None
    print(f"-- expected: {string_type(expected, with_shape=True)}")

    dynamic_shapes = dict(
        hidden_states={0: "hidden_width", 1: "hidden_height"},
        grid_thw={},  # {0: "n_images"}, # TODO: fix
    )

    filename = f"qwen25_vli_visual.{device}.{dtype}.{exporter}.onnx"
    print(f"-- export in {filename!r}")
    stat_file = filename.replace(".onnx", ".stats")
    begin = time.perf_counter()

    export_inputs = inputs
    with torch_export_patches(
        patch_torch=False,
        patch_sympy=False,
        patch_transformers=True,
        verbose=1,
        stop_if_static=2,
    ):
        if expected is None:
            expected = model_to_export(**inputs)
            expected_big = None if big_inputs is None else model_to_export(**big_inputs)
        to_onnx(
            model_to_export,
            kwargs=export_inputs,
            dynamic_shapes=dynamic_shapes,
            filename=filename,
            exporter=exporter,
            verbose=1,
            save_ep=None,
            target_opset=22,
            optimize=True,
            onnx_plugs=PLUGS,
        )
    duration = time.perf_counter() - begin

    if exporter == "onnx-dynamo":
        # onnx-dynamo fails at producing function body with sequences as input / output.
        # They are replaced by tensor type one step in the model.
        print("-- remove_body_last_input_output_for_loop")
        remove_inplace_body_last_input_output_type_for_loop(filename)
        print("-- done.")

    with open(stat_file, "w") as f:

        def fprint(s):
            print(s)
            f.write(f"{s}\n")

        fprint(f"-- export duration: {duration}")
        fprint("-- checking discrepancies")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if device == "cpu":
            providers = providers[1:]
        sess = onnxruntime.InferenceSession(filename, providers=providers)

        fprint(f"-- inputs {string_type(inputs, with_shape=True)}")
        fprint(f"-- expected {string_type(expected, with_shape=True)}")
        feeds = {k: v.detach().cpu().numpy() for k, v in inputs.items()}
        small = sess.run(None, feeds)
        diff = max_diff(expected, small[0], hist=[0.1])
        fprint(f"-- discrepancies={diff}")

        if second_input:
            fprint("")
            fprint(f"-- inputs {string_type(big_inputs, with_shape=True)}")
            fprint(f"-- expected {string_type(expected_big, with_shape=True)}")
            feeds = {k: v.detach().cpu().numpy() for k, v in big_inputs.items()}
            big = sess.run(None, feeds)
            diff = max_diff(expected_big, big[0], hist=[0.1])
            fprint(f"-- discrepancies={diff}")


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="qwen25", description="""Export visual part of model Qwen 2.5 VL."""
    )
    parser.add_argument(
        "-m",
        "--mid",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="model id, default is Qwen/Qwen2.5-VL-7B-Instruct",
    )
    parser.add_argument("-d", "--device", default="cpu", help="Device, cpu (default) or cuda.")
    parser.add_argument(
        "-t", "--dtype", default="float32", help="dtype, float32 (default) or float16"
    )
    parser.add_argument(
        "-e", "--exporter", default="onnx-dynamo", help="exporter, default is onnx-dynamo"
    )
    parser.add_argument(
        "--pretrained",
        default=True,
        help="use pretrained model or a random model",
        action=BooleanOptionalAction,
    )
    parser.add_argument(
        "--second-input",
        default=True,
        help="check discrepancies with other inputs",
        action=BooleanOptionalAction,
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args(sys.argv[1:])
    main(
        model_id=args.mid,
        device=args.device,
        dtype=args.dtype,
        exporter=args.exporter,
        pretrained=args.pretrained,
        second_input=args.second_input,
    )
