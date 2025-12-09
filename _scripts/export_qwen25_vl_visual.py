"""
Export visual embedding of Qwen/Qwen2.5-VL-7B-Instruct
======================================================

Requirements
++++++++++++

::

    git+https://github.com/sdpython/experimental-experiment.git  # optional
    huggingface_hub>=1.2.1
    onnx-diagnostic>=0.8.4
    onnxruntime>=1.23
    torch>=2.9  # weekly is better
    tqdm
    transformers>=4.57

Examples
++++++++

.. code-block:: bash

    python export_qwen25_vl_visual.py -m Qwen/Qwen2.5-VL-7B-Instruct --device cpu --dtype float32 --exporter onnx-dynamo --pretrained --second-input --zip

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

import datetime
import os
import subprocess
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
    del model.graph.value_info[:]
    model = onnx.shape_inference.infer_shapes(model)
    onnx.save(model, filename, save_as_external_data=False)


def simplify_model_id_for_a_filename(model_id: str) -> str:
    return model_id.lower().replace("/", ".")


def get_versions():
    import onnx
    import onnx_diagnostic
    import onnxruntime
    import torch
    import transformers

    return {
        "transformers": transformers.__version__,
        "onnxruntime": onnxruntime.__version__,
        "onnx": onnx.__version__,
        "onnx-diagnostic": onnx_diagnostic.__version__,
        "torch": torch.__version__,
    }


def main(
    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    device: str = "cpu",
    dtype: str = "float32",
    exporter: str = "onnx-dynamo",
    pretrained: bool = True,
    second_input: bool = True,
    make_zip: bool = False,
    output_folder: str = "dump_models",
):
    print("-- import torch")
    import torch

    print("-- import onnxruntime")
    import onnxruntime

    print("-- import transformers")
    from transformers import AutoModel, AutoProcessor

    print("-- import onnx_diagnostic")
    import tqdm
    from onnx_diagnostic.helpers import string_type, string_diff, max_diff
    from onnx_diagnostic.torch_export_patches.patches._patch_transformers_qwen2_5 import (
        PLUGS,
    )
    from onnx_diagnostic.torch_export_patches import torch_export_patches
    from onnx_diagnostic.torch_models.hghub.model_inputs import get_untrained_model_with_inputs
    from onnx_diagnostic.export.api import to_onnx

    if output_folder and output_folder != ".":
        os.makedirs(output_folder, exist_ok=True)

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
    model_to_export = model.visual if hasattr(model, "visual") else model.model.visual
    print(f"-- model_to_export={type(model_to_export)}")

    print("-- ############")
    print("-- INPUT/OUTPUT")
    print("-- ############")

    prefix = simplify_model_id_for_a_filename(model_id)
    input_filename = os.path.join(output_folder, f"inputs.{prefix}.visual.{device}.{dtype}.pt")
    if os.path.exists(input_filename):
        print(f"-- restore inputs from {input_filename!r}")
        data = torch.load(input_filename)
        export_inputs = data["export_inputs"]
        other_inputs = data["other_inputs"]
    else:
        export_inputs = dict(
            hidden_states=torch.randn((1292, 1176), dtype=torch_dtype).to(device),
            grid_thw=torch.tensor([[1, 34, 38]], dtype=torch.int64).to(device),
        )
        other_inputs = []
        if second_input:
            other_inputs = [
                dict(
                    hidden_states=torch.randn((1292, 1176), dtype=torch_dtype).to(device),
                    grid_thw=torch.tensor([[1, 34, 38]], dtype=torch.int64).to(device),
                ),
                dict(
                    hidden_states=torch.rand((1292, 1176), dtype=torch_dtype).to(device),
                    grid_thw=torch.tensor([[1, 34, 38]], dtype=torch.int64).to(device),
                ),
                dict(
                    hidden_states=torch.randn((14308, 1176), dtype=torch_dtype).to(device),
                    grid_thw=torch.tensor([[1, 98, 146]], dtype=torch.int64).to(device),
                ),
                dict(
                    hidden_states=torch.rand((14308, 1176), dtype=torch_dtype).to(device),
                    grid_thw=torch.tensor([[1, 98, 146]], dtype=torch.int64).to(device),
                ),
            ]
        data = dict(export_inputs=export_inputs, other_inputs=other_inputs)
        print(f"-- dump inputs into {input_filename!r}")
        torch.save(data, input_filename)

    print(f"-- export_inputs={string_type(export_inputs, with_shape=True, with_device=True)}")
    print(f"-- other_inputs={string_type(other_inputs, with_shape=True, with_device=True)}")

    def compute_expected():
        output_filename = os.path.join(
            output_folder, f"expected.{prefix}.visual.{device}.{dtype}.pt"
        )
        if os.path.exists(output_filename):
            print(f"-- restore expected outputs from {output_filename!r}")
            expected = torch.load(output_filename)
            export_expected = expected["export_expected"]
            other_expected = expected["other_expected"]
            duration = expected["duration"]
        else:
            print(
                f"-- compute with inputs: {string_type(export_inputs, with_shape=True, with_device=True)}"
            )
            export_expected = model_to_export(**export_inputs)
            print(f"-- got: {string_type(export_expected, with_shape=True)}")
            print(
                f"-- compute with inputs: {string_type(other_inputs, with_shape=True, with_device=True)}"
            )
            begin = time.perf_counter()
            other_expected = []
            for other in tqdm.tqdm(other_inputs):
                expected = model_to_export(**other)
                other_expected.append(expected)
            duration = time.perf_counter() - begin
            print(f"-- got: {string_type(other_expected, with_shape=True, with_device=True)}")

            expected = dict(
                export_expected=export_expected,
                other_expected=other_expected,
                duration=duration,
            )
            print(f"-- dump expected outputs into {output_filename!r}")
            torch.save(expected, output_filename)
        print(f"-- computation took {duration}")
        print(
            f"-- export_expected={string_type(export_expected, with_shape=True, with_device=True)}"
        )
        print(
            f"-- other_expected={string_type(other_expected, with_shape=True, with_device=True)}"
        )
        return export_expected, other_expected, duration

    export_expected, other_expected, duration = (
        compute_expected() if not os.environ.get("STOPAT", "") else (None, None)
    )

    print("-- ######")
    print("-- EXPORT")
    print("-- ######")

    dynamic_shapes = dict(
        hidden_states={0: "hidden_width", 1: "hidden_height"},
        grid_thw={},  # {0: "n_images"}, # TODO: fix
    )

    if "QWEN25ATTENTION" in os.environ:
        prefix = f"{prefix}.{os.environ['QWEN25ATTENTION']}"
    basename = os.path.join(
        output_folder, f"model.{prefix}.visual.{device}.{dtype}.{exporter}"
    )
    filename = f"{basename}.onnx"
    print(f"-- export in {filename!r}")
    stat_file = f"{basename}.stats"
    begin = time.perf_counter()

    target_opset = 22
    if exporter == "onnx-dynamo" and device == "cuda" and "QWEN25ATTENTION" not in os.environ:
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
        if export_expected is None:
            export_expected, other_expected, duration = compute_expected()
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
        remove_inplace_body_last_input_output_type_for_loop(filename)
        print("-- done.")

    with open(stat_file, "w") as f:

        def fprint(s):
            print(s)
            f.write(f"{s}\n")

        fprint(f"-- export duration: {export_duration}")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if device == "cpu":
            providers = providers[1:]
        fprint(f"-- checking discrepancies with providers={providers!r}")
        sess = onnxruntime.InferenceSession(filename, providers=providers)

        fprint(
            f"-- export_inputs {string_type(export_inputs, with_shape=True, with_device=True)}"
        )
        fprint(
            f"-- export_expected {string_type(export_expected, with_shape=True, with_device=True)}"
        )
        feeds = {k: v.detach().cpu().numpy() for k, v in export_inputs.items()}
        small = sess.run(None, feeds)
        diff = max_diff(export_expected, small[0], hist=[0.1, 0.01])
        fprint(f"-- discrepancies={diff}")

        if second_input:
            feeds = [
                {k: v.detach().cpu().numpy() for k, v in inputs.items()}
                for inputs in other_inputs
            ]
            fprint("")
            fprint(f"-- inputs {string_type(feeds, with_shape=True, with_device=True)}")
            fprint(
                f"-- expected {string_type(other_expected, with_shape=True, with_device=True)}"
            )
            begin = time.perf_counter()
            gots = []
            for feed in tqdm.tqdm(feeds):
                gots.append(sess.run(None, feed)[0])
            oduration = time.perf_counter() - begin
            fprint(
                f"-- torch duration={duration}, onnx duration={oduration}, speedup={duration/oduration}"
            )

            info = {
                "model_id": model_id,
                "device": device,
                "dtype": dtype,
                "exporter": exporter,
                "pretrained": pretrained,
                "attention": os.environ.get("QWEN25ATTENTION", "default"),
                "timestamp": datetime.datetime.now().isoformat(),
                "export_duration": export_duration,
                "latency_torch": duration,
                "latency_ort": oduration,
                "speedup": duration / oduration,
                "opset": target_opset,
                **get_versions(),
            }
            with open(os.path.join(output_folder, "collection_statistics.js"), "a") as fs:
                for fe, e, b in zip(feeds, other_expected, gots):
                    se = string_type(fe, with_shape=True)
                    diff = max_diff(e, b, hist=[0.1, 0.01])
                    js = string_diff(diff, js=True, ratio=True, inputs=se, **info)
                    fs.write(js)
                    fs.write("\n")
                    fprint(f"-- inputs={se} -- {js}")

    statistics = os.path.join(output_folder, "collection_statistics.js")
    if os.path.exists(statistics):
        print(f"-- statistics into excel {statistics!r}")
        import pandas

        df = pandas.read_json(statistics, lines=True)
        first = [
            "timestamp",
            "model_id",
            "pretrained",
            "device",
            "dtype",
            "attention",
            "opset",
        ]
        df = df[[*first, *[c for c in df.columns if c not in set(first)]]]
        df.to_excel(statistics + ".xlsx")

    if make_zip:
        tar_file_name = f"{basename}.zip"
        print()
        print(f"-- make file {tar_file_name!r}")
        cmd = ["zip", "-v", "-1", tar_file_name]
        for name in [filename, f"{filename}.data"]:
            print(f"-- add {name!r}")
            cmd.append(name)
        print(f"-- cmd: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        print("-- done.")


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
    parser.add_argument(
        "--zip",
        default=False,
        help="Creates a file .zip with onnx file and data file.",
        action=BooleanOptionalAction,
    )
    parser.add_argument(
        "-o",
        "--output-folder",
        default="dump_models",
        help="Folders where to put the results.",
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
        make_zip=args.zip,
        output_folder=args.output_folder,
    )
