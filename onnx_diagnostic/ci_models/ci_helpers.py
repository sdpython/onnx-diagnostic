import datetime
import os
import time
import subprocess
from argparse import ArgumentParser, BooleanOptionalAction
from typing import Any, Dict, List, Tuple
import onnx


def get_versions():
    """
    Returns the version of the package currently used.
    The output is a dictionary.
    The function uses delayed import to make to fail fast at startup.
    """
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


def get_torch_dtype_from_command_line_args(dtype: str) -> "torch.dtype":  # noqa: F821
    """
    Returns the torch dtype base on the argument provided on the command line.

    Imports are delayed to be faster when running the help of the command line.
    """
    import torch

    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    assert (
        dtype in torch_dtype
    ), f"Unexpected dtype {dtype!r}, not found in {set(torch_dtype)}."
    return torch_dtype[dtype]


def get_parser(name: str) -> ArgumentParser:
    """Creates a default parser for many models."""
    parser = ArgumentParser(
        prog=name, description=f"""Export command line for model {name!r}."""
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
    parser.add_argument(
        "-x",
        "--existing-onnx",
        default="",
        help="If an onnx file exists, only measures the discrepancies.",
    )
    parser.add_argument(
        "-p",
        "--part",
        default="visual",
        help="part of the model to export",
    )
    parser.add_argument(
        "-a",
        "--atol",
        type=float,
        default=1.0,
        help="fails if the maximum discrepancy is above that threshold",
    )
    parser.add_argument(
        "--mismatch01",
        type=float,
        default=0,
        help="fails if the ratio of mismatches at level 0.1 is above that threshold",
    )
    return parser


def remove_inplace_body_last_input_output_type_for_loop_because_they_might_be_sequences(
    filename: str,
):
    """
    Modified inplace an onnx file. It wipes out shapes provided
    in ``model.graph.value_info`` because they are wrong when a Loop outputs
    a sequence. It alose removes the types in attribute 'Body'
    of an operator Loop because it may be a tensor when a sequence is expected.
    This should not be needed in the future.
    """
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
    """Changes a model id in a way it can be used in a filename."""
    return model_id.lower().replace("/", ".")


def compute_expected_outputs(
    output_filename: str, model_to_export: "torch.nn.Module", input_filename: str  # noqa: F821
) -> Tuple[Any, List[Any], List[float]]:
    """
    Computes the expected outputs for a model.
    The function uses delayed import to make to fail fast at startup.

    It caches the expected outputs in a file. They are restored if the file exists
    or computed and saved if not.

    Imports are delayed to be faster when running the help of the command line.
    """
    import tqdm
    import torch
    from ..helpers import string_type

    inputs = torch.load(input_filename)
    export_inputs = inputs["export_inputs"]
    other_inputs = inputs["other_inputs"]

    if os.path.exists(output_filename):
        print(f"-- restore expected outputs from {output_filename!r}")
        expected = torch.load(output_filename)
        export_expected = expected["export_expected"]
        other_expected = expected["other_expected"]
        durations = expected["durations"]
    else:
        print(
            f"-- compute with inputs: "
            f"{string_type(export_inputs, with_shape=True, with_device=True)}"
        )
        export_expected = model_to_export(**export_inputs)
        print(f"-- got: {string_type(export_expected, with_shape=True)}")
        print(
            f"-- compute with inputs: "
            f"{string_type(other_inputs, with_shape=True, with_device=True)}"
        )
        other_expected = []
        durations = []
        for other in tqdm.tqdm(other_inputs):
            begin = time.perf_counter()
            expected = model_to_export(**other)
            other_expected.append(expected)
            durations.append(time.perf_counter() - begin)
        print(f"-- got: {string_type(other_expected, with_shape=True, with_device=True)}")

        expected = dict(
            export_expected=export_expected,
            other_expected=other_expected,
            durations=durations,
        )
        print(f"-- dump expected outputs into {output_filename!r}")
        torch.save(expected, output_filename)
    print(f"-- computation took {sum(durations)}")
    print(
        f"-- export_expected={string_type(export_expected, with_shape=True, with_device=True)}"
    )
    print(
        f"-- other_expected={string_type(other_expected, with_shape=True, with_device=True)}"
    )
    return export_expected, other_expected, durations


def check_for_discrepancies_and_log_everything_into_a_json_file(
    agg_stat_file: str,
    stat_file: str,
    export_duration: List[float],
    device: str,
    model_file: str,
    cached_inputs: str,
    cached_expected_outputs: str,
    main_info: Dict[str, Any],
    atol: float,
    mismatch01: float,
):
    """

    Imports are delayed to be faster when running the help of the command line.
    """
    import tqdm
    import onnxruntime
    import torch
    from ..helpers import max_diff, string_type, string_diff

    cached = (torch.load(cached_inputs), torch.load(cached_expected_outputs))
    durations = cached[0].get("durations", [])
    export_inputs = cached[0]["export_inputs"]
    other_inputs = cached[0]["other_inputs"]
    export_expected = cached[1]["export_expected"]
    other_expected = cached[1]["other_expected"]

    onx = onnx.load(model_file, load_external_data=False)
    opsets = [d for d in onx.opset_import if d.domain == ""]
    assert (
        opsets
    ), f"Unable to find standard opset in file {model_file!r}, opsets={onx.opset_import}"
    opset = opsets[0].version

    with open(stat_file, "w") as f:

        def fprint(s):
            print(s)
            f.write(f"{s}\n")

        fprint(f"-- export duration: {export_duration}")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if device == "cpu":
            providers = providers[1:]
        fprint(f"-- checking discrepancies with providers={providers!r}")
        fprint(f"-- model_file={model_file!r}")
        sess = onnxruntime.InferenceSession(model_file, providers=providers)

        fprint(
            f"-- export_inputs {string_type(export_inputs, with_shape=True, with_device=True)}"
        )
        fprint(
            f"-- export_expected "
            f"{string_type(export_expected, with_shape=True, with_device=True)}"
        )
        feeds = {k: v.detach().cpu().numpy() for k, v in export_inputs.items()}
        small = sess.run(None, feeds)
        diff = max_diff(export_expected, small[0], hist=[0.1, 0.01])
        fprint(f"-- discrepancies={diff}")
        assert diff["abs"] <= atol and diff["rep"][">0.1"] / diff["n"] <= mismatch01, (
            f"absolution tolerance is above {atol} or number of mismatches is above "
            f"{mismatch01}, dicrepancies={string_diff(diff)}"
        )

        if other_inputs and other_expected:
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
                f"-- torch duration={sum(durations[:len(gots)])}, onnx duration={oduration}, "
                f"speedup={sum(durations[:len(gots)])/oduration} n={len(gots)}"
            )

            info = {
                **main_info,
                "timestamp": datetime.datetime.now().isoformat(),
                "export_duration": export_duration,
                "latency_torch": sum(durations[: len(gots)]),
                "latency_ort": oduration,
                "speedup": sum(durations[: len(gots)]) / oduration,
                "latency_ort_n": len(gots),
                "target_opset": opset,
                **get_versions(),
            }
            with open(agg_stat_file, "a") as fs:
                for fe, e, b in zip(feeds, other_expected, gots):
                    se = string_type(fe, with_shape=True)
                    diff = max_diff(e, b, hist=[0.1, 0.01])
                    assert (
                        diff["abs"] <= atol and diff["rep"][">0.1"] / diff["n"] <= mismatch01
                    ), (
                        f"absolution tolerance is above {atol} or number of mismatches is "
                        f"above {mismatch01}, dicrepancies={string_diff(diff)}"
                    )
                    js = string_diff(diff, js=True, ratio=True, inputs=se, **info)
                    fs.write(js)
                    fs.write("\n")
                    fprint(f"-- inputs={se} -- {js}")

    if os.path.exists(agg_stat_file):
        print(f"-- statistics from {agg_stat_file!r}")
        import pandas

        df = pandas.read_json(agg_stat_file, lines=True)
        first = [
            "timestamp",
            "model_id",
            "pretrained",
            "part",
            "device",
            "dtype",
            "attention",
            "opset",
        ]
        index = [*first[1:], "exporter"]
        df = df[[*first, *[c for c in df.columns if c not in set(first)]]]
        df.to_excel(agg_stat_file + ".xlsx")

        values = [
            "abs",
            "%>0.1",
            "%>0.01",
            "export_duration",
            "speedup",
            "latency_torch",
            "latency_ort_n",
        ]
        agg = {
            **{c: "max" for c in values if c != "speedup"},
            "speedup": "min",
        }
        stat = df[[*index, *values]].groupby(index, dropna=False).agg(agg)
        stat.to_excel(agg_stat_file + ".agg.xlsx")
        stat = (
            df[df.exporter != "custom"][[*index, *values]]
            .groupby(index, dropna=False)
            .agg(agg)
        )
        stat.to_excel(agg_stat_file + ".agg.onnx-dynamo.xlsx")


def zip_model_and_data_into_a_single_file(zip_file: str, model_file: str):
    """
    Zips an onnx model and its data into a zingle file.

    :param zip_file: zip file to create
    :param model_file: onnx file
    """
    print()
    print(f"-- make file {zip_file!r}")
    cmd = ["zip", "-v", "-1", zip_file]
    for name in [model_file, f"{model_file}.data"]:
        print(f"-- add {name!r}")
        cmd.append(name)
    print(f"-- cmd: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print("-- done.")
