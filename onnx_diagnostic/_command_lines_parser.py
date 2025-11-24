import argparse
import contextlib
import json
import os
import re
import sys
import textwrap
import time
import onnx
from typing import Any, Dict, List, Optional, Union
from argparse import ArgumentParser, RawTextHelpFormatter, BooleanOptionalAction


def get_parser_lighten() -> ArgumentParser:
    parser = ArgumentParser(
        prog="lighten",
        description=textwrap.dedent(
            """
            Removes the weights from a heavy model, stores statistics to restore
            random weights.
            """
        ),
        epilog="This is mostly used to write unit tests without adding "
        "a big onnx file to the repository.",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="onnx model to lighten",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="onnx model to output",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        default=0,
        required=False,
        help="verbosity",
    )
    return parser


def _cmd_lighten(argv: List[Any]):
    from .helpers.onnx_helper import onnx_lighten

    parser = get_parser_lighten()
    args = parser.parse_args(argv[1:])
    onx = onnx.load(args.input)
    new_onx, stats = onnx_lighten(onx, verbose=args.verbose)
    jstats = json.dumps(stats)
    if args.verbose:
        print("save file {args.input!r}")
    if args.verbose:
        print("write file {args.output!r}")
    with open(args.output, "wb") as f:
        f.write(new_onx.SerializeToString())
    name = f"{args.output}.stats"
    with open(name, "w") as f:
        f.write(jstats)
    if args.verbose:
        print("done")


def get_parser_unlighten() -> ArgumentParser:
    parser = ArgumentParser(
        prog="unlighten",
        description=textwrap.dedent(
            """
            Restores random weights for a model reduces with command lighten,
            the command expects to find a file nearby with extension '.stats'.
            """
        ),
        epilog="This is mostly used to write unit tests without adding "
        "a big onnx file to the repository.",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="onnx model to unlighten",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="onnx model to output",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        default=0,
        required=False,
        help="verbosity",
    )
    return parser


def _cmd_unlighten(argv: List[Any]):
    from .helpers.onnx_helper import onnx_unlighten

    parser = get_parser_lighten()
    args = parser.parse_args(argv[1:])
    new_onx = onnx_unlighten(args.input, verbose=args.verbose)
    if args.verbose:
        print(f"save file {args.output}")
    with open(args.output, "wb") as f:
        f.write(new_onx.SerializeToString())
    if args.verbose:
        print("done")


def get_parser_print() -> ArgumentParser:
    parser = ArgumentParser(
        prog="print",
        description="Prints the model on the standard output.",
        epilog="To show a model.",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "fmt",
        choices=["pretty", "raw", "text", "printer"],
        default="pretty",
        help=textwrap.dedent(
            """
            Prints out a model on the standard output.
            raw     - just prints the model with print(...)
            printer - onnx.printer.to_text(...)
            pretty  - an improved rendering
            text    - uses GraphRendering
            """.strip(
                "\n"
            )
        ),
    )
    parser.add_argument("input", type=str, help="onnx model to load")
    return parser


def _cmd_print(argv: List[Any]):
    parser = get_parser_print()
    args = parser.parse_args(argv[1:])
    onx = onnx.load(args.input)
    if args.fmt == "raw":
        print(onx)
    elif args.fmt == "pretty":
        from .helpers.onnx_helper import pretty_onnx

        print(pretty_onnx(onx))
    elif args.fmt == "printer":
        print(onnx.printer.to_text(onx))
    elif args.fmt == "text":
        from .helpers.graph_helper import GraphRendering

        print(GraphRendering(onx).text_rendering())
    else:
        raise ValueError(f"Unexpected value fmt={args.fmt!r}")


def get_parser_find() -> ArgumentParser:
    parser = ArgumentParser(
        prog="find",
        description=textwrap.dedent(
            """
            Look into a model and search for a set of names,
            tells which node is consuming or producing it.
            """
        ),
        epilog="Enables Some quick validation.",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="onnx model to unlighten",
    )
    parser.add_argument(
        "-n",
        "--names",
        type=str,
        required=False,
        help="Names to look at comma separated values, if 'SHADOW', "
        "search for shadowing names.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        default=0,
        type=int,
        required=False,
        help="verbosity",
    )
    parser.add_argument(
        "--v2",
        default=False,
        action=BooleanOptionalAction,
        help="Uses enumerate_results instead of onnx_find.",
    )
    return parser


def _cmd_find(argv: List[Any]):
    from .helpers.onnx_helper import onnx_find, enumerate_results, shadowing_names

    parser = get_parser_find()
    args = parser.parse_args(argv[1:])
    if args.names == "SHADOW":
        onx = onnx.load(args.input, load_external_data=False)
        s, ps = shadowing_names(onx)[:2]
        print(f"shadowing names: {s}")
        print(f"post-shadowing names: {ps}")
    elif args.v2:
        onx = onnx.load(args.input, load_external_data=False)
        res = list(
            enumerate_results(onx, name=set(args.names.split(",")), verbose=args.verbose)
        )
        if not args.verbose:
            print("\n".join(map(str, res)))
    else:
        onnx_find(args.input, verbose=args.verbose, watch=set(args.names.split(",")))


def get_parser_config() -> ArgumentParser:
    parser = ArgumentParser(
        prog="config",
        description=textwrap.dedent(
            """
            Prints out a configuration for a model id,
            prints the associated task as well.
            """
        ),
        formatter_class=RawTextHelpFormatter,
        epilog="",
    )
    parser.add_argument(
        "-m",
        "--mid",
        type=str,
        required=True,
        help="model id, usually `<author>/<name>`",
    )
    parser.add_argument(
        "-t",
        "--task",
        default=False,
        action=BooleanOptionalAction,
        help="Displays the task as well.",
    )
    parser.add_argument(
        "-c",
        "--cached",
        default=True,
        action=BooleanOptionalAction,
        help="Uses cached configuration, only available for some of them,\n"
        "mostly for unit test purposes.",
    )
    parser.add_argument(
        "--mop",
        metavar="KEY=VALUE",
        nargs="*",
        help="Additional model options, used to change some parameters of the model, "
        "example:\n  --mop attn_implementation=sdpa or --mop attn_implementation=eager",
        action=_ParseDict,
    )
    return parser


def _cmd_config(argv: List[Any]):
    from .torch_models.hghub.hub_api import get_pretrained_config, task_from_id

    parser = get_parser_config()
    args = parser.parse_args(argv[1:])
    conf = get_pretrained_config(args.mid, **(args.mop or {}))
    print(conf)
    for k, v in sorted(conf.__dict__.items()):
        if "_implementation" in k:
            print(f"config.{k}={v!r}")
    if args.task:
        print("------")
        print(f"task: {task_from_id(args.mid)}")


def _parse_json(value: str) -> Union[str, Dict[str, Any]]:
    assert isinstance(value, str), f"value should be string but value={value!r}"
    if value and value[0] == "{" and value[-1] == "}":
        # a dictionary
        return json.loads(value.replace("'", '"'))
    return value


class _ParseDict(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        d = getattr(namespace, self.dest) or {}

        if values:
            for item in values:
                split_items = item.split("=", 1)
                key = split_items[0].strip()  # we remove blanks around keys, as is logical
                value = split_items[1]

                if value in ("True", "true", "False", "false"):
                    d[key] = value in ("True", "true")
                    continue
                try:
                    d[key] = int(value)
                    continue
                except (TypeError, ValueError):
                    pass
                try:
                    d[key] = float(value)
                    continue
                except (TypeError, ValueError):
                    pass
                d[key] = _parse_json(value)

        setattr(namespace, self.dest, d)


class _BoolOrParseDictPatch(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):

        if not values:
            return
        if len(values) == 1 and values[0] in (
            "True",
            "False",
            "true",
            "false",
            "0",
            "1",
            0,
            1,
        ):
            setattr(namespace, self.dest, values[0] in ("True", "true", 1, "1"))
            return
        d = getattr(namespace, self.dest) or {}
        if not isinstance(d, dict):
            d = {
                "patch_sympy": d,
                "patch_torch": d,
                "patch_transformers": d,
                "patch_diffusers": d,
            }
        for item in values:
            split_items = item.split("=", 1)
            key = split_items[0].strip()  # we remove blanks around keys, as is logical
            value = split_items[1]

            if value in ("True", "true", "False", "false"):
                d[key] = value in ("True", "true")
                continue
            try:
                d[key] = int(value)
                continue
            except (TypeError, ValueError):
                pass
            try:
                d[key] = float(value)
                continue
            except (TypeError, ValueError):
                pass
            d[key] = _parse_json(value)

        setattr(namespace, self.dest, d)


def get_parser_validate(name: str = "validate") -> ArgumentParser:
    parser = ArgumentParser(
        prog=name,
        description=textwrap.dedent(
            """
            Validates a model for a particular task given the model id.
            It exports the model and then validates it by computing the discrepancies
            on different input sets.
            """
            if name == "validate"
            else """
            Creates a script to export  a model for a particular task given the model id.
            """
        ),
        epilog=textwrap.dedent(
            f"""
            If the model id is specified, one untrained version of it is instantiated.
            Examples:

            python -m onnx_diagnostic {name} -m microsoft/Phi-4-mini-reasoning \\
                --run -v 1 -o dump_test --no-quiet --repeat 2 --warmup 2 \\
                --dtype float16 --device cuda --patch --export onnx-dynamo --opt ir

            python -m onnx_diagnostic {name} -m microsoft/Phi-4-mini-reasoning \\
                --run -v 1 -o dump_test --no-quiet --repeat 2 --warmup 2 \\
                --dtype float16 --device cuda --patch --export custom --opt default

            python -m onnx_diagnostic {name} -m microsoft/Phi-4-mini-reasoning \\
                --run -v 1 -o dump_test --no-quiet --repeat 2 --warmup 2 \\
                --dtype float16 --device cuda --export modelbuilder

            position_ids is usually not needed, they can be removed by adding:

                --drop position_ids

            The behaviour may be modified compare the original configuration,
            the following argument can be rope_scaling to dynamic:

                --mop \"rope_scaling={{'rope_type': 'dynamic', 'factor': 10.0}}\""

            You can profile the command line by running:

                pyinstrument -m onnx_diagnostic {name} ...
                pyinstrument -r html -o profile.html -m onnx_diagnostic {name} ...
            """
        ),
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument("-m", "--mid", type=str, help="model id, usually <author>/<name>")
    parser.add_argument("-t", "--task", default=None, help="force the task to use")
    parser.add_argument("-e", "--export", help="export the model with this exporter")
    parser.add_argument("--opt", help="optimization to apply after the export")
    parser.add_argument(
        "-r",
        "--run",
        default=False,
        action=BooleanOptionalAction,
        help="Runs the model to check it runs.",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        default=False,
        action=BooleanOptionalAction,
        help="Catches exception, reports them in the summary.",
    )
    parser.add_argument(
        "--patch",
        default=True,
        action=_BoolOrParseDictPatch,
        nargs="*",
        help=textwrap.dedent(
            """
        Applies patches before exporting, it can be a boolean
        to enable to disable the patches or be more finetuned
        (default is True). It is possible to disable patch for torch
        by adding:
            --patch "patch_sympy=False" --patch "patch_torch=False"
        """.strip(
                "\n"
            )
        ),
    )
    parser.add_argument(
        "--rewrite",
        default=True,
        action=BooleanOptionalAction,
        help="Applies rewrite before exporting.",
    )
    parser.add_argument(
        "--stop-if-static",
        default=0,
        type=int,
        help="Raises an exception if a dynamic dimension becomes static.",
    )
    parser.add_argument(
        "--same-as-trained",
        default=False,
        action=BooleanOptionalAction,
        help="Validates or exports a model identical to the trained model but not trained.",
    )
    parser.add_argument(
        "--trained",
        default=False,
        action=BooleanOptionalAction,
        help="Validates or exports the trained model (requires downloading).",
    )
    parser.add_argument(
        "--inputs2",
        default=1,
        type=int,
        help=textwrap.dedent(
            """
        Validates or exports the model on a second set of inputs
        to check the exported model supports dynamism. The values is used
        as an increment to the first set of inputs. A high value may trick
        a different behavior in the model and missed by the exporter.
        """.strip(
                "\n"
            )
        ),
    )
    parser.add_argument(
        "--runtime",
        choices=["onnxruntime", "torch", "ref", "orteval", "orteval10"],
        default="onnxruntime",
        help="onnx runtime to use, `onnxruntime` by default",
    )
    parser.add_argument(
        "-o",
        "--dump-folder",
        help="A folder is created to dumps statistics,\nexported program, onnx...",
    )
    parser.add_argument(
        "--drop",
        help="Drops the following inputs names, it should be a list\n"
        "with comma separated values, example:\n"
        "--drop position_ids",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=18,
        help="onnx opset to use, 18 by default",
    )
    parser.add_argument(
        "--subfolder",
        help="Subfolder where to find the model and the configuration.",
    )
    if name == "validate":
        parser.add_argument(
            "--ortfusiontype",
            required=False,
            help=textwrap.dedent(
                """
                Applies onnxruntime fusion, this parameter should contain the
                model type or multiple values separated by `|`. `ALL` can be used
                to run them all.
                """.strip(
                    "\n"
                )
            ),
        )
    parser.add_argument("-v", "--verbose", default=0, type=int, help="verbosity")
    parser.add_argument("--dtype", help="Changes dtype if necessary.")
    parser.add_argument("--device", help="Changes the device if necessary.")
    parser.add_argument(
        "--iop",
        metavar="KEY=VALUE",
        nargs="*",
        help=textwrap.dedent(
            """
        Additional input options, used to change the default
        inputs use to export. Examples:
            --iop cls_cache=SlidingWindowCache
            --iop cls_cache=StaticCache
        """.strip(
                "\n"
            )
        ),
        action=_ParseDict,
    )
    parser.add_argument(
        "--mop",
        metavar="KEY=VALUE",
        nargs="*",
        help=textwrap.dedent(
            """
            Additional model options, used to change some parameters
            of the model. Example:
                --mop attn_implementation=sdpa --mop attn_implementation=eager"
                --mop "rope_scaling={'rope_type': 'dynamic', 'factor': 10.0}"
            """.strip(
                "\n"
            )
        ),
        action=_ParseDict,
    )
    if name == "validate":
        parser.add_argument(
            "--repeat",
            default=1,
            type=int,
            help="number of times to run the model to measures inference time",
        )
        parser.add_argument(
            "--warmup",
            default=0,
            type=int,
            help="number of times to run the model to do warmup",
        )
    parser.add_argument(
        "--outnames",
        help="This comma separated list defines the output names "
        "the onnx exporter should use.",
        default="",
    )
    if name == "validate":
        parser.add_argument(
            "--ort-logs",
            default=False,
            action=BooleanOptionalAction,
            help="Enables onnxruntime logging when the session is created",
        )
        parser.add_argument(
            "--quiet-input-sets",
            default="",
            help=textwrap.dedent(
                """
                Avoids raising an exception when an input sets does not work with
                the exported model. Example:
                    --quiet-input-sets=inputs,inputs22
                """.strip(
                    "\n"
                )
            ),
        )
    parser.add_argument(
        "--expop",
        metavar="KEY=VALUE",
        nargs="*",
        help=textwrap.dedent(
            """
            Additional exporter options, use to change some parameters
            of the model. Examples:
                --expop report=True
                --expop report=True --expop verify=True
            """.strip(
                "\n"
            )
        ),
        action=_ParseDict,
    )
    parser.add_argument(
        "--save-ep",
        default="",
        help=textwrap.dedent(
            """
            saves the exported program with torch.export.save
            and the inputs sets with torch.save,
            then command line sbs can be used to look for discrepancies.
            """
        ),
    )

    return parser


def _cmd_validate(argv: List[Any]):
    from .helpers import string_type
    from .torch_models.validate import get_inputs_for_task, validate_model
    from .tasks import supported_tasks

    parser = get_parser_validate()
    args = parser.parse_args(argv[1:])
    if not args.task and not args.mid:
        print("-- list of supported tasks:")
        print("\n".join(supported_tasks()))
    elif not args.mid:
        data = get_inputs_for_task(args.task)
        if args.verbose:
            print(f"task: {args.task}")
        max_length = max(len(k) for k in data["inputs"]) + 1
        print("-- inputs")
        for k, v in data["inputs"].items():
            print(f"  + {k.ljust(max_length)}: {string_type(v, with_shape=True)}")
        print("-- dynamic_shapes")
        for k, v in data["dynamic_shapes"].items():
            print(f"  + {k.ljust(max_length)}: {string_type(v)}")
    else:
        # Let's skip any invalid combination if known to be unsupported
        if (
            "onnx" not in (args.export or "")
            and "custom" not in (args.export or "")
            and (args.opt or "")
        ):
            print(f"validate - unsupported args: export={args.export!r}, opt={args.opt!r}")
            return
        patch_dict = args.patch if isinstance(args.patch, dict) else {"patch": args.patch}
        summary, _data = validate_model(
            model_id=args.mid,
            task=args.task,
            do_run=args.run,
            verbose=args.verbose,
            quiet=args.quiet,
            same_as_pretrained=args.same_as_trained,
            use_pretrained=args.trained,
            dtype=args.dtype,
            device=args.device,
            patch=patch_dict,
            rewrite=args.rewrite and patch_dict.get("patch", True),
            stop_if_static=args.stop_if_static,
            optimization=args.opt,
            exporter=args.export,
            dump_folder=args.dump_folder,
            drop_inputs=None if not args.drop else args.drop.split(","),
            ortfusiontype=args.ortfusiontype,
            input_options=args.iop,
            model_options=args.mop,
            subfolder=args.subfolder,
            opset=args.opset,
            runtime=args.runtime,
            repeat=args.repeat,
            warmup=args.warmup,
            inputs2=args.inputs2,
            ort_logs=args.ort_logs,
            quiet_input_sets=set(args.quiet_input_sets.split(",")),
            output_names=(
                None if len(args.outnames.strip()) < 2 else args.outnames.strip().split(",")
            ),
            exporter_options=args.expop,
            save_ep=args.save_ep,
        )
        print("")
        print("-- summary --")
        for k, v in sorted(summary.items()):
            print(f":{k},{v};")


def _cmd_export_sample(argv: List[Any]):
    from .helpers import string_type
    from .torch_models.validate import get_inputs_for_task, _make_folder_name
    from .torch_models.code_sample import code_sample
    from .tasks import supported_tasks

    parser = get_parser_validate("exportsample")
    args = parser.parse_args(argv[1:])
    if not args.task and not args.mid:
        print("-- list of supported tasks:")
        print("\n".join(supported_tasks()))
    elif not args.mid:
        data = get_inputs_for_task(args.task)
        if args.verbose:
            print(f"task: {args.task}")
        max_length = max(len(k) for k in data["inputs"]) + 1
        print("-- inputs")
        for k, v in data["inputs"].items():
            print(f"  + {k.ljust(max_length)}: {string_type(v, with_shape=True)}")
        print("-- dynamic_shapes")
        for k, v in data["dynamic_shapes"].items():
            print(f"  + {k.ljust(max_length)}: {string_type(v)}")
    else:
        # Let's skip any invalid combination if known to be unsupported
        if (
            "onnx" not in (args.export or "")
            and "custom" not in (args.export or "")
            and (args.opt or "")
        ):
            print(f"code-sample - unsupported args: export={args.export!r}, opt={args.opt!r}")
            return
        patch_dict = args.patch if isinstance(args.patch, dict) else {"patch": args.patch}
        code = code_sample(
            model_id=args.mid,
            task=args.task,
            do_run=args.run,
            verbose=args.verbose,
            quiet=args.quiet,
            same_as_pretrained=args.same_as_trained,
            use_pretrained=args.trained,
            dtype=args.dtype,
            device=args.device,
            patch=patch_dict,
            rewrite=args.rewrite and patch_dict.get("patch", True),
            stop_if_static=args.stop_if_static,
            optimization=args.opt,
            exporter=args.export,
            dump_folder=args.dump_folder,
            drop_inputs=None if not args.drop else args.drop.split(","),
            input_options=args.iop,
            model_options=args.mop,
            subfolder=args.subfolder,
            opset=args.opset,
            runtime=args.runtime,
            output_names=(
                None if len(args.outnames.strip()) < 2 else args.outnames.strip().split(",")
            ),
        )
        if args.dump_folder:
            os.makedirs(args.dump_folder, exist_ok=True)
            name = (
                _make_folder_name(
                    model_id=args.mid,
                    exporter=args.export,
                    optimization=args.opt,
                    dtype=args.dtype,
                    device=args.device,
                    subfolder=args.subfolder,
                    opset=args.opset,
                    drop_inputs=None if not args.drop else args.drop.split(","),
                    same_as_pretrained=args.same_as_trained,
                    use_pretrained=args.trained,
                    task=args.task,
                ).replace("/", "-")
                + ".py"
            )
            fullname = os.path.join(args.dump_folder, name)
            if args.verbose:
                print(f"-- prints code in {fullname!r}")
            print("--")
            with open(fullname, "w") as f:
                f.write(code)
            if args.verbose:
                print("-- done")
        else:
            print(code)


def get_parser_stats() -> ArgumentParser:
    parser = ArgumentParser(
        prog="stats",
        description="Prints out statistics on an ONNX model.",
        epilog="",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="ONNX file",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=False,
        default="",
        help="outputs the statistics in a file",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        required=False,
        default=1,
        type=int,
        help="verbosity",
    )
    parser.add_argument(
        "-e",
        "--end",
        required=False,
        default=-1,
        type=int,
        help="ends after this many tensors",
    )
    parser.add_argument(
        "-b",
        "--begin",
        required=False,
        default=0,
        type=int,
        help="starts after this many tensors",
    )
    parser.add_argument(
        "-r",
        "--regex",
        required=False,
        default="",
        type=str,
        help="Keeps only tensors whose name verifies "
        "this regular expression, empty = no filter.",
    )
    return parser


def _cmd_stats(argv: List[Any]):
    from .helpers.onnx_helper import iterator_initializer_constant, tensor_statistics

    parser = get_parser_stats()
    args = parser.parse_args(argv[1:])
    assert os.path.exists(args.input), f"Missing filename {args.input!r}"
    if args.verbose:
        print(f"Loading {args.input}")
    onx = onnx.load(args.input)
    reg = re.compile(args.regex) if args.regex else None
    data = []
    for index, (name, init) in enumerate(iterator_initializer_constant(onx)):
        if reg and not reg.search(name):
            continue
        if index < args.begin:
            continue
        if args.end > 0 and index >= args.end:
            break
        if args.verbose:
            print(f"processing {index + 1}: {name!r}")
        stats = tensor_statistics(init)
        if not args.output:
            print(f"{name}: {stats}")
        stats["name"] = name
        data.append(stats)
    if args.output:
        if args.verbose:
            print(f"saving into {args.output!r}")
        import pandas

        df = pandas.DataFrame(data)
        ext = os.path.splitext(args.output)
        if ext[-1] == ".xlsx":
            df.to_excel(args.output, index=False)
        else:
            df.to_csv(args.output, index=False)
    if args.verbose:
        print("done.")


class _ParseNamedDict(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        assert ":" in values, f"':' missing from {values!r}"
        namespace_key, rest = values.split(":", 1)
        pairs = rest.split(",")
        inner_dict = {}

        for pair in pairs:
            if "=" not in pair:
                raise argparse.ArgumentError(self, f"Expected '=' in pair '{pair}'")
            key, value = pair.split("=", 1)
            inner_dict[key] = value
        assert inner_dict, f"Unable to parse {rest!r} into a dictionary"
        if not hasattr(namespace, self.dest) or getattr(namespace, self.dest) is None:
            setattr(namespace, self.dest, {})
        assert isinstance(
            getattr(namespace, self.dest), dict
        ), f"Unexpected type for namespace.{self.dest}={getattr(namespace, self.dest)}"
        getattr(namespace, self.dest).update({namespace_key: inner_dict})


def get_parser_agg() -> ArgumentParser:
    parser = ArgumentParser(
        prog="agg",
        description=textwrap.dedent(
            """
            Aggregates statistics coming from benchmarks.
            Every run is a row. Every row is indexed by some keys,
            and produces values. Every row has a date.
            The data can come any csv files produces by benchmarks,
            it can concatenates many csv files, or csv files inside zip files.
            It produces an excel file with many tabs, one per view.
            """
        ),
        epilog=textwrap.dedent(
            """
            examples:

                python -m onnx_diagnostic agg test_agg.xlsx raw/*.zip -v 1
                python -m onnx_diagnostic agg agg.xlsx raw/*.zip raw/*.csv -v 1 \\
                    --no-raw  --keep-last-date --filter-out "exporter:test-exporter"

            Another to create timeseries:

                python -m onnx_diagnostic agg history.xlsx raw/*.csv -v 1 --no-raw \\
                    --no-recent
            """
        ),
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument("output", help="output excel file")
    parser.add_argument(
        "inputs",
        nargs="+",
        help="input csv or zip files, at least 1, it can be a name, or search path",
    )
    parser.add_argument(
        "--filter", default="rawdata_.*.csv", help="filter for input files inside zip files"
    )
    parser.add_argument(
        "--recent",
        default=True,
        action=BooleanOptionalAction,
        help="Keeps only the most recent experiment for the same of keys.",
    )
    parser.add_argument(
        "--keep-last-date",
        default=False,
        action=BooleanOptionalAction,
        help="Rewrite all dates to the last one to simplifies the analysis, "
        "this assume changing the date does not add ambiguity, if any, option "
        "--recent should be added.",
    )
    parser.add_argument(
        "--raw",
        default=True,
        action=BooleanOptionalAction,
        help="Keeps the raw data in a sheet.",
    )
    parser.add_argument("-t", "--time", default="DATE", help="Date or time column")
    parser.add_argument(
        "-k",
        "--keys",
        default="^version_.*,^model_.*,device,opt_patterns,suite,memory_peak,"
        "machine,exporter,dynamic,rtopt,dtype,device,architecture",
        help="List of columns to consider as keys, "
        "multiple values are separated by `,`\n"
        "regular expressions are allowed",
    )
    parser.add_argument(
        "--drop-keys",
        default="",
        help="Drops keys from the given list. Something it is faster "
        "to remove one than to select all the remaining ones.",
    )
    parser.add_argument(
        "-w",
        "--values",
        default="^time_.*,^disc.*,^ERR_.*,CMD,^ITER.*,^onnx_.*,^op_onnx_.*,^peak_gpu_.*",
        help="List of columns to consider as values, "
        "multiple values are separated by `,`\n"
        "regular expressions are allowed",
    )
    parser.add_argument(
        "-i", "--ignored", default="^version_.*", help="List of columns to ignore"
    )
    parser.add_argument(
        "-f",
        "--formula",
        default="speedup,bucket[speedup],ERR1,n_models,n_model_eager,"
        "n_model_running,n_model_acc01,n_model_acc001,n_model_dynamic,"
        "n_model_pass,n_model_faster,"
        "n_model_faster2x,n_model_faster3x,n_model_faster4x,n_node_attention,"
        "n_node_attention23,n_node_rotary_embedding,n_node_rotary_embedding23,"
        "n_node_gqa,n_node_layer_normalization,n_node_layer_normalization23,"
        "peak_gpu_torch,peak_gpu_nvidia,n_node_control_flow,n_node_random,"
        "n_node_constant,n_node_shape,n_node_expand,"
        "n_node_function,n_node_initializer,n_node_scatter,"
        "time_export_unbiased,onnx_n_nodes_no_cst,n_node_initializer_small",
        help="Columns to compute after the aggregation was done.",
    )
    parser.add_argument(
        "--views",
        default="agg-suite,agg-all,disc,speedup,time,time_export,err,cmd,"
        "bucket-speedup,raw-short,counts,peak-gpu,onnx",
        help=textwrap.dedent(
            """
            Views to add to the output files. Each view becomes a tab.
            A view is defined by its name, among
            agg-suite, agg-all, disc, speedup, time, time_export, err,
            cmd, bucket-speedup, raw-short, counts, peak-gpu, onnx.
            Their definition is part of class CubeLogsPerformance.
            """
        ),
    )
    parser.add_argument(
        "--csv",
        default="raw-short",
        help="Views to dump as csv files.",
    )
    parser.add_argument("-v", "--verbose", type=int, default=0, help="verbosity")
    parser.add_argument(
        "--filter-in",
        default="",
        help="adds a filter to filter in data, syntax is\n"
        '``"<column1>:<value1>;<value2>//<column2>:<value3>"`` ...',
    )
    parser.add_argument(
        "--filter-out",
        default="",
        help="adds a filter to filter out data, syntax is\n"
        '``"<column1>:<value1>;<value2>//<column2>:<value3>"`` ...',
    )
    parser.add_argument(
        "--sbs",
        help=textwrap.dedent(
            """
            Defines an exporter to compare to another, there must be at least
            two arguments defined with --sbs. Example:
                --sbs dynamo:exporter=onnx-dynamo,opt=ir,attn_impl=eager
                --sbs custom:exporter=custom,opt=default,attn_impl=eager
            """
        ),
        action=_ParseNamedDict,
    )
    return parser


def _cmd_agg(argv: List[Any]):
    from .helpers._log_helper import open_dataframe, enumerate_csv_files, filter_data
    from .helpers.log_helper import CubeLogsPerformance

    parser = get_parser_agg()
    args = parser.parse_args(argv[1:])
    reg = re.compile(args.filter)

    csv = list(
        enumerate_csv_files(
            args.inputs, verbose=args.verbose, filtering=lambda name: bool(reg.search(name))
        )
    )
    assert csv, f"No csv files in {args.inputs}, args.filter={args.filter!r}, csv={csv}"
    if args.verbose:
        from tqdm import tqdm

        loop = tqdm(csv)
    else:
        loop = csv
    dfs = []
    for c in loop:
        df = open_dataframe(c)
        assert (
            args.time in df.columns
        ), f"Missing time column {args.time!r} in {c!r}\n{df.head()}\n{sorted(df.columns)}"
        dfs.append(filter_data(df, filter_in=args.filter_in, filter_out=args.filter_out))

    drop_keys = set(args.drop_keys.split(","))
    cube = CubeLogsPerformance(
        dfs,
        time=args.time,
        keys=[a for a in args.keys.split(",") if a and a not in drop_keys],
        values=[a for a in args.values.split(",") if a],
        ignored=[a for a in args.ignored.split(",") if a],
        recent=args.recent,
        formulas={k: k for k in args.formula.split(",")},
        keep_last_date=args.keep_last_date,
    )
    cube.load(verbose=max(args.verbose - 1, 0))
    if args.verbose:
        print(f"Dumps final file into {args.output!r}")
    cube.to_excel(
        args.output,
        {k: k for k in args.views.split(",")},
        verbose=args.verbose,
        csv=args.csv.split(","),
        raw=args.raw,
        time_mask=True,
        sbs=args.sbs,
    )
    if args.verbose:
        print(f"Wrote {args.output!r}")


def get_parser_sbs() -> ArgumentParser:
    parser = ArgumentParser(
        prog="side-by-side (sbs)",
        description=textwrap.dedent(
            """
            Compares the intermediate outputs between the exported program and
            the exported onnx model. It assumes some names are common.
            The execution of the exported program and the onnx model
            are done in parallel. The device is the one used to store the
            model and the inputs.
            Where do discrepancies start? This function tries to answer that question.
            """
        ),
        epilog=textwrap.dedent(
            """
            The command line expects the following files to be saved with
            the following function. inputs is a dictionary of the input of the model.

            - torch.export.save(ep: torch.export.ExportedProgram)
            - torch.save(**inputs)
            - onnx.save(...)

            The Replay functionality is just a way to investigates a part of a model.
            It saves torch and onnx inputs, the torch outputs, and the minimal onnx model
            which shares its inputs with the exported program.
            This is used to investigate the discrepancies between the torch
            model (through the exported program) and its onnx conversion.
            This functionality dumps everything it can to disk
            so that it be replayed in a separate process.
            """
        ),
    )
    parser.add_argument(
        "-i",
        "--inputs",
        type=str,
        required=True,
        help="model inputs saved with torch.save",
    )
    parser.add_argument(
        "-e",
        "--ep",
        type=str,
        required=True,
        help=textwrap.dedent(
            """
            exported program saved with torch.export.save,
            input sets saved with torch.save,
            """
        ),
    )
    parser.add_argument(
        "-m",
        "--onnx",
        type=str,
        required=True,
        help="exported model in onnx format",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="output name to stored what the command line produces, "
        "it should be an excel file",
    )
    parser.add_argument(
        "--atol",
        default=1e-5,
        required=False,
        help="absolute tolerance",
    )
    parser.add_argument(
        "--rtol",
        default=1e-5,
        required=False,
        help="relative tolerance",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        default=0,
        required=False,
        help="verbosity",
    )
    parser.add_argument(
        "-r",
        "--ratio",
        default=100,
        required=False,
        help="Saves the result in an excel file every <ratio> nodes.",
    )
    parser.add_argument(
        "--first",
        action=BooleanOptionalAction,
        default=False,
        help="First runs the whole model.",
    )
    parser.add_argument(
        "--reset",
        required=False,
        default="",
        help=textwrap.dedent(
            """
            List of result names separated by a comma. For those results,
            the side-by-side will take torch results instead of onnx results
            to compute the rest of the onnx model.
            """
        ),
    )
    parser.add_argument(
        "-s",
        "--replay-threshold",
        type=float,
        required=False,
        default=1e6,
        help="Triggers the replay if the discrepancies are higher than this value.",
    )
    parser.add_argument(
        "-n",
        "--replay-names",
        required=False,
        default="",
        help="Triggers the replay if a result name is in this set of values (comma separated)",
    )
    parser.add_argument(
        "-t",
        "--replay-op-types",
        required=False,
        default="",
        help="Triggers the replay if an onnx type is in this set of values (comma separated)",
    )
    parser.add_argument(
        "-f",
        "--replay-folder",
        required=False,
        default="replay",
        help="If the replay is triggered, this defines the folder where everything is dumped.",
    )

    return parser


def _cmd_sbs(argv: List[Any]):
    import pandas
    import torch
    from .helpers import flatten_object, max_diff, string_diff, string_type
    from .torch_onnx.sbs import run_aligned, ReplayConfiguration
    from .reference import OnnxruntimeEvaluator

    parser = get_parser_sbs()
    args = parser.parse_args(argv[1:])

    def _size(name):
        s = os.stat(name).st_size
        return f"{s / 2**20:1.3f} Mb"

    print("-- side by side")
    print(f"-- ep:     {_size(args.ep)}: {args.ep}")
    print(f"-- inputs: {_size(args.inputs)}: {args.inputs}")
    print(f"-- onnx:   {_size(args.onnx)}: {args.onnx}")
    print(f"-- output: {args.output}")

    print(f"-- load inputs {args.inputs!r}")
    begin = time.perf_counter()
    inputs = torch.load(args.inputs)
    s = string_type(inputs, with_shape=True, with_device=True)
    print(f"-- done in {time.perf_counter() - begin:1.1f}s - {s}")

    if isinstance(inputs, dict) and len(inputs) == 2 and set(inputs) == {"args", "kwargs"}:
        margs = inputs["args"]
        mkwargs = inputs["kwargs"]
    elif isinstance(inputs, tuple):
        margs = inputs
        mkwargs = {}
    elif isinstance(inputs, dict):
        margs = tuple()
        mkwargs = inputs
    else:
        raise ValueError(
            f"Unable to infer args, kwargs from inputs {string_type(inputs, with_shape=True)}"
        )

    print("-- import transformers.modeling_outputs to register serialization functions")
    with contextlib.suppress(ImportError):
        import transformers.modeling_outputs  # noqa: F401
    print(f"-- load ep {args.ep!r}")
    begin = time.perf_counter()
    # We need to load the plugs.
    from .torch_export_patches.patches.patch_transformers import get_transformers_plugs

    plugs = get_transformers_plugs()
    assert plugs, "Missing PLUGS for Qwen2.5"
    ep = torch.export.load(args.ep)
    print(f"-- done in {time.perf_counter() - begin:1.1f}s")

    if args.first:
        print("-- compare first, run ep")
        print(f"-- args: {string_type(margs, with_shape=True, with_device=True)}")
        print(f"-- mkwargs: {string_type(mkwargs, with_shape=True, with_device=True)}")
        expected = ep.module()(*margs, **mkwargs)
        print(f"-- expected: {string_type(expected, with_shape=True, with_device=True)}")
        sess = OnnxruntimeEvaluator(args.onnx, whole=True)
        onx_inputs = flatten_object([margs, mkwargs], drop_keys=True)
        feeds = dict(zip(sess.input_names, onx_inputs))
        print(f"-- feeds: {string_type(feeds, with_shape=True, with_device=True)}")
        got = sess.run(None, feeds)
        print(f"-- got: {string_type(got, with_shape=True, with_device=True)}")
        diff = max_diff(expected, got, hist=[0.1])
        print(f"-- diff: {string_diff(diff)}")
        print("-- done")
        del sess

    print(f"-- load onnx {args.onnx!r}")
    begin = time.perf_counter()
    onx = onnx.load(args.onnx)
    print(f"-- done in {time.perf_counter() - begin:1.1f}s")

    replay_configuration = None
    if args.replay_threshold < 1e6 or args.replay_names or args.replay_op_types:
        replay_configuration = ReplayConfiguration(
            threshold=args.replay_threshold,
            selected_names=set(args.replay_names.split(",")) if args.replay_names else None,
            selected_op_types=(
                set(args.replay_op_types.split(",")) if args.replay_op_types else None
            ),
            dump_folder=args.replay_folder,
        )

    print("-- starts side-by-side")
    ratio = int(args.ratio)
    data = []
    for obs in run_aligned(
        ep,
        onx,
        run_cls=OnnxruntimeEvaluator,  # type: ignore[arg-type]
        atol=float(args.atol),
        rtol=float(args.rtol),
        verbose=int(args.verbose),
        args=margs,
        kwargs=mkwargs,
        use_tensor=True,
        reset_names=args.reset.split(","),
        exc=False,
        replay_configuration=replay_configuration,
    ):
        data.append(obs)
        if (
            obs.onnx_op_type != "initializer"
            and obs.ep_target != "placeholder"
            and len(data) % ratio == 0
        ):
            df = pandas.DataFrame(data).apply(
                lambda col: col.fillna("") if col.dtype == "object" else col
            )
            df.to_excel(args.output)
    print(f"-- final saves into {args.output!r}")
    df = pandas.DataFrame(data).apply(
        lambda col: col.fillna("") if col.dtype == "object" else col
    )
    df.to_excel(args.output, index=False)
    print("-- done")


def get_main_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="onnx_diagnostic",
        description="onnx_diagnostic main command line.\n",
        formatter_class=RawTextHelpFormatter,
        epilog=textwrap.dedent(
            """
            Type 'python -m onnx_diagnostic <cmd> --help'
            to get help for a specific command.

            agg          - aggregates statistics from multiple files
            config       - prints a configuration for a model id
            exportsample - produces a code to export a model
            find         - find node consuming or producing a result
            lighten      - makes an onnx model lighter by removing the weights,
            print        - prints the model on standard output
            sbs          - compares an exported program and a onnx model
            stats        - produces statistics on a model
            unlighten    - restores an onnx model produces by the previous experiment
            validate     - validate a model
            """
        ),
    )
    parser.add_argument(
        "cmd",
        choices=[
            "agg",
            "config",
            "exportsample",
            "find",
            "lighten",
            "print",
            "sbs",
            "stats",
            "unlighten",
            "validate",
        ],
        help="Selects a command.",
    )
    return parser


def main(argv: Optional[List[Any]] = None):
    fcts = dict(
        agg=_cmd_agg,
        config=_cmd_config,
        exportsample=_cmd_export_sample,
        find=_cmd_find,
        lighten=_cmd_lighten,
        print=_cmd_print,
        sbs=_cmd_sbs,
        stats=_cmd_stats,
        unlighten=_cmd_unlighten,
        validate=_cmd_validate,
    )

    if argv is None:
        argv = sys.argv[1:]
    if (
        len(argv) == 0
        or (len(argv) <= 1 and argv[0] not in fcts)
        or argv[-1] in ("--help", "-h")
    ):
        if len(argv) < 2:
            parser = get_main_parser()
            parser.parse_args(argv)
        else:
            parsers = dict(
                agg=get_parser_agg,
                config=get_parser_config,
                exportsample=lambda: get_parser_validate("exportsample"),  # type: ignore[operator]
                find=get_parser_find,
                lighten=get_parser_lighten,
                print=get_parser_print,
                sbs=get_parser_sbs,
                stats=get_parser_stats,
                unlighten=get_parser_unlighten,
                validate=get_parser_validate,
            )
            cmd = argv[0]
            if cmd not in parsers:
                raise ValueError(
                    f"Unknown command {cmd!r}, it should be in {list(sorted(parsers))}."
                )
            parser = parsers[cmd]()  # type: ignore[operator]
            parser.parse_args(argv[1:])
        raise RuntimeError("The programme should have exited before.")

    cmd = argv[0]
    if cmd in fcts:
        fcts[cmd](argv)
    else:
        raise ValueError(
            f"Unknown command {cmd!r}, use --help to get the list of known command."
        )
