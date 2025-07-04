import argparse
import json
import os
import re
import sys
import textwrap
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
        help="Additional model options, use to change some parameters of the model, "
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
                    d[key] = bool(value)
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


def get_parser_validate() -> ArgumentParser:
    parser = ArgumentParser(
        prog="validate",
        description=textwrap.dedent(
            """
            Prints out dummy inputs for a particular task or a model id.
            If both mid and task are empty, the command line displays the list
            of supported tasks.
            """
        ),
        epilog=textwrap.dedent(
            """
            If the model id is specified, one untrained version of it is instantiated.
            Examples:

            python -m onnx_diagnostic validate -m microsoft/Phi-4-mini-reasoning \\
                --run -v 1 -o dump_test --no-quiet --repeat 2 --warmup 2 \\
                --dtype float16 --device cuda --patch --export onnx-dynamo --opt ir

            python -m onnx_diagnostic validate -m microsoft/Phi-4-mini-reasoning \\
                --run -v 1 -o dump_test --no-quiet --repeat 2 --warmup 2 \\
                --dtype float16 --device cuda --patch --export custom --opt default

            python -m onnx_diagnostic validate -m microsoft/Phi-4-mini-reasoning \\
                --run -v 1 -o dump_test --no-quiet --repeat 2 --warmup 2 \\
                --dtype float16 --device cuda --export modelbuilder

            position_ids is usually not needed, they can be removed by adding:

            --drop position_ids

            The behaviour may be modified compare the original configuration,
            the following argument can be rope_scaling to dynamic:

            --mop \"rope_scaling={'rope_type': 'dynamic', 'factor': 10.0}\""
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
        action=BooleanOptionalAction,
        help="Applies patches before exporting.",
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
        help="Validates a model identical to the trained model but not trained.",
    )
    parser.add_argument(
        "--trained",
        default=False,
        action=BooleanOptionalAction,
        help="Validates the trained model (requires downloading).",
    )
    parser.add_argument(
        "--inputs2",
        default=1,
        type=int,
        help="Validates the model on a second set of inputs\n"
        "to check the exported model supports dynamism. The values is used "
        "as an increment to the first set of inputs. A high value may trick "
        "a different behavior in the model and missed by the exporter.",
    )
    parser.add_argument(
        "--runtime",
        choices=["onnxruntime", "torch", "ref"],
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
    parser.add_argument(
        "--ortfusiontype",
        required=False,
        help="Applies onnxruntime fusion, this parameter should contain the\n"
        "model type or multiple values separated by `|`. `ALL` can be used\n"
        "to run them all.",
    )
    parser.add_argument("-v", "--verbose", default=0, type=int, help="verbosity")
    parser.add_argument("--dtype", help="Changes dtype if necessary.")
    parser.add_argument("--device", help="Changes the device if necessary.")
    parser.add_argument(
        "--iop",
        metavar="KEY=VALUE",
        nargs="*",
        help="Additional input options, use to change the default"
        "inputs use to export, example:\n  --iop cls_cache=SlidingWindowCache"
        "\n  --iop cls_cache=StaticCache",
        action=_ParseDict,
    )
    parser.add_argument(
        "--mop",
        metavar="KEY=VALUE",
        nargs="*",
        help="Additional model options, use to change some parameters of the model, "
        "example:\n  --mop attn_implementation=sdpa --mop attn_implementation=eager\n  "
        "--mop \"rope_scaling={'rope_type': 'dynamic', 'factor': 10.0}\"",
        action=_ParseDict,
    )
    parser.add_argument(
        "--repeat",
        default=1,
        type=int,
        help="number of times to run the model to measures inference time",
    )
    parser.add_argument(
        "--warmup", default=0, type=int, help="number of times to run the model to do warmup"
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
            patch=args.patch,
            rewrite=args.rewrite,
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
        )
        print("")
        print("-- summary --")
        for k, v in sorted(summary.items()):
            print(f":{k},{v};")


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


def get_parser_agg() -> ArgumentParser:
    parser = ArgumentParser(
        prog="agg",
        description=textwrap.dedent(
            """
            Aggregates statistics coming from benchmarks.
            Every run is a row. Every row is indexed by some keys,
            and produces values. Every row has a date.
            """
        ),
        epilog=textwrap.dedent(
            """
            examples:\n

                python -m onnx_diagnostic agg test_agg.xlsx raw/*.zip -v 1
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
        "peak_gpu_torch,peak_gpu_nvidia,n_node_control_flow,"
        "n_node_constant,n_node_shape,n_node_expand,"
        "n_node_function,n_node_initializer,n_node_scatter,"
        "time_export_unbiased,onnx_n_nodes_no_cst,n_node_initializer_small",
        help="Columns to compute after the aggregation was done.",
    )
    parser.add_argument(
        "--views",
        default="agg-suite,agg-all,disc,speedup,time,time_export,err,cmd,"
        "bucket-speedup,raw-short,counts,peak-gpu,onnx",
        help="Views to add to the output files.",
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
        '``"<column1>:<value1>;<value2>/<column2>:<value3>"`` ...',
    )
    parser.add_argument(
        "--filter-out",
        default="",
        help="adds a filter to filter out data, syntax is\n"
        '``"<column1>:<value1>;<value2>/<column2>:<value3>"`` ...',
    )
    return parser


def _cmd_agg(argv: List[Any]):
    from .helpers.log_helper import (
        CubeLogsPerformance,
        open_dataframe,
        enumerate_csv_files,
        filter_data,
    )

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
    )
    if args.verbose:
        print(f"Wrote {args.output!r}")


def get_main_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="onnx_diagnostic",
        description="onnx_diagnostic main command line.\n",
        formatter_class=RawTextHelpFormatter,
        epilog=textwrap.dedent(
            """
            Type 'python -m onnx_diagnostic <cmd> --help'
            to get help for a specific command.

            agg        - aggregates statistics from multiple files
            config     - prints a configuration for a model id
            find       - find node consuming or producing a result
            lighten    - makes an onnx model lighter by removing the weights,
            print      - prints the model on standard output
            stats      - produces statistics on a model
            unlighten  - restores an onnx model produces by the previous experiment
            validate   - validate a model
            """
        ),
    )
    parser.add_argument(
        "cmd",
        choices=[
            "agg",
            "config",
            "find",
            "lighten",
            "print",
            "stats",
            "unlighten",
            "validate",
        ],
        help="Selects a command.",
    )
    return parser


def main(argv: Optional[List[Any]] = None):
    fcts = dict(
        lighten=_cmd_lighten,
        unlighten=_cmd_unlighten,
        print=_cmd_print,
        find=_cmd_find,
        config=_cmd_config,
        validate=_cmd_validate,
        stats=_cmd_stats,
        agg=_cmd_agg,
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
                lighten=get_parser_lighten,
                unlighten=get_parser_unlighten,
                print=get_parser_print,
                find=get_parser_find,
                config=get_parser_config,
                validate=get_parser_validate,
                stats=get_parser_stats,
                agg=get_parser_agg,
            )
            cmd = argv[0]
            if cmd not in parsers:
                raise ValueError(
                    f"Unknown command {cmd!r}, it should be in {list(sorted(parsers))}."
                )
            parser = parsers[cmd]()
            parser.parse_args(argv[1:])
        raise RuntimeError("The programme should have exited before.")

    cmd = argv[0]
    if cmd in fcts:
        fcts[cmd](argv)
    else:
        raise ValueError(
            f"Unknown command {cmd!r}, use --help to get the list of known command."
        )
