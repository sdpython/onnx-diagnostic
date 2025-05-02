import argparse
import json
import sys
import textwrap
import onnx
from typing import Any, List, Optional
from argparse import ArgumentParser, RawTextHelpFormatter, BooleanOptionalAction
from textwrap import dedent


def get_parser_lighten() -> ArgumentParser:
    parser = ArgumentParser(
        prog="lighten",
        description=dedent(
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
        description=dedent(
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
        description=dedent(
            """
        Prints the model on the standard output.
        """
        ),
        epilog="To show a model.",
    )
    parser.add_argument(
        "fmt", choices=["pretty", "raw"], help="Format to use.", default="pretty"
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
    else:
        raise ValueError(f"Unexpected value fmt={args.fmt!r}")


def get_parser_find() -> ArgumentParser:
    parser = ArgumentParser(
        prog="find",
        description=dedent(
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
        help="names to look at comma separated values",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        default=0,
        required=False,
        help="verbosity",
    )
    return parser


def _cmd_find(argv: List[Any]):
    from .helpers.onnx_helper import onnx_find

    parser = get_parser_find()
    args = parser.parse_args(argv[1:])
    onnx_find(args.input, verbose=args.verbose, watch=set(args.names.split(",")))


def get_parser_config() -> ArgumentParser:
    parser = ArgumentParser(
        prog="config",
        description=dedent(
            """
        Prints out a configuration for a model id,
        prints the associated task as well.
        """
        ),
        epilog="",
    )
    parser.add_argument(
        "-m",
        "--mid",
        type=str,
        required=True,
        help="model id, usually <author>/<name>",
    )
    parser.add_argument(
        "-t",
        "--task",
        default=False,
        action=BooleanOptionalAction,
        help="displays the task as well",
    )
    parser.add_argument(
        "-c",
        "--cached",
        default=True,
        action=BooleanOptionalAction,
        help="uses cached configuration, only available for some of them, "
        "mostly for unit test purposes",
    )
    parser.add_argument(
        "--mop",
        metavar="KEY=VALUE",
        nargs="*",
        help="Additional model options, use to change some parameters of the model, "
        "example: --mop attn_implementation=eager",
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
                d[key] = value

        setattr(namespace, self.dest, d)


def get_parser_validate() -> ArgumentParser:
    parser = ArgumentParser(
        prog="test",
        description=dedent(
            """
        Prints out dummy inputs for a particular task or a model id.
        If both mid and task are empty, the command line displays the list
        of supported tasks.
        """
        ),
        epilog="If the model id is specified, one untrained version of it is instantiated.",
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
        help="runs the model to check it runs",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        default=False,
        action=BooleanOptionalAction,
        help="catches exception, report them in the summary",
    )
    parser.add_argument(
        "-p",
        "--patch",
        default=True,
        action=BooleanOptionalAction,
        help="applies patches before exporting",
    )
    parser.add_argument(
        "--stop-if-static",
        default=0,
        type=int,
        help="raises an exception if a dynamic dimension becomes static",
    )
    parser.add_argument(
        "--trained",
        default=False,
        action=BooleanOptionalAction,
        help="validate the trained model (requires downloading)",
    )
    parser.add_argument(
        "-o",
        "--dump-folder",
        help="if not empty, a folder is created to dumps statistics, "
        "exported program, onnx...",
    )
    parser.add_argument(
        "--drop",
        help="drops the following inputs names, it should be a list "
        "with comma separated values",
    )
    parser.add_argument(
        "--subfolder",
        help="subfolder where to find the model and the configuration",
    )
    parser.add_argument(
        "--ortfusiontype",
        required=False,
        help="applies onnxruntime fusion, this parameter should contain the "
        "model type or multiple values separated by `|`. `ALL` can be used "
        "to run them all",
    )
    parser.add_argument("-v", "--verbose", default=0, type=int, help="verbosity")
    parser.add_argument("--dtype", help="changes dtype if necessary")
    parser.add_argument("--device", help="changes the device if necessary")
    parser.add_argument(
        "--iop",
        metavar="KEY=VALUE",
        nargs="*",
        help="Additional input options, use to change the default "
        "inputs use to export, example: --iop cls_cache=SlidingWindowCache",
        action=_ParseDict,
    )
    parser.add_argument(
        "--mop",
        metavar="KEY=VALUE",
        nargs="*",
        help="Additional model options, use to change some parameters of the model, "
        "example: --mop attn_implementation=eager",
        action=_ParseDict,
    )
    return parser


def _cmd_validate(argv: List[Any]):
    from .helpers import string_type
    from .torch_models.test_helper import get_inputs_for_task, validate_model
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
            trained=args.trained,
            dtype=args.dtype,
            device=args.device,
            patch=args.patch,
            stop_if_static=args.stop_if_static,
            optimization=args.opt,
            exporter=args.export,
            dump_folder=args.dump_folder,
            drop_inputs=None if not args.drop else args.drop.split(","),
            ortfusiontype=args.ortfusiontype,
            input_options=args.iop,
            model_options=args.mop,
            subfolder=args.subfolder,
        )
        print("")
        print("-- summary --")
        for k, v in sorted(summary.items()):
            print(f":{k},{v};")


def get_main_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="onnx_diagnostic",
        description="onnx_diagnostic main command line.\n",
        formatter_class=RawTextHelpFormatter,
        epilog=textwrap.dedent(
            """
        Type 'python -m onnx_diagnostic <cmd> --help'
        to get help for a specific command.

        config     - prints a configuration for a model id
        find       - find node consuming or producing a result
        lighten    - makes an onnx model lighter by removing the weights,
        unlighten  - restores an onnx model produces by the previous experiment
        print      - prints the model on standard output
        validate   - validate a model
        """
        ),
    )
    parser.add_argument(
        "cmd",
        choices=["config", "find", "lighten", "print", "unlighten", "validate"],
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
