from typing import Optional, Union
import pprint
import onnx


def optimize_model(
    algorithm: str,
    model: Union[onnx.ModelProto, str],
    output: Optional[str] = None,
    processor: Optional[str] = None,
    infer_shapes: bool = True,
    remove_shape_info: bool = False,
    verbose: int = 1,
):
    """
    Optimizes an onnx model by fusing nodes. It looks for patterns in the graphs
    and replaces them by the corresponding nodes. It also does basic optimization
    such as removing identity nodes or unused nodes.

    :param algorithm: algorithm to choose
    :param model: model to optimize as a proto or a filename
    :param output: if not empty, the optimized model is saved
    :param processor: optimization are done for the processor
    :param infer_shapes: infer shapes before optimizing, this might not be
        available for all algorithm
    :param remove_shape_info: remove shape information before saving the model
    :param verbose: verbosity level
    :return: optimized model

    The goal is to make the model faster.
    Argument patterns defines the patterns to apply or the set of patterns.
    It is possible to show statistics or to remove a particular pattern.
    Here are some environment variables which can be used to trigger
    these displays.

    Available options algorithms, default and default+runtime:

    - ``DROPPATTERN=<pattern1,patterns2,...>``: do not apply
      those patterns when optimizing a model
    - ``DUMPPATTERNS=<folder>``: dumps all matched and applied nodes when a pattern is applied
    - ``PATTERN=<pattern1,pattern2,...>``: increase verbosity
      for specific patterns to understand why one pattern was not applied,
      this shows which line is rejecting a pattern if it seems one pattern was missed
    """
    if isinstance(model, str):
        if verbose:
            print(f"[optimize_model] load {model!r}")
        proto = onnx.load(model)
        if verbose:
            print("[optimize_model] done loading.")
    else:
        proto = model

    if verbose:
        print(f"[optimize_model] optimize with {algorithm!r}")
    if algorithm in {"default", "default+onnxruntime"}:
        from experimental_experiment.xoptim import get_pattern_list
        from experimental_experiment.xbuilder import GraphBuilder, OptimizationOptions

        pats = get_pattern_list(algorithm)

        gr = GraphBuilder(
            proto,
            infer_shapes_options=infer_shapes,
            optimization_options=OptimizationOptions(
                patterns=pats,
                verbose=verbose,
                remove_unused=True,
                constant_folding=True,
                remove_identity=True,
                max_iter=max(100, len(proto.graph.node) // 2),
                processor=processor or "CPU",
            ),
        )
        if verbose:
            print(f"[optimize_model] starts optimizing with {len(pats)} patterns")
            print(f"[optimize_model] model has {len(proto.graph.node)} nodes")
        opt_onx, report = gr.to_onnx(optimize=True, return_optimize_report=True)
        if verbose:
            print("[optimize_model] optimization report")
            pprint.print(report)
            print("[optimize_model] done")

    elif algorithm == "slim":
        import onnxslim

        opt_onx = onnxslim.slim(proto, no_shape_infer=not infer_shapes)
    elif algorithm in {"ir", "os_ort"}:
        import onnx_ir
        import onnxscript.optimizer
        from onnxscript.rewriter.ort_fusions import optimize_for_ort

        model_ir = onnx_ir.from_proto(proto)
        if algorithm == "ir":
            onnxscript.optimizer.optimize(model_ir)
        else:
            optimize_for_ort(model_ir)
        opt_onx = onnx_ir.serde.serialize_model(model_ir)

    del proto
    if verbose:
        print(f"[optimize_model] done optimizing, model has {len(opt_onx.graph.node)} nodes")
    if remove_shape_info:
        if verbose:
            print(f"[optimize_model] remove shape information {len(opt_onx.graph.value_info)}")
        del opt_onx.graph.value_info[:]
        if verbose:
            print("[optimize_model] done removing shape info")

    if output:
        if verbose:
            print(f"[optimize_model] save file into {output!r}")
        onnx.save(opt_onx, output, save_as_external_data=True)
        if verbose:
            print("[optimize_model] done saving")
    return opt_onx
