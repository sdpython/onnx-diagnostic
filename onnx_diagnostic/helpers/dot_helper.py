from typing import Dict
import numpy as np
import onnx
import onnx.numpy_helper as onh
from ..reference import ExtendedReferenceEvaluator as Inference
from .onnx_helper import onnx_dtype_name, pretty_onnx, get_hidden_inputs


def _make_node_label(node: onnx.NodeProto, tiny_inits: Dict[str, str]) -> str:
    els = [f"{node.domain}.\\n{node.op_type}" if node.domain else node.op_type, "\\n("]
    ee = [tiny_inits.get(i, ".") if i else "" for i in node.input]
    for att in node.attribute:
        if att.name == "to":
            ee.append(f"{att.name}={onnx_dtype_name(att.i)}")
        elif att.name in {"to", "axis", "value_int", "stash_type", "start", "end"}:
            ee.append(f"{att.name}={att.i}")
        elif att.name in {"value_float"}:
            ee.append(f"{att.name}={att.f}")
        elif att.name in {"value_floats"}:
            ee.append(f"{att.name}={att.floats}")
        elif att.name in {"value_ints", "perm"}:
            ee.append(f"{att.name}={att.ints}")
    els.append(", ".join(ee))
    els.append(")")
    if node.op_type == "Constant":
        els.extend([" -> ", node.output[0]])
    res = "".join(els)
    if len(res) < 40:
        return res.replace("\\n(", "(")
    return res


def _make_edge_label(value_info: onnx.ValueInfoProto, multi_line: bool = False) -> str:
    itype = value_info.type.tensor_type.elem_type
    if itype == onnx.TensorProto.UNDEFINED:
        return ""
    shape = tuple(
        d.dim_param if d.dim_param else d.dim_value
        for d in value_info.type.tensor_type.shape.dim
    )
    res = [
        str(a)
        for a in [("?" if isinstance(s, str) and s.startswith("unk") else s) for s in shape]
    ]
    sshape = ",".join(res)
    if multi_line and len(sshape) > 30:
        sshape = ",\\n".join(res)
    return f"{onnx_dtype_name(itype)}({sshape})"


def to_dot(model: onnx.ModelProto) -> str:
    """
    Converts a model into a dot graph.
    Here is an example:

    .. gdot::
        :script: DOT-SECTION
        :process:

        from onnx_diagnostic.helpers.dot_helper import to_dot
        from onnx_diagnostic.export.api import to_onnx
        from onnx_diagnostic.torch_export_patches import torch_export_patches
        from onnx_diagnostic.torch_models.hghub import get_untrained_model_with_inputs

        data = get_untrained_model_with_inputs("arnir0/Tiny-LLM")
        model, inputs, ds = data["model"], data["inputs"], data["dynamic_shapes"]
        with torch_export_patches(patch_transformers=True):
            em = to_onnx(model, inputs, dynamic_shapes=ds, exporter="custom")
        dot = to_dot(em.model_proto)
        print("DOT-SECTION", dot)

    Or this one obtained with :func:`torch.onnx.export`.

    .. gdot::
        :script: DOT-SECTION
        :process:

        from onnx_diagnostic.helpers.dot_helper import to_dot
        from onnx_diagnostic.export.api import to_onnx
        from onnx_diagnostic.torch_export_patches import torch_export_patches
        from onnx_diagnostic.torch_models.hghub import get_untrained_model_with_inputs

        data = get_untrained_model_with_inputs("arnir0/Tiny-LLM")
        model, inputs, ds = data["model"], data["inputs"], data["dynamic_shapes"]
        with torch_export_patches(patch_transformers=True):
            em = to_onnx(model, kwargs=inputs, dynamic_shapes=ds, exporter="onnx-dynamo")
        dot = to_dot(em.model_proto)
        print("DOT-SECTION", dot)
    """
    _unique: Dict[int, int] = {}

    def _mkn(obj: object) -> int:
        id_obj = id(obj)
        if id_obj in _unique:
            return _unique[id_obj]
        i = len(_unique)
        _unique[id_obj] = i
        return i

    model = onnx.shape_inference.infer_shapes(model)

    op_type_colors = {
        "Shape": "#d2a81f",
        "MatMul": "#ee9999",
        "Transpose": "#ee99ee",
        "Reshape": "#eeeeee",
        "Squeeze": "#eeeeee",
        "Unsqueeze": "#eeeeee",
    }

    edge_label = {}
    for val in model.graph.value_info:
        edge_label[val.name] = _make_edge_label(val, multi_line=True)

    rows = [
        "digraph {",
        (
            "  graph [rankdir=TB, splines=true, overlap=false, nodesep=0.2, "
            "ranksep=0.2, fontsize=8];"
        ),
        '  node [style="rounded,filled", color="#888888", fontcolor="#222222", shape=box];',
        "  edge [arrowhead=vee, fontsize=7, labeldistance=-5, labelangle=0];",
    ]
    inputs = list(model.graph.input)
    outputs = list(model.graph.output)
    nodes = list(model.graph.node)
    inits = list(model.graph.initializer)
    tiny_inits = {}
    name_to_ids = {}

    for inp in inputs:
        if not inp.name:
            continue
        lab = _make_edge_label(inp)
        rows.append(f'  I_{_mkn(inp)} [label="{inp.name}\\n{lab}", fillcolor="#aaeeaa"];')
        name_to_ids[inp.name] = f"I_{_mkn(inp)}"
        edge_label[inp.name] = _make_edge_label(inp, multi_line=True)

    # Small constant --> initializer
    output_names = {n.name for n in outputs}
    for node in nodes:
        if node.op_type != "Constant" or node.output[0] in output_names:
            continue
        skip = False
        for att in node.attribute:
            if att.name == "value" and (
                len(att.t.dims) > 1 or np.prod(tuple(att.t.dims)) > 10
            ):
                skip = True
                break
        if skip:
            continue

        sess = Inference(node)
        value = sess.run(None, {})[0]
        inits.append(onh.from_array(value, name=node.output[0]))

    for init in inits:
        if init.name in name_to_ids:
            # hide optional inputs
            continue
        shape = tuple(init.dims)
        if len(shape) == 0 or (len(shape) == 1 and shape[0] < 10):
            a = onh.to_array(init)
            tiny_inits[init.name] = (
                str(a) if len(shape) == 0 else f"[{', '.join([str(i) for i in a])}]"
            )
        else:
            ls = f"{onnx_dtype_name(init.data_type)}({', '.join(map(str,shape))})"
            rows.append(f'  i_{_mkn(init)} [label="{init.name}\\n{ls}", fillcolor="#cccc00"];')
            name_to_ids[init.name] = f"i_{_mkn(init)}"
            edge_label[init.name] = ls

    for node in nodes:
        if node.op_type == "Constant" and node.output[0] in tiny_inits:
            continue
        color = op_type_colors.get(node.op_type, "#cccccc")
        label = _make_node_label(node, tiny_inits)
        rows.append(f'  {node.op_type}_{_mkn(node)} [label="{label}", fillcolor="{color}"];')
        name_to_ids.update({o: f"{node.op_type}_{_mkn(node)}" for o in node.output if o})

    # nodes
    done = set()
    for node in nodes:
        names = list(node.input)
        for i in names:
            if not i or i in tiny_inits:
                continue
            if i not in name_to_ids:
                raise ValueError(f"Unable to find {i!r}\n{pretty_onnx(model)}")
            edge = name_to_ids[i], f"{node.op_type}_{_mkn(node)}"
            if edge in done:
                continue
            done.add(edge)
            lab = edge_label.get(i, "")
            if lab:
                ls = ",".join([f'label="{lab}"'])
                lab = f" [{ls}]"
            rows.append(f"  {edge[0]} -> {edge[1]}{lab};")
        if node.op_type in {"Scan", "Loop", "If"}:
            unique = set()
            for att in node.attribute:
                if att.type == onnx.AttributeProto.GRAPH:
                    unique |= get_hidden_inputs(att.g)
            for i in unique:
                edge = name_to_ids[i], _mkn(node)  # type: ignore[assignment]
                if edge in done:
                    continue
                done.add(edge)
                rows.append(f"  {edge[0]} -> {edge[1]} [style=dotted];")

    # outputs
    for out in outputs:
        if not out.name:
            continue
        lab = _make_edge_label(out)
        rows.append(f'  O_{_mkn(out)} [label="{out.name}\\n{lab}", fillcolor="#aaaaee"];')
        edge = name_to_ids[out.name], f"O_{_mkn(out)}"
        rows.append(f"  {edge[0]} -> {edge[1]};")

    rows.append("}")
    return "\n".join(rows)
