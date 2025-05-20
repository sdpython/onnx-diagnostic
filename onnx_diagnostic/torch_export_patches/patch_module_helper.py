import ast
from typing import Any, List, Optional


class OrToBitOrTransformer(ast.NodeTransformer):
    def visit_BoolOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.Or):
            new_node = node.values[0]
            for value in node.values[1:]:
                new_node = ast.BinOp(left=new_node, op=ast.BitOr(), right=value)
            return ast.copy_location(new_node, node)
        return node


def ast_or_into_bitor(node: "ast.Node") -> "ast.Node":
    """Replaces every operator ``or`` into ``|``."""
    new_node = OrToBitOrTransformer().visit(node)
    return new_node


def code_needing_rewriting(cls_name: str) -> Optional[List[Any]]:
    """
    Returns a known list of methods or functions to rewrite because of control flow
    for a specific model class.

    :param cls_name: name of the class
    :return: a list of rewriting

    .. runpython::
        :showcode:

        from onnx_diagnostic.torch_export_patches.patch_module_helper import (
            code_needing_rewriting,
        )

        print(code_needing_rewriting("BartForConditionalGeneration"))
    """
    if cls_name in {
        "BartEncoderLayer",
        "BartForConditionalGeneration",
        "PLBartEncoderLayer",
        "PLBartForConditionalGeneration",
    }:
        import transformers

        bd = dict(
            filter_node=(
                lambda node: isinstance(node, ast.If) and not isinstance(node.test, ast.Name)
            ),
            pre_rewriter=ast_or_into_bitor,
        )

        def _add(f):
            g = bd.copy()
            g["function"] = f
            return g

        return [
            _add(transformers.models.bart.modeling_bart.BartEncoderLayer.forward),
            _add(transformers.models.plbart.modeling_plbart.PLBartEncoderLayer.forward),
        ]
    return None
