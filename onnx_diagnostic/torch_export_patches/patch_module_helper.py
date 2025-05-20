import ast


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
