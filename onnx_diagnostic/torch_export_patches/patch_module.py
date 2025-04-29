import ast
import inspect
import types
import textwrap
from typing import Callable, Dict
import torch

NODE_TYPES = tuple(
    getattr(ast, k)
    for k in dir(ast)
    if "A" <= k[0] <= "Z" and isinstance(getattr(ast, k), type)
)


def _settl(node, lineno, level=0):
    if isinstance(node, (str, int, float)):
        return node
    if isinstance(node, list):
        for n in node:
            _settl(n, lineno, level=level + 1)
        return node
    if isinstance(node, NODE_TYPES):
        if not hasattr(node, "lineno") or node.lineno is None:
            node.lineno = lineno
        for k in dir(node):
            if k in {"s", "n"}:
                continue
            if k[0] == "_":
                continue
            v = getattr(node, k)
            _settl(v, max(lineno, node.lineno), level=level + 1)
    return node


class RewriteControlFlow(ast.NodeTransformer):
    def __init__(self, wrapper_name):
        self.wrapper_name = wrapper_name
        self.counter = 0
        self.current_func_args = None

    def visit_FunctionDef(self, node):
        # Capture argument names for branch functions
        old_args = self.current_func_args
        self.current_func_args = [arg.arg for arg in node.args.args]
        node.body = [self.visit(n) for n in node.body]
        self.current_func_args = old_args
        return node

    def visit_If(self, node):
        # First recurse into subnodes
        node = self.generic_visit(node)
        test_node = node.test

        # Case 1: simple assignment in both branches
        if (
            len(node.body) == 1
            and isinstance(node.body[0], ast.Assign)
            and len(node.orelse) == 1
            and isinstance(node.orelse[0], ast.Assign)
            and self.current_func_args is not None
        ):
            then_assign = node.body[0]
            else_assign = node.orelse[0]
            tgt = then_assign.targets[0]
            if (
                isinstance(tgt, ast.Name)
                and isinstance(else_assign.targets[0], ast.Name)
                and tgt.id == else_assign.targets[0].id
            ):
                self.counter += 1
                then_name = f"{self.wrapper_name}_then_{self.counter}"
                else_name = f"{self.wrapper_name}_else_{self.counter}"
                then_expr = then_assign.value
                else_expr = else_assign.value
                # extract free variables
                then_vars = sorted(
                    {
                        n.id
                        for n in ast.walk(then_expr)
                        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
                    }
                )
                else_vars = sorted(
                    {
                        n.id
                        for n in ast.walk(else_expr)
                        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
                    }
                )
                # build local funcs
                then_args = [ast.arg(arg=v, annotation=None) for v in then_vars]
                then_def = ast.FunctionDef(
                    name=then_name,
                    args=ast.arguments(
                        posonlyargs=[],
                        args=then_args,
                        kwonlyargs=[],
                        kw_defaults=[],
                        defaults=[],
                    ),
                    body=[ast.Return(then_expr)],
                    decorator_list=[],
                    returns=None,
                )
                else_args = [ast.arg(arg=v, annotation=None) for v in else_vars]
                else_def = ast.FunctionDef(
                    name=else_name,
                    args=ast.arguments(
                        posonlyargs=[],
                        args=else_args,
                        kwonlyargs=[],
                        kw_defaults=[],
                        defaults=[],
                    ),
                    body=[ast.Return(else_expr)],
                    decorator_list=[],
                    returns=None,
                )
                # fix locations
                for n in (then_def, else_def):
                    ast.copy_location(n, node)
                    ast.fix_missing_locations(n)
                    assert hasattr(n, "lineno")
                # wrapper call and assignment
                then_args_tuple = ast.Tuple(
                    [ast.Name(id=v, ctx=ast.Load()) for v in then_vars],
                    ctx=ast.Load(),
                )
                else_args_tuple = ast.Tuple(
                    [ast.Name(id=v, ctx=ast.Load()) for v in else_vars],
                    ctx=ast.Load(),
                )
                call = ast.Call(
                    func=ast.Name(id=self.wrapper_name, ctx=ast.Load()),
                    args=[
                        test_node,
                        ast.Name(id=then_name, ctx=ast.Load()),
                        ast.Name(id=else_name, ctx=ast.Load()),
                        then_args_tuple,
                        else_args_tuple,
                    ],
                    keywords=[],
                )
                assign = ast.Assign(targets=[tgt], value=call)
                ast.copy_location(assign, node)
                ast.fix_missing_locations(assign)
                return [then_def, else_def, assign]

        # Case 2: simple return in both branches
        if (
            len(node.body) == 1
            and isinstance(node.body[0], ast.Return)
            and len(node.orelse) == 1
            and isinstance(node.orelse[0], ast.Return)
            and self.current_func_args is not None
        ):
            then_ret = node.body[0]
            else_ret = node.orelse[0]
            then_expr = then_ret.value
            else_expr = else_ret.value
            self.counter += 1
            then_name = f"{self.wrapper_name}_then_{self.counter}"
            else_name = f"{self.wrapper_name}_else_{self.counter}"
            # extract free variables
            then_vars = sorted(
                {
                    n.id
                    for n in ast.walk(then_expr)
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
                }
            )
            else_vars = sorted(
                {
                    n.id
                    for n in ast.walk(else_expr)
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
                }
            )

            then_else_vars = set(_ for _ in [*then_vars, *else_vars] if _ != "torch")

            # build local funcs
            then_args = [ast.arg(arg=v, annotation=None) for v in then_else_vars]
            then_def = ast.FunctionDef(
                name=then_name,
                args=ast.arguments(
                    posonlyargs=[],
                    args=then_args,
                    kwonlyargs=[],
                    kw_defaults=[],
                    defaults=[],
                ),
                body=[ast.Return(then_expr)],
                decorator_list=[],
                returns=None,
            )
            else_args = [ast.arg(arg=v, annotation=None) for v in then_else_vars]
            else_def = ast.FunctionDef(
                name=else_name,
                args=ast.arguments(
                    posonlyargs=[],
                    args=else_args,
                    kwonlyargs=[],
                    kw_defaults=[],
                    defaults=[],
                ),
                body=[ast.Return(else_expr)],
                decorator_list=[],
                returns=None,
            )
            for n in (then_def, else_def):
                ast.copy_location(n, node)
                ast.fix_missing_locations(n)
            # wrapper call and return
            then_else_args_list = ast.List(
                [ast.Name(id=v, ctx=ast.Load()) for v in then_else_vars],
                ctx=ast.Load(),
            )

            call = ast.Call(
                func=ast.Name(id=self.wrapper_name, ctx=ast.Load()),
                args=[
                    test_node,
                    ast.Name(id=then_name, ctx=ast.Load()),
                    ast.Name(id=else_name, ctx=ast.Load()),
                    then_else_args_list,
                ],
                keywords=[],
            )
            ret = ast.Return(call)
            ast.copy_location(ret, node)
            ast.fix_missing_locations(ret)
            return [then_def, else_def, ret]
        return node

    def generic_visit(self, node):
        return super().generic_visit(node)


class RewrittenMethod:
    """
    Stores a rewritten method using
    :func:`onnx_diagnostic.torch_export_patches.patch_module.transform_method`.

    :param tree: ast tree
    :param func: callable compiled from the tree
    """

    def __init__(self, tree, func):
        self.tree = tree
        self.func = func

    @property
    def code(self) -> str:
        """Returns the source."""
        return ast.unparse(self.tree)

    def __repr__(self):
        "usual"
        return f"{self.__class__.__name__}({self.func})"


def transform_method(func: Callable, if_name="torch_cond") -> RewrittenMethod:
    """
    Returns a new function based on `func` where every test (if)
    is replaced by a call to :func:`torch.cond`.

    :param func: method or function to rewrite
    :param if_name: function calling the test
    :return: rewritten method
    """
    # Retrieve source of the function
    src = inspect.getsource(func)
    # Parse into AST
    tree = ast.parse(textwrap.dedent(src))
    # Apply transformation
    transformer = RewriteControlFlow(if_name)
    new_tree = transformer.visit(tree)
    ast.fix_missing_locations(new_tree)
    _settl(new_tree, 0)
    try:
        mod = compile(new_tree, filename="<ast>", mode="exec")
    except TypeError as e:
        if 'required field "lineno" missing from stmt' in str(e):
            # Could not find a way to avoid compilng a string.
            # The error message still pops up without indicating which node is not
            # properly set.
            code = ast.unparse(new_tree)
            mod = compile(code, filename="<source>", mode="exec")
        else:
            kws = dict(include_attributes=True, annotate_fields=True, indent=4)
            raise RuntimeError(
                f"Unable to compile code\n--CODE--\n"
                f"{ast.unparse(new_tree)}\n--TREE--\n"
                f"{ast.dump(new_tree, **kws)}"
            ) from e
    namespace: Dict[str, type] = {}
    globs = func.__globals__.copy()
    globs[if_name] = torch.cond
    exec(mod, globs, namespace)
    new_func = namespace.get(func.__name__)
    if not isinstance(new_func, types.FunctionType):
        raise RuntimeError("Transformed function not found")
    return RewrittenMethod(new_tree, new_func)
