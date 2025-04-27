import ast
import inspect
import types
import textwrap


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
                # wrapper call and assignment
                then_args_tuple = ast.Tuple(
                    [ast.Name(id=v, ctx=ast.Load()) for v in then_vars], ctx=ast.Load()
                )
                else_args_tuple = ast.Tuple(
                    [ast.Name(id=v, ctx=ast.Load()) for v in else_vars], ctx=ast.Load()
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
            # build local funcs
            then_args = [ast.arg(arg=v, annotation=None) for v in then_vars]
            then_def = ast.FunctionDef(
                name=then_name,
                args=ast.arguments(
                    posonlyargs=[], args=then_args, kwonlyargs=[], kw_defaults=[], defaults=[]
                ),
                body=[ast.Return(then_expr)],
                decorator_list=[],
                returns=None,
            )
            else_args = [ast.arg(arg=v, annotation=None) for v in else_vars]
            else_def = ast.FunctionDef(
                name=else_name,
                args=ast.arguments(
                    posonlyargs=[], args=else_args, kwonlyargs=[], kw_defaults=[], defaults=[]
                ),
                body=[ast.Return(else_expr)],
                decorator_list=[],
                returns=None,
            )
            for n in (then_def, else_def):
                ast.copy_location(n, node)
                ast.fix_missing_locations(n)
            # wrapper call and return
            then_args_tuple = ast.Tuple(
                [ast.Name(id=v, ctx=ast.Load()) for v in then_vars], ctx=ast.Load()
            )
            else_args_tuple = ast.Tuple(
                [ast.Name(id=v, ctx=ast.Load()) for v in else_vars], ctx=ast.Load()
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
            ret = ast.Return(call)
            ast.copy_location(ret, node)
            ast.fix_missing_locations(ret)
            return [then_def, else_def, ret]
        return node

    def generic_visit(self, node):
        return super().generic_visit(node)


def _fix_missing_locations_node(node):
    if not hasattr(node, "lineno"):
        node.lineno = 999
    for chi in ast.iter_child_nodes(node):
        _fix_missing_locations_node(chi)


def _fix_missing_locations(new_tree):
    for node in ast.walk(new_tree):
        _fix_missing_locations_node(node)


def transform_method(func, wrapper_name="torch_cond"):
    """
    Returns a new function based on `func` where every test (if, while, assert,
    ternary, comparison, boolean op) is replaced by a call to `wrapper_name`.

    wrapper_name should refer to a function taking a single boolean argument.
    """
    # Retrieve source of the function
    src = inspect.getsource(func)
    # Parse into AST
    tree = ast.parse(textwrap.dedent(src))
    # Apply transformation
    transformer = RewriteControlFlow(wrapper_name)
    new_tree = transformer.visit(tree)
    ast.fix_missing_locations(new_tree)

    # fix other location
    _fix_missing_locations(new_tree)
    mod = compile(new_tree, filename="<ast>", mode="exec")
    namespace = {}
    exec(mod, func.__globals__, namespace)
    new_func = namespace.get(func.__name__)
    if not isinstance(new_func, types.FunctionType):
        raise RuntimeError("Transformed function not found")
    return new_tree, new_func
