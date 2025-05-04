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
    """
    The class rewrites tests with function ``torch_cond`` :func:`torch.cond`.
    """

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

    def _find_id(self, exprs):
        vars = []
        for expr in exprs:
            for n in ast.walk(expr):
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                    vars.append(n.id)
        return sorted(set(vars))

    def _rewrite_if(self, node, then_exprs, else_exprs, tgt_mapping=None):
        test_node = node.test

        # extract free variables
        then_name = f"{self.wrapper_name}_then_{self.counter}"
        else_name = f"{self.wrapper_name}_else_{self.counter}"
        then_vars = self._find_id(then_exprs)
        else_vars = self._find_id(else_exprs)
        then_else_vars = set(_ for _ in [*then_vars, *else_vars] if _ != "torch")
        then_ret, else_ret = None, None
        if tgt_mapping is None and len(then_exprs) == 1 and len(else_exprs) == 1:
            # return
            then_exprs = [n for n in node.body if not isinstance(n, ast.Return)]
            else_exprs = [n for n in node.orelse if not isinstance(n, ast.Return)]
            then_ret = then_exprs[0]
            else_ret = else_exprs[0]
        else:
            assert tgt_mapping, (
                f"then and else branchs do not have the same number "
                f"of assignments, we need more information to understand "
                f"which ones to return,"
                f"\n--\n{ast.unparse(node)}\n--\n{ast.dump(node, indent=2)}"
            )
            then_exprs, else_exprs = node.body, node.orelse
            then_rets, else_rets = [], []
            for t in tgt_mapping:
                then_e, else_e = tgt_mapping[t]
                then_rets.append(then_e or ast.Name(else_e.id, ctx=ast.Load()))
                else_rets.append(else_e or ast.Name(then_e.id, ctx=ast.Load()))
            then_ret = (
                then_rets[0] if len(then_rets) == 1 else ast.Tuple(then_rets, ctx=ast.Load())
            )
            else_ret = (
                else_rets[0] if len(else_rets) == 1 else ast.Tuple(else_rets, ctx=ast.Load())
            )

        # build local funcs
        then_def = ast.FunctionDef(
            name=then_name,
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg=v, annotation=None) for v in then_else_vars],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[],
            ),
            body=[*then_exprs, ast.Return(then_ret)],
            decorator_list=[],
            returns=None,
        )
        else_def = ast.FunctionDef(
            name=else_name,
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg=v, annotation=None) for v in then_else_vars],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[],
            ),
            body=[*else_exprs, ast.Return(else_ret)],
            decorator_list=[],
            returns=None,
        )
        # fix locations
        for n in (then_def, else_def):
            ast.copy_location(n, node)
            ast.fix_missing_locations(n)
            assert hasattr(n, "lineno")
        # wrapper call and assignment
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
        return then_def, else_def, call

    def _make_targets(self, node, then_assigns, else_assigns):
        tgt_mapping = {}
        for a, then_or_else in [
            *[(a, True) for a in then_assigns],
            *[(a, False) for a in else_assigns],
        ]:
            for t in a.targets:
                if isinstance(t, ast.Name):
                    if t.id not in tgt_mapping:
                        tgt_mapping[t.id] = (t, None) if then_or_else else (None, t)
                    else:
                        v = tgt_mapping[t.id]
                        tgt_mapping[t.id] = (t, v[1]) if then_or_else else (v[0], t)
                    continue

                assert isinstance(t, ast.Tuple) and all(
                    isinstance(_, ast.Name) for _ in t.elts
                ), (
                    f"Unexpected assignment. Not Supported."
                    f"\n--\n{ast.unparse(node)}\n--\n{ast.dump(node, indent=2)}"
                )
                for _t in t.elts:
                    if _t.id not in tgt_mapping:
                        tgt_mapping[_t.id] = (_t, None) if then_or_else else (None, _t)
                    else:
                        v = tgt_mapping[_t.id]
                        tgt_mapping[_t.id] = (_t, v[1]) if then_or_else else (v[0], _t)

        d = [(v[0] or v[1]) for k, v in sorted(dict(tgt_mapping).items())]
        tgt = d[0] if len(d) == 1 else ast.Tuple(d, ctx=ast.Load())
        return tgt, tgt_mapping

    def visit_If(self, node):
        # First recurse into subnodes
        node = self.generic_visit(node)

        has_then_return = any(isinstance(n, ast.Return) for n in node.body)
        has_else_return = any(isinstance(n, ast.Return) for n in node.orelse)
        ok = (has_then_return and has_else_return) or (
            not has_then_return and not has_else_return
        )
        assert ok, (
            f"Cannot mix return and assignment in a test\n--\n"
            f"{ast.unparse(node)}\n--\n{ast.dump(node, indent=2)}"
        )
        assert self.current_func_args is not None, (
            f"current_func_args is None\n--\n"
            f"{ast.unparse(node)}\n--\n{ast.dump(node, indent=2)}"
        )
        self.counter += 1

        if not has_then_return:
            # Case 1: simple assignment in both branches
            then_assigns = [n for n in node.body if isinstance(n, ast.Assign)]
            else_assigns = [n for n in node.orelse if isinstance(n, ast.Assign)]
            assert then_assigns or else_assigns, (
                f"Missing assignment"
                f"\n--\n{ast.unparse(node)}\n--\n{ast.dump(node, indent=2)}"
            )

            # the targets we need to export
            tgt, tgt_mapping = self._make_targets(node, then_assigns, else_assigns)

            then_def, else_def, call = self._rewrite_if(
                node, then_assigns, else_assigns, tgt_mapping=tgt_mapping
            )

            assign = ast.Assign(targets=[tgt], value=call)
            ast.copy_location(assign, node)
            ast.fix_missing_locations(assign)
            return [then_def, else_def, assign]

        # Case 2: return in both branches, we assume both branches return the same results.
        then_ret = node.body[-1]
        else_ret = node.orelse[-1]
        assert isinstance(then_ret, ast.Return), (
            f"return is not the last instruction of then branch"
            f"\n--\n{ast.unparse(node)}\n--\n{ast.dump(node, indent=2)}"
        )
        assert isinstance(else_ret, ast.Return), (
            f"return is not the last instruction of else branch"
            f"\n--\n{ast.unparse(node)}\n--\n{ast.dump(node, indent=2)}"
        )
        then_expr = then_ret.value
        else_expr = else_ret.value
        then_def, else_def, call = self._rewrite_if(node, [then_expr], [else_expr])
        ret = ast.Return(call)
        ast.copy_location(ret, node)
        ast.fix_missing_locations(ret)
        return [then_def, else_def, ret]

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

    @property
    def dump(self) -> str:
        """Returns the tree dumped as a string."""
        return ast.dump(self.tree, indent=2)

    def __repr__(self):
        "usual"
        return f"{self.__class__.__name__}({self.func})"


def transform_method(
    func: Callable, if_name="torch_cond", verbose: int = 0
) -> RewrittenMethod:
    """
    Returns a new function based on `func` where every test (if)
    is replaced by a call to :func:`torch.cond`.

    A test must return the same things if it returns something
    or assign something. It cannot return in one branch and assign
    in the other branch.

    .. warning:: room for improvment

        When it assigns a value to a constant,
        the current implementation does check which ones is really used
        after the test. The rewritten local functions returns every
        assigned variable. This could be reduced.

    :param func: method or function to rewrite
    :param if_name: function calling the test
    :param verbose: verbosity
    :return: rewritten method

    .. runpython::
        :showcode:

        import torch
        from onnx_diagnostic.torch_export_patches.patch_module import transform_method

        class Model(torch.nn.Module):
            def forward(self, x, y):
                if x.sum() > 0:
                    return x + y, x - y
                else:
                    return torch.abs(x) + y, torch.abs(x) - y

        x, y = torch.rand((3, 4)), torch.rand((3, 4))
        expected = Model()(x, y)

        rewritten = transform_method(Model.forward, verbose=10)
        print("-- code --")
        print(rewritten.code)

        print(" -- export --")
        Model.forward = rewritten.func

        DYN = torch.export.Dim.DYNAMIC
        ds = ({0: DYN, 1: DYN}, {0: DYN, 1: DYN})
        ep = torch.export.export(Model(), (x, y), dynamic_shapes=ds)
        print(ep)
    """
    # Retrieve source of the function
    src = inspect.getsource(func)
    if verbose:
        print(f"[transform_method] -- source -- {func}\n\n{src}\n\n[transform_method] --")
    # Parse into AST
    tree = ast.parse(textwrap.dedent(src))
    if verbose > 1:
        print(f"[transform_method] -- tree --\n\n{ast.dump(tree, indent=2)}")
    # Apply transformation
    transformer = RewriteControlFlow(if_name)
    new_tree = transformer.visit(tree)
    if verbose > 1:
        print(f"[transform_method] -- new tree --\n\n{ast.dump(tree, indent=2)}")
    ast.fix_missing_locations(new_tree)
    _settl(new_tree, 0)

    if verbose > 0:
        print(
            f"[transform_method] -- new code --\n\n"
            f"{ast.unparse(new_tree)}\n\n[transform_method] --"
        )
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
