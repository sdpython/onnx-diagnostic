import ast
import inspect
import types
import textwrap
from typing import Callable, Dict, List, Set, Optional

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
            if k in {"s", "n", "parent"}:
                continue
            if k[0] == "_":
                continue
            v = getattr(node, k)
            _settl(v, max(lineno, node.lineno), level=level + 1)
    return node


class RewriteControlFlow(ast.NodeTransformer):
    """
    The class rewrites tests with function :func:`torch.cond`.
    ``empty_tensor`` is a function returning an empty tensor,
    when a branch returns something the other branch does not.
    """

    def __init__(
        self,
        prefix: str = "branch_cond",
        skip_objects: Optional[Dict[str, object]] = None,
        args_names: Optional[Set[str]] = None,
    ):
        self.counter = 0
        self.current_func_args = None
        self.prefix = prefix
        self.skip_objects = skip_objects or {}
        self.args_names = args_names or set()
        self.local_variables = self.args_names.copy()

    def _check(
        self, cond: bool, node: "ast.Node", msg: str, cls: Optional[type[Exception]] = None
    ):
        if cls is not None:
            if not cond:
                raise cls(f"{msg}\n\n--\n{ast.unparse(node)}\n--\n{ast.dump(node, indent=2)}")
            return
        assert cond, f"{msg}\n\n--\n{ast.unparse(node)}\n--\n{ast.dump(node, indent=2)}"

    def visit_Name(self, node):
        node = self.generic_visit(node)
        if isinstance(node.ctx, ast.Store):
            self.local_variables.add(node.id)
        return node

    def visit_FunctionDef(self, node):
        # Capture argument names for branch functions
        old_args = self.current_func_args
        self.current_func_args = [arg.arg for arg in node.args.args]
        node.body = [self.visit(n) for n in node.body]
        self.current_func_args = old_args
        return node

    def _find_id(self, exprs: List["ast.Node"]) -> List[str]:
        vars = []
        for expr in exprs:
            for n in ast.walk(expr):
                if (
                    isinstance(n, ast.Name)
                    and isinstance(n.ctx, ast.Load)
                    and n.id not in self.skip_objects
                ):
                    vars.append(n.id)
        return sorted(set(vars))

    def _rewrite_if(
        self, node, then_exprs, else_exprs, tgt_mapping=None, known_local_variables=None
    ):
        assert known_local_variables is not None, "known_local_variables cannot be None"
        test_node = node.test
        drop = set()

        # extract free variables
        then_name = f"{self.prefix}_then_{self.counter}"
        else_name = f"{self.prefix}_else_{self.counter}"
        then_vars = self._find_id(then_exprs)
        else_vars = self._find_id(else_exprs)
        then_else_vars = set(_ for _ in [*then_vars, *else_vars] if _ in known_local_variables)
        then_ret, else_ret = None, None
        if tgt_mapping is None and len(then_exprs) == 1 and len(else_exprs) == 1:
            # return
            then_ret = then_exprs[0]
            else_ret = else_exprs[0]
            then_exprs = [n for n in node.body if not isinstance(n, ast.Return)]
            else_exprs = [n for n in node.orelse if not isinstance(n, ast.Return)]
        else:
            self._check(
                tgt_mapping,
                node,
                "then and else branches do not have the same number "
                "of assignments, we need more information to understand "
                "which ones to return",
            )
            drop = set()
            then_exprs, else_exprs = node.body, node.orelse
            then_rets, else_rets = [], []
            for t, then_else in sorted(tgt_mapping.items()):
                then_e, else_e = then_else
                if (then_e is None or else_e is None) and t not in then_else_vars:
                    # The variable is not used by one branch and it is not an input.
                    # Let's drop it.
                    drop.add(t)
                    continue
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
            func=ast.Attribute(
                value=ast.Name(id="torch", ctx=ast.Load()), attr="cond", ctx=ast.Load()
            ),
            args=[
                test_node,
                ast.Name(id=then_name, ctx=ast.Load()),
                ast.Name(id=else_name, ctx=ast.Load()),
                then_else_args_list,
            ],
            keywords=[],
        )
        return then_def, else_def, call, drop

    def _filter_target(self, node, tgt_mapping):
        """
        This function should reduce the number of elements to return
        by looking at the one used after the If statement.
        """
        return tgt_mapping

    def _make_targets(self, node, then_assigns, else_assigns):
        tgt_mapping = {}
        for a, then_or_else in [
            *[(a, True) for a in then_assigns],
            *[(a, False) for a in else_assigns],
        ]:
            for t in a.targets:
                if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Store):
                    if t.id not in tgt_mapping:
                        tgt_mapping[t.id] = (t, None) if then_or_else else (None, t)
                    else:
                        v = tgt_mapping[t.id]
                        tgt_mapping[t.id] = (t, v[1]) if then_or_else else (v[0], t)
                    continue

                self._check(
                    isinstance(t, ast.Tuple) and all(isinstance(_, ast.Name) for _ in t.elts),
                    node,
                    "Unexpected assignment. Not Supported.",
                )
                for _t in t.elts:
                    if not isinstance(_t, ast.Name) or not isinstance(_t.ctx, ast.Store):
                        continue
                    if _t.id not in tgt_mapping:
                        tgt_mapping[_t.id] = (_t, None) if then_or_else else (None, _t)
                    else:
                        v = tgt_mapping[_t.id]
                        tgt_mapping[_t.id] = (_t, v[1]) if then_or_else else (v[0], _t)

        tgt_mapping = self._filter_target(node, tgt_mapping)
        d = [(v[0] or v[1]) for k, v in sorted(dict(tgt_mapping).items())]
        tgt = d[0] if len(d) == 1 else ast.Tuple(d, ctx=ast.Load())
        return tgt, tgt_mapping

    def visit_If(self, node):
        # First recurse into subnodes
        known_local_variables = self.local_variables.copy()
        node = self.generic_visit(node)

        has_then_return = any(isinstance(n, ast.Return) for n in node.body)
        has_else_return = any(isinstance(n, ast.Return) for n in node.orelse)
        ok = (has_then_return and has_else_return) or (
            not has_then_return and not has_else_return
        )
        self._check(
            ok,
            node,
            "Cannot mix return and assignment in a test or a "
            "unique then branch with a return",
            NotImplementedError,
        )
        self._check(self.current_func_args is not None, node, "current_func_args is None")
        self.counter += 1

        if not has_then_return:
            # Case 1: simple assignment in both branches
            then_assigns = [n for n in node.body if isinstance(n, ast.Assign)]
            else_assigns = [n for n in node.orelse if isinstance(n, ast.Assign)]
            self._check(then_assigns or else_assigns, node, "Missing assignment")

            # the targets we need to export
            tgt, tgt_mapping = self._make_targets(node, then_assigns, else_assigns)

            then_def, else_def, call, dropped = self._rewrite_if(
                node,
                then_assigns,
                else_assigns,
                tgt_mapping=tgt_mapping,
                known_local_variables=known_local_variables,
            )
            if dropped and isinstance(tgt, ast.Tuple):
                tgt = ast.Tuple(
                    tuple(t for t in tgt.elts if t.id not in dropped), ctx=ast.Store()
                )

            assign = ast.Assign(targets=[tgt], value=call)
            ast.copy_location(assign, node)
            ast.fix_missing_locations(assign)
            return [then_def, else_def, assign]

        # Case 2: return in both branches, we assume both branches return the same results.
        then_ret = node.body[-1]
        else_ret = node.orelse[-1]
        self._check(
            isinstance(then_ret, ast.Return),
            node,
            "return is not the last instruction of then branch",
        )
        self._check(
            isinstance(else_ret, ast.Return),
            node,
            "return is not the last instruction of else branch",
        )
        then_expr = then_ret.value
        else_expr = else_ret.value
        then_def, else_def, call, dropped = self._rewrite_if(
            node, [then_expr], [else_expr], known_local_variables=known_local_variables
        )
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


class _AddParentTransformer(ast.NodeTransformer):
    parent = None

    def visit(self, node):
        node.parent = self.parent
        self.parent = node
        node = super().visit(node)
        if isinstance(node, ast.AST):
            self.parent = node.parent
        return node


def inplace_add_parent(tree: "ast.Node"):
    """Adds parents to an AST tree."""
    _AddParentTransformer().visit(tree)


def transform_method(
    func: Callable,
    prefix: str = "branch_cond",
    verbose: int = 0,
) -> RewrittenMethod:
    """
    Returns a new function based on `func` where every test (if)
    is replaced by a call to :func:`torch.cond`.

    A test must return the same things if it returns something
    or assign something. It cannot return in one branch and assign
    in the other branch.

    .. warning:: room for improvement

        When it assigns a value to a constant,
        the current implementation does check which ones is really used
        after the test. The rewritten local functions returns every
        assigned variable. This could be reduced.
        See method ``_filter_target``.

    :param func: method or function to rewrite
    :param prefix: prefix used to create the functions for the branches
    :param verbose: verbosity
    :return: rewritten method

    An example with **return**:

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

        rewritten = transform_method(Model.forward)
        print("-- code --")
        print(rewritten.code)

        print(" -- export --")
        Model.forward = rewritten.func

        DYN = torch.export.Dim.DYNAMIC
        ds = ({0: DYN, 1: DYN}, {0: DYN, 1: DYN})
        ep = torch.export.export(Model(), (x, y), dynamic_shapes=ds)
        print(ep)

    An example with **assignments**:

    .. runpython::
        :showcode:

        import torch
        from onnx_diagnostic.torch_export_patches.patch_module import transform_method

        class Model(torch.nn.Module):
            def forward(self, x, y):
                if x.sum() > 0:
                    w = x + y
                    z = x - y
                else:
                    w = torch.abs(x) + y
                    z = torch.abs(x) - y
                return w, z

        x, y = torch.rand((3, 4)), torch.rand((3, 4))
        expected = Model()(x, y)

        rewritten = transform_method(Model.forward)
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
    modules = {k: v for k, v in func.__globals__.items() if inspect.ismodule(v)}
    src = inspect.getsource(func)
    sig = inspect.signature(func)
    if verbose:
        print(f"[transform_method] -- source -- {func}\n\n{src}\n\n[transform_method] --")
    # Parse into AST
    tree = ast.parse(textwrap.dedent(src))
    if verbose > 1:
        print(f"[transform_method] -- tree --\n\n{ast.dump(tree, indent=2)}")
    # Apply transformation
    transformer = RewriteControlFlow(
        prefix=prefix,
        skip_objects=modules,
        args_names=set(sig.parameters),
    )
    inplace_add_parent(tree)
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
    exec(mod, globs, namespace)
    new_func = namespace.get(func.__name__)
    if not isinstance(new_func, types.FunctionType):
        raise RuntimeError("Transformed function not found")
    return RewrittenMethod(new_tree, new_func)
