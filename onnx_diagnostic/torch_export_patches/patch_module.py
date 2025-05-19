import ast
import copy
import contextlib
import inspect
import types
import textwrap
import sys
from typing import Callable, Dict, List, Set, Optional, Tuple, Union

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


class UsedVarsFinder(ast.NodeVisitor):
    """Finds used and defined local variables with a section."""

    def __init__(self):
        self.used = set()
        self.defined = set()

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            self.used.add(node.id)
        elif isinstance(node.ctx, ast.Store):
            self.defined.add(node.id)
        self.generic_visit(node)

    def visit_Global(self, node):
        pass

    def visit_Nonlocal(self, node):
        pass


class ShapeFinder(ast.NodeVisitor):
    """Finds <x> in the expression ``x.shape[0]``."""

    def __init__(self):
        self.found_shape = set()
        super().__init__()

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == "range" and len(node.args) == 1:
            n = node.args[0]
            if (
                isinstance(n, ast.Subscript)
                and isinstance(n.slice, ast.Constant)
                and isinstance(n.slice.value, int)
                and n.slice.value == 0
                and isinstance(n.value, ast.Attribute)
                and isinstance(n.value.value, ast.Name)
                and n.value.attr == "shape"
            ):
                self.found_shape.add(n.value.value.id)

        self.generic_visit(node)


class RewriteControlFlow(ast.NodeTransformer):
    """
    The class rewrites tests with function :func:`torch.cond`.
    ``empty_tensor`` is a function returning an empty tensor,
    when a branch returns something the other branch does not.

    :param prefix: prefix used for nested tests
    :param skip_objects: to skip variable names if included in that list
        such as modules
    :param args_names: defines the local variables
    """

    def __init__(
        self,
        prefix: str = "branch_cond",
        skip_objects: Optional[Dict[str, object]] = None,
        args_names: Optional[Set[str]] = None,
    ):
        self.counter_test = 0
        self.counter_loop = 0
        self.current_func_args = None
        self.prefix = prefix
        self.skip_objects = skip_objects or {}
        self.args_names = args_names or set()
        self.local_variables = self.args_names.copy()

    def generic_visit(self, node):
        return super().generic_visit(node)

    def _check(
        self, cond: bool, node: "ast.Node", msg: str, cls: Optional[type[Exception]] = None
    ):
        """
        Checks the condition is True, otherwise raises an exception with an error message
        including the parsed code.
        """
        if cls is not None:
            if not cond:
                smsg = msg if isinstance(msg, str) else msg()
                raise cls(f"{smsg}\n\n--\n{ast.unparse(node)}\n--\n{ast.dump(node, indent=2)}")
            return
        assert cond, (
            f"{msg if isinstance(msg, str) else msg()}\n\n--\n"
            f"{ast.unparse(node)}\n--\n{ast.dump(node, indent=2)}"
        )

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
                    # and isinstance(n.ctx, ast.Load)
                    and n.id not in self.skip_objects
                ):
                    vars.append(n.id)
        return sorted(set(vars))

    def _clone(self, name):
        assert isinstance(name, ast.Name), f"Unexpected type {type(name)} for name"
        return ast.Call(
            func=ast.Attribute(value=name, attr="clone", ctx=ast.Load()), args=[], keywords=[]
        )

    def _rewrite_if(
        self, node, then_exprs, else_exprs, tgt_mapping=None, known_local_variables=None
    ):
        assert known_local_variables is not None, "known_local_variables cannot be None"
        test_node = node.test
        drop = set()

        # extract free variables
        then_name = f"{self.prefix}_then_{self.counter_test}"
        else_name = f"{self.prefix}_else_{self.counter_test}"
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
            is_tuple_or_list = (
                isinstance(then_ret, (ast.Tuple, ast.List)),
                isinstance(else_ret, (ast.Tuple, ast.List)),
            )
            assert len(set(is_tuple_or_list)) == 1, (
                f"is_tuple_or_list={is_tuple_or_list}, inconsistencies return "
                f"then value={then_ret}, "
                f"else value={else_ret}"
            )
            if is_tuple_or_list[0]:
                assert len(then_ret.elts) == len(else_ret.elts), (
                    f"Unexpected number of elements on both branches, "
                    f"then:{then_ret.elts}, else:{else_ret.elts}"
                )
                n_returned_values = len(then_ret.elts)
            else:
                n_returned_values = 0
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
                self._clone(then_rets[0])
                if len(then_rets) == 1
                else ast.Tuple([self._clone(r) for r in then_rets], ctx=ast.Load())
            )
            else_ret = (
                self._clone(else_rets[0])
                if len(else_rets) == 1
                else ast.Tuple([self._clone(r) for r in else_rets], ctx=ast.Load())
            )
            n_returned_values = len(then_rets) if len(then_rets) > 1 else 0

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
        return then_def, else_def, call, drop, n_returned_values

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
        self.counter_test += 1

        if not has_then_return:
            # Case 1: simple assignment in both branches
            then_assigns = [n for n in node.body if isinstance(n, ast.Assign)]
            else_assigns = [n for n in node.orelse if isinstance(n, ast.Assign)]
            self._check(then_assigns or else_assigns, node, "Missing assignment")

            # the targets we need to export
            tgt, tgt_mapping = self._make_targets(node, then_assigns, else_assigns)

            then_def, else_def, call, dropped, n_returned_values = self._rewrite_if(
                node,
                then_assigns,
                else_assigns,
                tgt_mapping=tgt_mapping,
                known_local_variables=known_local_variables,
            )
            if dropped and isinstance(tgt, ast.Tuple):
                tgt_elts = tuple(t for t in tgt.elts if t.id not in dropped)
            elif isinstance(tgt, ast.Tuple):
                tgt_elts = tuple(t for t in tgt.elts if t.id not in dropped)
            else:
                tgt_elts = [tgt]

            if n_returned_values == 0:
                assert len(tgt_elts) == 1, (
                    f"Inconsistencies between n_returned_values={n_returned_values}, "
                    f"dropped={dropped}, tgt.elts={tgt.elts}, tgt_elts={tgt_elts}"
                )
                tgt = tgt_elts[0]
            else:
                assert n_returned_values == len(tgt_elts), (
                    f"Inconsistencies between n_returned_values={n_returned_values}, "
                    f"dropped={dropped}, tgt.elts={tgt.elts}, tgt_elts={tgt_elts}"
                )
                tgt = ast.Tuple(tgt_elts, ctx=ast.Store())

            added = {tgt.id} if isinstance(tgt, ast.Name) else set(t.id for t in tgt.elts)
            assign = ast.Assign(targets=[tgt], value=call)
            ast.copy_location(assign, node)
            ast.fix_missing_locations(assign)
            self.local_variables = known_local_variables | added
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
        then_def, else_def, call, dropped, n_returned_values = self._rewrite_if(
            node, [then_expr], [else_expr], known_local_variables=known_local_variables
        )
        ret = ast.Return(call)
        ast.copy_location(ret, node)
        ast.fix_missing_locations(ret)
        return [then_def, else_def, ret]

    def _find_loop_vars(self, node):
        assert isinstance(node, ast.For), f"Unexpected type {type(node)} for node"
        finder = ShapeFinder()
        finder.visit(node.iter)
        scan_shape_vars = finder.found_shape
        scan_vars = set()

        finder = UsedVarsFinder()
        for stmt in node.body:
            finder.visit(stmt)

        assigned_in_body = set()
        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                for tgt in stmt.targets:
                    if isinstance(tgt, ast.Name) and isinstance(tgt.value.ctx, ast.Store):
                        assigned_in_body |= {tgt.value.id}

        extra_defined = set()
        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                for tgt in stmt.targets:
                    if isinstance(tgt, ast.Subscript):
                        # It means the target existed before.
                        if (
                            isinstance(tgt.value, ast.Name)
                            and tgt.value.id not in assigned_in_body
                        ):
                            extra_defined.add(tgt.value.id)

        loop_vars = set()
        if isinstance(node.target, ast.Name):
            loop_vars.add(node.target.id)
        elif isinstance(node.target, (ast.Tuple, ast.List)):
            loop_vars |= {elt.id for elt in node.target.elts if isinstance(elt, ast.Name)}

        output_vars = finder.defined | assigned_in_body
        input_vars = (
            finder.used
            - finder.defined
            - loop_vars
            - scan_shape_vars
            - scan_vars
            - output_vars
            - assigned_in_body
            - extra_defined
        )
        return dict(
            init=sorted(extra_defined),
            loop=sorted(loop_vars),
            scan_shape=sorted(scan_shape_vars),
            scan=sorted(scan_vars),
            input=sorted(input_vars),
            output=sorted(output_vars),
        )

    def visit_For(self, node):
        # For nested loops.
        self.generic_visit(node)
        # look for variables, loop, inputs and outputs of the body
        vars = self._find_loop_vars(node)
        init_vars, loop_vars, scan_shape_vars, scan_vars, input_vars, output_vars = [
            vars[k] for k in ["init", "loop", "scan_shape", "scan", "input", "output"]
        ]
        self._check(
            len(scan_shape_vars) == len(loop_vars),
            node,
            lambda: (
                f"Inconsistencies between loop_vars={loop_vars} "
                f"and scan_shape_vars={scan_shape_vars}"
            ),
        )
        self._check(
            len(scan_shape_vars) in {0, 1},
            node,
            lambda: f"Inconsistencies with scan_shape_vars={scan_shape_vars}",
        )
        self._check(
            (len(scan_shape_vars) == 0 or len(scan_vars) == 0)
            and (scan_shape_vars or scan_vars),
            node,
            lambda: (
                f"Inconsistencies between scan_vars={scan_vars} "
                f"and scan_shape_vars={scan_shape_vars}"
            ),
        )

        # creates the function
        func_name = f"loop_body_{self.counter_loop}"
        self.counter_loop += 1
        func_def = ast.FunctionDef(
            name=func_name,
            args=ast.arguments(
                posonlyargs=[],
                args=[
                    ast.arg(arg=v)
                    for v in [
                        *init_vars,
                        *loop_vars,
                        *scan_vars,
                        *scan_shape_vars,
                        *input_vars,
                    ]
                ],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[],
            ),
            body=[
                *[
                    ast.Assign(
                        targets=[ast.Name(id=i, ctx=ast.Load())],
                        value=[
                            ast.Call(
                                func=ast.Attribute(
                                    value=ast.Name(id=i, ctx=ast.Load()),
                                    attr="clone",
                                    ctx=ast.Load(),
                                ),
                                args=[],
                                keywords=[],
                                ctx=ast.Load(),
                            )
                        ],
                    )
                    for i in init_vars
                ],
                *node.body,
                ast.Return(
                    value=ast.List(
                        [
                            ast.Name(id=v, ctx=ast.Load())
                            for v in [*init_vars, *loop_vars, *output_vars]
                        ],
                        ctx=ast.Load(),
                    )
                ),
            ],
            decorator_list=[],
            ctx=ast.Store(),
        )

        # final rewriting
        call = ast.Call(
            func=(
                ast.Attribute(
                    value=ast.Attribute(
                        value=ast.Attribute(
                            value=ast.Name(id="torch", ctx=ast.Load()),
                            attr="ops",
                            ctx=ast.Load(),
                        ),
                        attr="higher_order",
                        ctx=ast.Load(),
                    ),
                    attr="scan",
                    ctx=ast.Load(),
                )
            ),
            args=[
                ast.Name(id=func_name, ctx=ast.Load()),
                ast.List(
                    elts=[ast.Name(id=v, ctx=ast.Load()) for v in init_vars], ctx=ast.Store()
                ),
                ast.List(
                    elts=[
                        *[
                            ast.Call(
                                ast.Attribute(
                                    value=ast.Name(id="torch", ctx=ast.Load()),
                                    attr="arange",
                                    ctx=ast.Load(),
                                ),
                                args=[
                                    ast.Subscript(
                                        value=ast.Attribute(
                                            value=ast.Name(id=v, ctx=ast.Load()),
                                            attr="shape",
                                            ctx=ast.Load(),
                                        ),
                                        slice=ast.Constant(value=0, ctx=ast.Load()),
                                        ctx=ast.Load(),
                                    ),
                                ],
                                keywords=[
                                    ast.keyword(
                                        arg="dtype",
                                        value=ast.Attribute(
                                            value=ast.Name(id="torch", ctx=ast.Load()),
                                            attr="int64",
                                            ctx=ast.Load(),
                                        ),
                                    )
                                ],
                                ctx=ast.Load(),
                            )
                            for v in scan_shape_vars
                        ],
                        *[ast.Name(id=v, ctx=ast.Load()) for v in scan_vars],
                    ],
                    ctx=ast.Store(),
                ),
                ast.List(
                    elts=[
                        ast.Name(id=v, ctx=ast.Load()) for v in [*scan_shape_vars, *input_vars]
                    ],
                    ctx=ast.Store(),
                ),
            ],
            keywords=[],
            ctx=ast.Load(),
        )
        target = ast.Tuple(
            [ast.Name(id=v, ctx=ast.Store()) for v in [*init_vars, *loop_vars, *output_vars]],
            ctx=ast.Store(),
        )
        assign = ast.Assign(targets=[target], value=call)
        return [func_def, assign]


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


class _SelectiveAssignNormalizer(ast.NodeTransformer):
    def visit_If(self, node):
        self.generic_visit(node)
        node.body = [self._transform_if_needed(stmt) for stmt in node.body]
        node.orelse = [self._transform_if_needed(stmt) for stmt in node.orelse]
        return node

    def _transform_if_needed(self, stmt):
        if isinstance(stmt, ast.AugAssign):
            return ast.Assign(
                targets=[stmt.target],
                value=ast.BinOp(left=copy.deepcopy(stmt.target), op=stmt.op, right=stmt.value),
            )
        if isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
            return ast.Assign(targets=[stmt.target], value=stmt.value)
        return self.visit(stmt)


def inplace_add_parent(tree: "ast.Node"):
    """Adds parents to an AST tree."""
    _AddParentTransformer().visit(tree)


def normalize_assignment_in_test(tree: "ast.Node"):
    """Split AugAssign into BinOp and Assign to simplify whatever comes after."""
    _SelectiveAssignNormalizer().visit(tree)


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
        :process:
        :store_in_file: test_example_transform_method_1.py

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
        :process:
        :store_in_file: test_example_transform_method_2.py

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
    normalize_assignment_in_test(tree)
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


@contextlib.contextmanager
def torch_export_rewrite(
    rewrite: Optional[List[Union[Tuple[type, str], Callable]]] = None,
    dump_rewriting: Optional[str] = None,
    verbose: int = 0,
):
    """
    Automatically rewrite the methods given in `rewrite` to export
    control flows (test and loops).

    :param rewrite: methods of functions to rewrite, if not empty, the function may try
        to discover them, a method is defined by its class (a type) and its name
        if the class is local, by itself otherwise
    :param dump_rewriting: dumps rewriting information in file beginning with that prefix
    :param verbose: verbosity, up to 10, 10 shows the rewritten code,
        ``verbose=1`` shows the rewritten function,
        ``verbose=2`` shows the rewritten code as well

    Example:

    .. code-block:: python

        class Model(torch.nn.Module):
            def forward(self, x, y):
                if x.sum() > 0:
                    return x + y
                else:
                    return torch.abs(x) + y + 1

        model = Model()
        x, y = torch.rand((4, 5)), torch.rand((4, 5))
        DYN = torch.export.Dim.DYNAMIC
        ds = ({0: DYN, 1: DYN}, {0: DYN, 1: DYN})
        with torch_export_rewrite(rewrite=[(Model, "forward")]):
            ep = torch.export.export(model, (x, y), dynamic_shapes=ds)

    If the method to rewrite is not local, then the following can be used:

    .. code-block:: python

        with torch_export_rewrite(rewrite=[Model.forward]):
            ep = torch.export.export(model, (x, y), dynamic_shapes=ds)

    Functions (if not local) can also be rewritten:

    .. code-block:: python

        def outside(x, y):
            if x.sum() > 0:
                return x + y
            else:
                return torch.abs(x) + y + 1

        class Model(torch.nn.Module):
            def forward(self, x, y):
                return outside(x, y)

        model = Model()
        x, y = torch.rand((4, 5)), torch.rand((4, 5))
        DYN = torch.export.Dim.DYNAMIC
        ds = ({0: DYN, 1: DYN}, {0: DYN, 1: DYN})
        with torch_export_rewrite(rewrite=[outside]):
            ep = torch.export.export(model, (x, y), dynamic_shapes=ds)
    """
    assert rewrite, "rewrite is empty, automated discovery is not implemented yet"
    keep = {}
    for me in rewrite:
        if isinstance(me, tuple):
            assert len(me) == 2, f"Unexpected value for a rewritten method or function {me}"
            cls, name = me
            to_rewrite = getattr(cls, name)
            kind = "method"
        else:
            name = me.__qualname__
            spl = name.split(".")
            if len(spl) == 1:
                # This a function
                module = me.__module__
                if module in me.__globals__:
                    mod = me.__globals__[module]
                else:
                    assert module in sys.modules, (
                        f"Cannot find module name {module!r} in sys.modules or "
                        f"__globals__={sorted(me.__globals__)}"
                    )
                    mod = sys.modules[module]
                cls_name = module
                cls = mod
                name = name
                to_rewrite = me
                kind = "function"
            else:
                kind = "method"
                # This is a method
                assert len(spl) >= 2, (
                    f"{me} is not method, its name {name!r} does not contain a class name, "
                    f"dir(me)={dir(me)}"
                )
                cls_name = spl[-2]
                assert cls_name in me.__globals__, (
                    f"Class name {cls_name!r} from method {name!r} "
                    f"could not be found in set(me.__globals__)={sorted(me.__globals__)}"
                )
                cls = me.__globals__[cls_name]
                name = me.__name__
                to_rewrite = me
                assert hasattr(
                    cls, name
                ), f"Method {name!r} inferred form {me} was not found in class {cls}."
        assert (cls, name) not in keep, f"{kind} {me} cannot be rewritten twice."
        if verbose:
            print(f"[torch_export_rewrite] rewrites {kind} {cls.__name__}.{name}")
        keep[cls, name] = to_rewrite
        if dump_rewriting:
            filename = f"{dump_rewriting}.{kind}.{cls_name}.{name}.original.py"
            if verbose:
                print(f"[torch_export_rewrite] dump original code in {filename!r}")
            with open(filename, "w") as f:
                f.write(inspect.getsource(to_rewrite))
        rewr = transform_method(to_rewrite, verbose=max(verbose - 1, 0))
        if dump_rewriting:
            filename = f"{dump_rewriting}.{kind}.{cls_name}.{name}.rewritten.py"
            if verbose:
                print(f"[torch_export_rewrite] dump rewritten code in {filename!r}")
            with open(filename, "w") as f:
                f.write(rewr.code)
        setattr(cls, name, rewr.func)

    try:
        yield
    finally:
        for (cls, name), me in keep.items():
            if verbose:
                print(f"[torch_export_rewrite] restored {kind} {cls.__name__}.{name}")
            setattr(cls, name, me)
