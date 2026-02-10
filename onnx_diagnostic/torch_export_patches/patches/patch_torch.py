import functools
import inspect
import operator
import os
import traceback
from functools import reduce
from typing import Any, Callable, cast, Dict, List, Optional, Sequence, Tuple, Union
import sympy
import torch
import torch.export._trace
from torch._subclasses.fake_tensor import FakeTensorMode


def retrieve_stacktrace():
    """Retrieves and prints the current stack trace, avoids every torch file."""
    rows = []
    stack_frames = traceback.extract_stack()
    for frame in stack_frames:
        filename, lineno, function_name, code_line = frame
        if "/torch/" in filename:
            continue
        rows.append(f"File: {filename}, Line {lineno}, in {function_name}")
        if code_line:
            rows.append(f"    {code_line}")
    return "\n".join(rows)


def _catch_produce_guards_and_solve_constraints(
    previous_function: Callable,
    fake_mode: FakeTensorMode,
    gm: torch.fx.GraphModule,
    dynamic_shapes: Union[Dict[str, Any], Tuple[Any], List[Any], None],
    equalities_inputs: "EqualityConstraint",  # noqa: F821
    original_signature: inspect.Signature,
    verbose: int = 0,
    **kwargs,
):
    try:
        return previous_function(
            fake_mode=fake_mode,
            gm=gm,
            dynamic_shapes=dynamic_shapes,
            equalities_inputs=equalities_inputs,
            original_signature=original_signature,
            **kwargs,
        )
    except Exception as e:
        if not int(os.environ.get("SKIP_SOLVE_CONSTRAINTS", "1")):
            raise
        if verbose:
            print(
                f"[_catch_produce_guards_and_solve_constraints] ERROR: "
                f"produce_guards_and_solve_constraints failed, "
                f"use SKIP_SOLVE_CONSTRAINTS=0 to avoid skipping\n"
                f"fake_mode={fake_mode}\n"
                f"dynamic_shapes={dynamic_shapes}\n"
                f"equalities_inputs={equalities_inputs}\n"
                f"original_signature={original_signature}\n"
                f"kwargs={kwargs}\n"
                f"exc={e}\ngm={gm}"
            )
        torch._dynamo.reset()


def patched__check_input_constraints_for_graph(
    previous_function: Callable,
    input_placeholders: list[torch.fx.Node],
    flat_args_with_path,
    range_constraints,
    verbose: int = 0,
) -> None:
    try:
        # PATCHED: catches exception and prints out the information instead of
        # stopping the conversion.
        return previous_function(input_placeholders, flat_args_with_path, range_constraints)
    except Exception as e:
        if not int(os.environ.get("SKIP_SOLVE_CONSTRAINTS", "1")):
            raise
        if verbose:
            print(
                f"[_check_input_constraints_for_graph] ERROR: "
                f"_check_input_constraints_for_graph failed, "
                f"use SKIP_SOLVE_CONSTRAINTS=0 to avoid skipping\n"
                f"input_placeholders={input_placeholders}\n"
                f"range_constraints={range_constraints}\n"
                f"exc={e}"
            )
        torch._dynamo.reset()


def patched_infer_size(a, b):
    """Patches ``torch._subclasses.fake_impls.infer_size``."""
    from torch.fx.experimental.symbolic_shapes import guard_or_false

    dimsA = len(a)
    dimsB = len(b)
    ndim = max(dimsA, dimsB)
    expandedSizes = [0] * ndim
    for i in range(ndim - 1, -1, -1):
        offset = ndim - 1 - i
        dimA = dimsA - 1 - offset
        dimB = dimsB - 1 - offset
        sizeA = a[dimA] if dimA >= 0 else 1
        sizeB = b[dimB] if dimB >= 0 else 1

        # NB: It is very important to test for broadcasting, before testing
        # sizeA == sizeB.  This is because the broadcasting tests are likely
        # to be statically known (in particular, if sizeA/sizeB is unbacked
        # but size-like, we will unsoundly assume they never equal 1), but
        # the sizeA == sizeB test may not be statically known.  However, once
        # we have established that no broadcasting is happening, the
        # sizeA == sizeB is now expect_true and we can defer it as a runtime
        # assert (this works because Python will return the terminal
        # expression of an or statement as-is, without bool()'ing it; if this
        # were not the case, we'd need to write this using torch.sym_or() or
        # something like that).
        try:
            b1 = guard_or_false(sizeA == 1)
        except torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode:
            b1 = False
        try:
            b2 = guard_or_false(sizeB == 1)
        except torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode:
            b2 = False
        try:
            b3 = guard_or_false(sizeA == sizeB)
        except torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode:
            b3 = False
        if b1 or b2 or b3:
            expandedSizes[i] = sizeB if guard_or_false(sizeA == 1) else sizeA
        else:
            # PATCHED: generic case, the dimension is known, no need to assert
            expandedSizes[i] = torch.sym_max(sizeA, sizeB)
    return tuple(expandedSizes)


def patched__get_range_constraints(
    mod: torch.nn.Module,
    export_artifact: torch.export._trace.ExportArtifact,
    args,
    kwargs,
    dynamic_shapes,
):
    """
    Patches ``torch.export._trace._get_range_constraints``.
    See PR `#174593 <https://github.com/pytorch/pytorch/pull/174593>`_.
    """
    gm: torch.fx.GraphModule = export_artifact.aten.gm
    export_graph_signature: torch.export.graph_signature.ExportGraphSignature = (
        export_artifact.aten.sig
    )
    fake_mode: FakeTensorMode = export_artifact.fake_mode
    num_lifted = next(
        (
            i
            for i, s in enumerate(export_graph_signature.input_specs)
            if s.kind == torch.export.graph_signature.InputKind.USER_INPUT
        ),
        len(export_graph_signature.input_specs),
    )

    combined_args = torch.export._trace._combine_args(mod, args, kwargs)

    # _combine_args does not preserve the order.
    if isinstance(combined_args, dict):
        input_names = [
            s.arg.name
            for s in export_graph_signature.input_specs
            if s.kind == torch.export.graph_signature.InputKind.USER_INPUT
        ]
        new_args = {}
        for k in input_names:
            if k in combined_args:
                new_args[k] = combined_args[k]
        for k in combined_args:
            if k not in new_args:
                new_args[k] = combined_args[k]
        combined_args = new_args

    range_constraints = torch._export.non_strict_utils.make_constraints(
        fake_mode, gm, combined_args, dynamic_shapes, num_lifted
    )
    return range_constraints


def patched__broadcast_shapes(*_shapes):
    """Patches ``torch._refs._broadcast_shapes``."""
    from functools import reduce
    from torch._prims_common import IntLike
    from torch.fx.experimental.symbolic_shapes import (
        guard_or_false,
        is_nested_int,
    )

    shapes = tuple(
        (x,) if isinstance(x, IntLike) else x for x in filter(lambda x: x is not None, _shapes)
    )

    # Short-circuits on no input
    if len(shapes) == 0:
        return None

    for shape in shapes:
        if not isinstance(shape, Sequence):
            raise RuntimeError(
                "Input shapes should be of type ints, a tuple of ints, "
                "or a list of ints, got ",
                shape,
            )

    # Computes common shape
    common_shape = [1] * reduce(max, (len(shape) for shape in shapes))
    for _arg_idx, shape in enumerate(shapes):
        for idx in range(-1, -1 - len(shape), -1):
            if is_nested_int(shape[idx]):
                # Broadcasting is allowed for (j0, 1) or (j0, j0);
                # not (j0, j1), (j0, 5), etc.
                if is_nested_int(common_shape[idx]) and guard_or_false(
                    shape[idx] == common_shape[idx]
                ):
                    continue
            else:
                if guard_or_false(shape[idx] == common_shape[idx]):
                    continue
            # PATCHED: two cases, if == for sure, no broadcast,
            # otherwise maybe broadcast with max(dimensions)
            if guard_or_false(common_shape[idx] != 1):
                pass
            elif guard_or_false(common_shape[idx] == 1) or guard_or_false(shape[idx] != 1):
                if shape[idx] < 0:
                    raise ValueError(
                        "Attempting to broadcast a dimension with negative length!"
                    )
                common_shape[idx] = shape[idx]
            else:
                common_shape[idx] = torch.sym_max(common_shape[idx], shape[idx])

    return common_shape


def value_ranges_is_positive(value_ranges: torch.utils._sympy.value_ranges.ValueRanges):
    """Tells if an interval is equivalent to a positive or null integer."""
    return value_ranges.lower == 0 and value_ranges.upper > 4623372036854775806


class patched_ShapeEnv:

    def _check_frozen(
        self, expr: "sympy.Basic", concrete_val: "sympy.Basic"  # noqa: F821
    ) -> None:
        if self.frozen:
            self.counter["ignored_backward_guard"] += 1
            # PATCHED: raised an exception instead of logging.
            import transformers

            raise AssertionError(
                f"[patched_ShapeEnv] Ignored guard {expr} == {concrete_val}, "
                f"this could result in accuracy problems, transformers.__version__="
                f"{transformers.__version__!r}"
            )

    def _set_replacement(
        self, a: "sympy.Symbol", tgt: "sympy.Expr", msg: str  # noqa: F821
    ) -> None:
        """
        Adds or updates a replacement for a symbol.
        Use this instead of `self.replacements[a] = tgt`.
        """
        if tgt == self.replacements.get(a, None):
            return

        if a in tgt.free_symbols:
            return

        import sympy
        from torch._logging import structured
        from torch.utils._traceback import CapturedTraceback
        from torch._logging import trace_structured
        from torch._guards import TracingContext
        from torch.utils._sympy.functions import FloorToInt, CeilToInt
        from torch.utils._sympy.solve import try_solve
        from torch.fx.experimental.symbolic_shapes import (
            _is_supported_equivalence,
            ValueRanges,
        )

        # Precondition: a == tgt
        assert isinstance(a, sympy.Symbol)

        if (
            getattr(self, "allow_complex_guards_as_runtime_asserts", False)
            or getattr(self, "prefer_deferred_runtime_asserts_over_guards", False)
        ) and not _is_supported_equivalence(tgt):
            # continuing leads to placeholder shapes
            # having complex expressions that we can't resolve
            return

        # Handles nested tensor symbolic variables which don't have
        # var_to_range bounds
        tgt_bound = None
        if a in self.var_to_range:
            src_bound = self.var_to_range[a]

            # First, refine the value range of a based on the computed value range
            # of tgt.  This is always OK to do, even if we decide not to do the
            # substitution in the end.  This might be a no-op, if a already has
            # a tighter bound
            tgt_bound = self.bound_sympy(tgt)
            self._update_var_to_range(a, tgt_bound)

            # Next, check if we can update the range of free symbols in tgt
            # based on the range in a. But only do it if:
            #  - the source bound non-trivially improves over what we get out of
            #    the existing bounds.
            #  - the replacement is univariate and we can invert the tgt expression
            if not tgt_bound.issubset(src_bound) and len(tgt.free_symbols) == 1:
                b = next(iter(tgt.free_symbols))
                # Try to invert the equality
                r = try_solve(sympy.Eq(a, tgt), b, floordiv_inequality=False)
                if r is not None:
                    self.log.debug(
                        "set_replacement: solve for %s in %s == %s gives %s",
                        b,
                        a,
                        tgt,
                        r,
                    )
                    # The solution here can be non-integral, for example, if
                    # we have s0 = 2*s1, then s1 = s0/2.  What we would like
                    # to do is calculated the bounds in arbitrary precision,
                    # and then requantize the bound to integers when we are
                    # done.
                    rat_b_bound = self.bound_sympy(r[1])
                    b_bound = ValueRanges(
                        CeilToInt(rat_b_bound.lower), FloorToInt(rat_b_bound.upper)
                    )
                    self._update_var_to_range(b, b_bound, self.var_to_range_sloc[a])
                    tgt_bound = self.bound_sympy(tgt)
                    assert (
                        value_ranges_is_positive(tgt_bound)
                        and value_ranges_is_positive(src_bound)
                    ) or tgt_bound.issubset(
                        src_bound
                    ), f"{tgt_bound=} not a subset of {src_bound=}"

            # TODO: Should we propagate size-like-ness?
            #
            # Pros: if u0 is size-like, intuitively u0 == u1 should cause u1
            # to become size-like.
            #
            # Cons: if u0 is size-like, what about u0 - 1 == u1?  You CAN'T
            # propagate in this case, because what if u0 == 0, then u1 is negative
            # and clearly isn't a size.  So, at minimum, any f(x) whose value
            # range isn't [0, inf] given x in [0, inf] cannot propagate
            # size-like-ness.  But there are many situations where you could
            # imagine u1 is going to be size-like and actually you just didn't
            # have a refined enough value range on u0.  Since even innocuous
            # looking arithmetic operations can destroy size-like-ness, it's
            # best to not propagate it at all and force the user to annotate it
            # as necessary.
            #
            # Compromise: we preserve size-like-ness only for exact equality
            # and nothing else.
            if a in self.size_like and isinstance(tgt, sympy.Symbol):
                self.size_like.add(tgt)
            elif isinstance(tgt, sympy.Symbol) and tgt in self.size_like:
                self.size_like.add(a)

            # Now, decide if we will do the substitution.
            #
            #  - If the source has a non-trivial range, only substitute if
            #    we preserve this range.  Note that we may have propagated
            #    the src_range to free variables in tgt when tgt is univariate
            #    and we could find an inverse, which helps us achieve this.
            #    This ensures we never "forget" about user defined ranges,
            #    even if they end up being defined on composite formulas
            #    like s0 + s1.
            #
            #  - If the variable is unbacked, only substitute if the substitution
            #    would preserve the bounds also under size-like-ness conditions.

            if not tgt_bound.issubset(src_bound):
                self.log.debug(
                    "skipped set_replacement %s = %s (%s) [%s not subset of %s]",
                    a,
                    tgt,
                    msg,
                    tgt_bound,
                    src_bound,
                )
                return
            elif a in self.size_like:
                tgt_bound_so = self.bound_sympy(tgt, size_oblivious=True)
                src_bound_so = self.bound_sympy(a, size_oblivious=True)
                if not tgt_bound_so.issubset(src_bound_so):
                    self.log.debug(
                        "skipped set_replacement %s = %s (%s) "
                        "[%s not subset of %s (size-oblivious conditions)]",
                        a,
                        tgt,
                        msg,
                        tgt_bound_so,
                        src_bound_so,
                    )
                    return

        if isinstance(tgt, (sympy.Integer, sympy.Float)):
            # specializing to a constant, which is likely unexpected (unless
            # you specified dynamic=True)

            user_tb = TracingContext.extract_stack()
            trace_structured(
                "symbolic_shape_specialization",
                metadata_fn=lambda: {
                    "symbol": repr(a),
                    "sources": [s.name() for s in self.var_to_sources.get(a, [])],
                    "value": repr(tgt),
                    "reason": msg,
                    "stack": structured.from_traceback(
                        CapturedTraceback.extract(skip=1).summary()
                    ),
                    "user_stack": (structured.from_traceback(user_tb) if user_tb else None),
                },
            )

            for source in self.var_to_sources.get(a, []):
                if user_tb:
                    self.specialization_stacks[source] = user_tb

            # PATCHED: removed lines
            # if config.print_specializations:
            #    self.log.warning(
            #         "Specializing %s to %s", self.var_to_sources[a][0].name(), tgt
            #     )
            #     self.log.debug("SPECIALIZATION", stack_info=True)
        # PATCHED: replaces logging by raising an exception
        assert msg != "range_refined_to_singleton", (
            f"patched_ShapeEnv: A dynamic dimension becomes static! "
            f"a={a!r}, tgt={tgt!r}, msg={msg!r}, tgt_bound={tgt_bound}"
        )
        # log.info("set_replacement %s = %s (%s) %s", a, tgt, msg, tgt_bound)
        self.replacements[a] = tgt
        # NB: the replacement may get refined, but the user will find the
        # FIRST one most useful (TODO: Maybe we could consider tracking all of
        # them)
        if a not in self.replacements_slocs:
            self.replacements_slocs[a] = self._get_sloc()
        self._update_version_counter()

        # When specializing 'a == tgt', the equality should be also conveyed to
        # Z3, in case an expression uses 'a'.
        self._add_target_expr(sympy.Eq(a, tgt, evaluate=False))

    def _log_guard(
        self, prefix: str, g: "SympyBoolean", forcing_spec: bool  # noqa: F821
    ) -> None:
        self._log_guard_remember(prefix=prefix, g=g, forcing_spec=forcing_spec)
        # PATCHED: removed
        # It happens too often to be relevant.
        # sloc, _maybe_extra_debug = self._get_stack_summary(True)
        # warnings.warn(
        #     f"A guard was added, prefix={prefix!r}, g={g!r}, "
        #     f"forcing_spec={forcing_spec}, location=\n{sloc}\n"
        #    f"--stack trace--\n{retrieve_stacktrace()}",
        #     RuntimeWarning,
        #     stacklevel=0,
        # )

    def _evaluate_expr(
        self,
        orig_expr: "sympy.Basic",  # noqa: F821
        hint: Optional[Union[bool, int, float]] = None,
        fx_node: Optional[torch.fx.Node] = None,
        size_oblivious: bool = False,
        fallback_value: Optional[bool] = None,
        *,
        forcing_spec: bool = False,
    ) -> "sympy.Basic":  # noqa: F821
        # TODO: split conjunctions and evaluate them separately
        import sympy
        from torch.fx.experimental import _config as config
        from torch.fx.experimental.symbolic_shapes import (
            SympyBoolean,
            log,
            SymT,
            symbol_is_type,
        )
        from torch._guards import ShapeGuard

        if isinstance(
            orig_expr,
            (sympy.logic.boolalg.BooleanTrue, sympy.logic.boolalg.BooleanFalse),
        ):
            return orig_expr

        # Don't track this one. (Because this cache is inside this function the
        # cache only lasts for the invocation of this function call)
        @functools.cache
        def compute_concrete_val() -> sympy.Basic:
            if hint is None:
                # This is only ever called for expressions WITHOUT unbacked
                # symbols
                r = self.size_hint(orig_expr)
                assert r is not None
                return r
            else:
                return sympy.sympify(hint)

        concrete_val: Optional[sympy.Basic]

        # Check if:
        #   1. 'translation_validation' is set
        #   2. the corresponding 'fx_node' is not 'None'
        #   3. the guard should not be suppressed
        #   4. the guard doesn't contain backed symfloat symbols
        #      since z3 can't handle floats
        #   5. fallback_value is none.
        # If all of the above check, we create an FX node representing the
        # actual expression to be guarded.
        node = None
        fresh = False
        if (
            self._translation_validation_enabled
            and fx_node is not None
            and not self._suppress_guards_tls()
            and not size_oblivious
            and not any(symbol_is_type(s, SymT.FLOAT) for s in orig_expr.free_symbols)
            and fallback_value is None
        ):
            # TODO: does this even worked with unbacked :think:
            concrete_val = compute_concrete_val()
            if concrete_val is sympy.true:
                node, fresh = self._create_fx_call_function(torch._assert, (fx_node,))
            elif concrete_val is sympy.false:
                neg, _ = self._create_fx_call_function(operator.not_, (fx_node,))
                node, fresh = self._create_fx_call_function(torch._assert, (neg,))
            else:
                eql, _ = self._create_fx_call_function(operator.eq, (fx_node, concrete_val))
                node, fresh = self._create_fx_call_function(torch._assert, (eql,))

            assert node is not None
            # If this is a fresh node, we have to remember the event index that
            # corresponds to this assertion node.
            # Reason: so that, given an assertion node, we can replay the ShapeEnv
            # events until the point where this assertion node was freshly created.
            if fresh:
                self._add_fx_node_metadata(node)

        # After creating the FX node corresponding to orig_expr, we must make sure that
        # no error will be raised until the end of this function.
        #
        # Reason: the translation validation may become invalid otherwise.
        #
        # If an error is raised before the end of this function, we remove the FX node
        # inserted, and re-raise the error.
        guard = None

        try:
            if orig_expr.is_number:
                self.log.debug("eval %s [trivial]", orig_expr)
                if hint is not None:
                    if isinstance(hint, bool):
                        assert orig_expr == hint, f"{orig_expr} != {hint}"
                    else:
                        assert sympy.Eq(orig_expr, hint), f"{orig_expr} != {hint}"
                return orig_expr

            expr = orig_expr

            static_expr = self._maybe_evaluate_static(expr, size_oblivious=size_oblivious)
            if static_expr is not None:
                self.log.debug(
                    "eval %s == %s [statically known]",
                    (f"size_oblivious({orig_expr})" if size_oblivious else size_oblivious),
                    static_expr,
                )
                if not size_oblivious and config.backed_size_oblivious and hint is not None:
                    # TODO: maybe reconcile this with use of counterfactual hints
                    # in unbacked case
                    assert static_expr == hint, f"{static_expr} != {hint}"
                return static_expr

            transmute_into_runtime_assert = False

            backed_var_to_val = (
                self.backed_var_to_val
                if hasattr(self, "backed_var_to_val")
                else self.var_to_val
            )
            concrete_val = None
            if not (expr.free_symbols <= backed_var_to_val.keys()):
                # TODO: dedupe this with _maybe_evaluate_static
                # Attempt to eliminate the unbacked SymInt
                new_expr = self._maybe_evaluate_static(expr, unbacked_only=True)
                assert new_expr is not None
                if not (new_expr.free_symbols <= backed_var_to_val.keys()):
                    ok = False

                    # fallback_value is set when guard_or_true or guard_or_false are used.
                    if not ok and fallback_value is not None:
                        self._log_suppressed_dde(orig_expr, fallback_value)
                        return fallback_value

                    # oblivious_var_to_val will be defined iff we have sizes
                    # with DimDynamic.OBLIVIOUS_SIZE type.
                    # See https://github.com/pytorch/pytorch/issues/137100#issuecomment-2495778113
                    if (
                        backed_var_to_val
                        and getattr(self, "real_tensor_prop_unbacked_vals", True)
                        and not (
                            correct_hint := orig_expr.xreplace(backed_var_to_val)
                        ).free_symbols
                        and not (
                            counterfactual_hint := orig_expr.xreplace(
                                {k: max(2, v) for k, v in backed_var_to_val.items()}
                            )
                        ).free_symbols
                        and correct_hint == counterfactual_hint
                    ):
                        # TODO: better logging
                        log.info(
                            "oblivious_size %s -> %s (passed counterfactual)",
                            orig_expr,
                            # pyrefly: ignore  # unbound-name
                            correct_hint,
                        )
                        # pyrefly: ignore  # unbound-name
                        concrete_val = correct_hint
                        # NB: do NOT transmute into runtime assert
                        ok = True

                    # unbacked_var_to_val is not None iff propagate_real_tensors is on.
                    # if propagate_real_tensors is on, we check the example values
                    # to generate (unsound_result)
                    # and if they pass we add a runtime assertions and continue.
                    if (
                        not ok
                        and backed_var_to_val
                        and not (
                            unsound_result := orig_expr.xreplace(backed_var_to_val).xreplace(
                                backed_var_to_val
                            )
                        ).free_symbols
                    ):
                        # pyrefly: ignore  # unbound-name
                        self._log_real_tensor_propagation(orig_expr, unsound_result)
                        transmute_into_runtime_assert = True
                        # pyrefly: ignore  # unbound-name
                        concrete_val = unsound_result
                        ok = True

                    # Check if this is coming from a python assert statement,
                    # if so, convert it to a runtime assertion
                    # instead of failing.
                    if not ok and self.trace_asserts and self._is_python_assert():
                        concrete_val = sympy.true
                        transmute_into_runtime_assert = True
                        ok = True

                    # PATCHED: ok -> True
                    ok = True
                    # if not ok:
                    #    raise self._make_data_dependent_error(
                    #        expr.xreplace(self.var_to_val),
                    #        expr,
                    #        expr_sym_node_id=self._expr_sym_node_id,
                    #    )
                else:
                    expr = new_expr

            if concrete_val is None:
                concrete_val = compute_concrete_val()
            self._check_frozen(expr, concrete_val)

            if (
                config.inject_EVALUATE_EXPR_flip_equality_TESTING_ONLY
                and isinstance(hint, bool)
                and isinstance(expr, (sympy.Eq, sympy.Ne))
            ):
                expr = sympy.Not(expr)

            # Turn this into a boolean expression, no longer need to consult
            # concrete_val
            if concrete_val is sympy.true:
                g = cast(SympyBoolean, expr)
            elif concrete_val is sympy.false:
                g = sympy.Not(expr)
            else:
                g = sympy.Eq(expr, concrete_val)  # type: ignore[arg-type]

            if transmute_into_runtime_assert:
                self.guard_or_defer_runtime_assert(
                    g, f"propagate_real_tensors: {orig_expr} == {concrete_val}"
                )
                return concrete_val

            if not self._suppress_guards_tls():
                self._log_guard("eval", g, forcing_spec=forcing_spec)

                # TODO: If we successfully eliminate a symbol via equality, it
                # is not actually necessary to save a guard for the equality,
                # as we will implicitly generate a guard when we match that
                # input against the symbol.  Probably the easiest way to
                # implement this is to have maybe_guard_rel return a bool
                # saying if it "subsumed" the guard (and therefore the guard
                # is no longer necessary)
                self._maybe_guard_rel(g)

                if (
                    torch.compiler.is_exporting()
                    and self.prefer_deferred_runtime_asserts_over_guards
                ):
                    # it's fine to defer simple guards here without checking,
                    # the _maybe_guard_rel() call above will set replacements if possible,
                    # and so the result here will be statically known
                    self.guard_or_defer_runtime_assert(g, f"evaluate_expr: {orig_expr}")
                else:
                    # at this point, we've evaluated the concrete expr value, and have
                    # flipped/negated the guard if necessary. Now we know what to guard
                    # or defer to runtime assert on.
                    guard = ShapeGuard(g, self._get_sloc(), size_oblivious=size_oblivious)
                    self.guards.append(guard)
                    self.axioms.update(dict(self.get_implications(self.simplify(g))))
            else:
                self._log_guard("eval [guard suppressed]", g, forcing_spec=forcing_spec)

        except Exception:
            if fresh:
                self._remove_fx_node(node)
            raise

        if not self._suppress_guards_tls():
            if guard is not None:  # we might have deferred this to runtime assert
                for s in g.free_symbols:
                    self.symbol_guard_counter[s] += 1
                    # Forcing_spec to avoid infinite recursion
                    if (
                        not forcing_spec
                        and config.symbol_guard_limit_before_specialize is not None
                        and self.symbol_guard_counter[s]
                        > config.symbol_guard_limit_before_specialize
                    ):
                        # Force specialization
                        self.log.info(
                            "symbol_guard_limit_before_specialize=%s exceeded on %s",
                            config.symbol_guard_limit_before_specialize,
                            s,
                        )
                        self.evaluate_expr(s, forcing_spec=True)

        return concrete_val


def patched_vmap(func, in_dims=0, out_dims=0, use_scan: bool = False):
    """
    Python implementation of :func:`torch.vmap`.
    The implementation raises an issue when it is being exported with
    :func:`torch.export.export` when the function is called with
    non tensors arguments and the batch size is dynamic.
    """
    from ...helpers import string_type

    def wrapped(*args):
        assert all(not isinstance(a, dict) for a in args), (
            f"dictionaries are not implemented in "
            f"args={string_type(args, with_shape=True)}"
        )

        in_dims_ = (
            ([in_dims] * len(args))
            if not isinstance(in_dims, (list, tuple))
            else list(in_dims)
        )
        assert len(in_dims_) == len(args), (
            f"Mismtch between in_dims={in_dims_} and "
            f"args={string_type(args, with_shape=True)}"
        )

        batch_size = None
        batched_args = []
        for arg, in_dim in zip(args, in_dims_):
            if in_dim is None:
                batched_args.append(arg)
                continue

            assert batch_size is None or batch_size == arg.size(in_dim), (
                f"Unable to continue, batch_size={batch_size}, in_dim={in_dim}, "
                f"arg.size(in_dim)={arg.size(in_dim)}"
            )
            if batch_size is None:
                batch_size = arg.size(in_dim)
            arg = arg.movedim(in_dim, 0)
            batched_args.append(arg)

        if use_scan or (
            all(isinstance(a, torch.Tensor) for a in args)
            and isinstance(batch_size, torch.SymInt)
        ):
            batched_tensors = [
                (
                    arg
                    if (isinstance(arg, torch.Tensor) and in_dim is not None)
                    else arg.unsqueeze(0).expand((batch_size, *arg.shape))
                )
                for arg, in_dim in zip(batched_args, in_dims_)
            ]
            results = torch.ops.higher_order.scan(
                lambda *args, **kwargs: [func(*args, **kwargs)], [], batched_tensors, []
            )
            stacked = results[0]
            if out_dims != 0:
                return stacked.movedim(0, out_dims)
            return stacked

        else:
            torch._check(
                not isinstance(batch_size, torch.SymInt),
                lambda: (
                    f"patched_vmap supports dynamic batch_size only if all arguments "
                    f"are tensors but types are {[type(a) for a in args]}"
                ),
            )
            batched_tensors = [
                (
                    (None, arg)
                    if (isinstance(arg, torch.Tensor) and in_dim is not None)
                    else (arg, arg)
                )
                for arg, in_dim in zip(batched_args, in_dims_)
            ]

            results = []
            for i in range(batch_size):
                input_slice = [v if v is not None else arg[i] for v, arg in batched_tensors]
                result = func(*input_slice)
                results.append(result)

            if isinstance(results[0], torch.Tensor):
                stacked = torch.stack(results)
                if out_dims != 0:
                    return stacked.movedim(0, out_dims)
                return stacked
            return results

    return wrapped


def patched__constrain_user_specified_dimhint_range(
    symint: torch.SymInt,
    hint: int,
    dim: "_DimHint",  # noqa: F821
    range_constraints,
    shape_env,
    keypath: "KeyPath",  # noqa: F821
    i: Optional[int] = None,
) -> Optional[str]:
    """Patches ``torch._export.non_strict_utils._constrain_user_specified_dimhint_range``."""
    from torch._export.non_strict_utils import is_int, int_oo, _DimHintType, ValueRanges

    trace_vr = (
        range_constraints[symint.node.expr]
        if not is_int(symint)
        else ValueRanges(int(symint), int(symint))
    )
    # warn on 0/1 specialization for Dim.AUTO; not an actual error
    # PATCHED: remove logging
    # if dim.type == _DimHintType.AUTO and trace_vr.is_singleton() and hint in (0, 1):
    #    pathstr = f"inputs{pytree.keystr(keypath)}"
    #    if i is not None:
    #        pathstr += f".shape[{i}]"
    #    msg = (
    #        f"dimension {pathstr} 0/1 specialized; Dim.AUTO was specified along "
    #        f"with a sample input with hint = {hint}."
    #    )
    #    log.warning(msg)

    try:
        user_vr = ValueRanges(
            lower=0 if dim.min is None else dim.min,
            upper=int_oo if dim.max is None else dim.max,
        )
        if is_int(symint):
            out_vr = trace_vr & user_vr
        else:
            range_constraints[symint.node.expr] &= user_vr
            shape_env.var_to_range[symint.node._expr] &= user_vr
            out_vr = range_constraints[symint.node.expr]

        # check for Dim.DYNAMIC specializations; special case error message on 0/1
        if dim.type == _DimHintType.DYNAMIC and out_vr.is_singleton():
            path = f"inputs{torch.utils._pytree.keystr(keypath)}"
            if i is not None:
                path += f".shape[{i}]"
            if (
                trace_vr.is_singleton()
                and hint in (0, 1)
                # PATCHED: line removed
                # and not torch.fx.experimental._config.backed_size_oblivious
            ):
                return None
                # PATCHED: line removed
                # msg = (
                #     f"- Received user-specified dim hint "
                #     f"Dim.DYNAMIC(min={dim.min}, max={dim.max}), "
                #     f"but export 0/1 specialized due to hint of "
                #     f"{hint} for dimension {path}."
                # )
            else:
                msg = (
                    f"- Received user-specified dim hint "
                    f"Dim.DYNAMIC(min={dim.min}, max={dim.max}), "
                    f"but tracing inferred a static shape of "
                    f"{out_vr.lower} for dimension {path}."
                )
            return msg

    except torch.utils._sympy.value_ranges.ValueRangeError:
        path = f"inputs{torch.utils._pytree.keystr(keypath)}"
        if i is not None:
            path += f".shape[{i}]"
        msg = (
            f"- Received user-specified min/max range of [{dim.min}, {dim.max}], "
            f"conflicting with the inferred min/max range of "
            f"[{trace_vr.lower}, {trace_vr.upper}], "
            f"for {path}."
        )
        return msg

    return None


def patched__maybe_broadcast(*args, preserve_cpu_scalar_tensors=True):
    """Patches ``torch._refs._maybe_broadcast``."""
    from torch._prims_common import ShapeType, TensorLike, Number

    # Computes common shape
    common_shape = patched__broadcast_shapes(
        *(t.shape if isinstance(t, TensorLike) else None for t in args)
    )

    def should_expand(a: ShapeType, b: ShapeType) -> bool:
        from torch.fx.experimental.symbolic_shapes import (
            guard_or_false,
            sym_and,
            sym_or,
        )

        if len(a) != len(b):
            return True

        for x, y in zip(a, b):
            if guard_or_false(x != y):
                # We know they are not the same.
                return True

            # They are the same or we do not know if they are the same or not.
            # 1==1 no-broadcast
            # u0==1 and 1==u0 cases. We broadcast!
            if guard_or_false(sym_and(x == 1, y == 1)):
                pass
            elif guard_or_false(sym_or(x == 1, y == 1)):
                # assume broadcasting.
                return True

            # u0==u1 assume the same, no broadcasting!
            # PATCHED: avoid errors
            return True  # guard_or_true(x != y)
            # torch._check(
            #    x == y,
            #    lambda x=x, y=y: (
            #        f"sizes assumed to be the same due to unbacked "
            #        f"broadcasting semantics x={x!r}, y={y!r}"
            #    ),
            # )

        return False

    def __maybe_broadcast(x, shape):
        if x is None:
            return None
        elif isinstance(x, Number):
            return x
        elif isinstance(x, TensorLike):
            if preserve_cpu_scalar_tensors and torch._prims_common.is_cpu_scalar_tensor(x):
                return x

            if should_expand(x.shape, common_shape):
                return x.expand(common_shape)

            return x
        else:
            raise RuntimeError(f"Unexpected type when broadcasting: {str(type(x))}!")

    return tuple(__maybe_broadcast(x, common_shape) for x in args)


def patched__broadcast_in_dim_meta(
    a: torch._prims_common.TensorLikeType,
    shape: torch._prims_common.ShapeType,
    broadcast_dimensions: Sequence[int],
):
    """Patches ``torch._prims._broadcast_in_dim_meta``."""
    from torch.fx.experimental.symbolic_shapes import (
        guard_or_false,
        guard_or_true,
        sym_or,
    )

    # Type checks
    assert isinstance(a, torch._prims_common.TensorLike)
    assert isinstance(shape, Sequence)
    assert isinstance(broadcast_dimensions, Sequence)

    # every dimension must be accounted for
    assert a.ndim == len(broadcast_dimensions)

    # broadcast shape must have weakly more dimensions
    assert len(shape) >= a.ndim

    # broadcast_dimensions must be an ascending sequence
    # (no relative reordering of dims) of integers and
    # each dimension must be within the new shape
    def _greater_than_reduce(acc, x):
        assert isinstance(x, (int, torch.export.Dim)), f"unexpected type {type(x)} for x"
        assert x > acc
        assert x < len(shape)

        return x

    reduce(_greater_than_reduce, broadcast_dimensions, -1)

    # shape must be broadcastable to
    for idx, new_idx in enumerate(broadcast_dimensions):
        torch._check(
            sym_or(a.shape[idx] == 1, shape[new_idx] == a.shape[idx]),
            lambda idx=idx, new_idx=new_idx: (
                f"{a.shape[idx]} must be broadcastable to {shape[new_idx]}"
            ),
        )

    new_strides = []
    original_idx = 0
    for idx in range(len(shape)):
        if idx in broadcast_dimensions:
            # Assigns a stride of zero to dimensions
            # which were actually broadcast
            if guard_or_false(a.shape[original_idx] == 1):
                if guard_or_false(a.shape[original_idx] == shape[idx]):
                    new_strides.append(a.stride()[original_idx])
                else:
                    new_strides.append(0)
            # PATCHED: disabled this check
            elif guard_or_false(a.shape[original_idx] != 1):
                new_strides.append(a.stride()[original_idx])
            else:
                # This checks generates the following issue:
                # non-broadcasting semantics require s3 == Max(s10, s3), False,
                # guard_or_false(a.shape[idx]==1)=False, a.stride()=(1, 2),
                # idx=1, a.shape=torch.Size([2, s3]), shape=[2, Max(s10, s3)],
                # original_idx=1
                torch._check(
                    a.shape[original_idx] == shape[idx],
                    lambda idx=idx, original_idx=original_idx: (
                        f"non-broadcasting semantics require "
                        f"{a.shape[original_idx]} == {shape[idx]}, "
                        f"{guard_or_false(a.shape[idx] != 1)}, "
                        f"guard_or_false(a.shape[idx]==1)="
                        f"{guard_or_false(a.shape[idx] == 1)}, "
                        f"a.stride()={a.stride()}, idx={idx}, a.shape={a.shape}, "
                        f"shape={shape}, original_idx={original_idx}"
                    ),
                )
                new_strides.append(a.stride()[original_idx])
            original_idx = original_idx + 1
        else:
            if guard_or_true(shape[idx] != 1):
                # consistent with previous use of guard_size_oblivious
                new_strides.append(0)
            elif original_idx == a.ndim:
                new_strides.append(1)
            else:
                new_strides.append(a.stride()[original_idx] * a.size()[original_idx])

    return a.as_strided(shape, new_strides, a.storage_offset())


def patched__broadcast_in_dim_meta_level_2(
    a: torch._prims_common.TensorLikeType,
    shape: torch._prims_common.ShapeType,
    broadcast_dimensions: Sequence[int],
):
    """Patches ``torch._prims._broadcast_in_dim_meta``."""
    from torch.fx.experimental.symbolic_shapes import (
        guard_or_false,
        guard_or_true,
        sym_or,
    )

    # Type checks
    assert isinstance(a, torch._prims_common.TensorLike)
    assert isinstance(shape, Sequence)
    assert isinstance(broadcast_dimensions, Sequence)

    # every dimension must be accounted for
    assert a.ndim == len(broadcast_dimensions)

    # broadcast shape must have weakly more dimensions
    assert len(shape) >= a.ndim

    # broadcast_dimensions must be an ascending sequence
    # (no relative reordering of dims) of integers and
    # each dimension must be within the new shape
    def _greater_than_reduce(acc, x):
        assert isinstance(x, (int, torch.export.Dim)), f"unexpected type {type(x)} for x"
        assert x > acc
        assert x < len(shape)

        return x

    reduce(_greater_than_reduce, broadcast_dimensions, -1)

    # shape must be broadcastable to
    for idx, new_idx in enumerate(broadcast_dimensions):
        torch._check(
            sym_or(a.shape[idx] == 1, shape[new_idx] == a.shape[idx]),
            lambda idx=idx, new_idx=new_idx: (
                f"{a.shape[idx]} must be broadcastable to {shape[new_idx]}"
            ),
        )

    new_strides = []
    original_idx = 0
    for idx in range(len(shape)):
        if idx in broadcast_dimensions:
            # Assigns a stride of zero to dimensions
            # which were actually broadcast
            if guard_or_false(a.shape[original_idx] == 1):
                if guard_or_false(a.shape[original_idx] == shape[idx]):
                    new_strides.append(a.stride()[original_idx])
                else:
                    new_strides.append(0)
            # PATCHED: disabled this check
            elif guard_or_false(a.shape[original_idx] != 1):
                new_strides.append(a.stride()[original_idx])
            else:
                # PATCHED: torch._check was removed
                new_strides.append(a.stride()[original_idx])
            original_idx = original_idx + 1
        else:
            if guard_or_true(shape[idx] != 1):
                # consistent with previous use of guard_size_oblivious
                new_strides.append(0)
            elif original_idx == a.ndim:
                new_strides.append(1)
            else:
                new_strides.append(a.stride()[original_idx] * a.size()[original_idx])

    return a.as_strided(shape, new_strides, a.storage_offset())


class patched_DynamicDimConstraintPrinter:
    """
    Patches
    ``torch.tx.experimental.symbolic_shapes.DynamicDimConstraintPrinter._print_Symbol``.
    Valid for ``torch>=2.10``.
    """

    def _print_Symbol(self, expr: sympy.Symbol) -> str:
        assert isinstance(expr, sympy.Symbol), str(type(expr))
        if self.symbol_to_source.get(expr):
            return self.symbol_to_source[expr][0].name
        return str(expr)
