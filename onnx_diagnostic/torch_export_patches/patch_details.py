import difflib
import inspect
import re
import textwrap
from typing import Any, Dict, Callable, List, Optional, Tuple, Union


def clean_code_with_black(code: str) -> str:
    """Changes the code style with :epkg:`black` if available."""
    code = textwrap.dedent(code)
    try:
        import black
    except ImportError:
        return code
    try:
        return black.format_str(code, mode=black.FileMode(line_length=98))
    except black.parsing.InvalidInput as e:
        raise RuntimeError(f"Unable to parse code\n\n---\n{code}\n---\n") from e


def make_diff_code(code1: str, code2: str, output: Optional[str] = None) -> str:
    """
    Creates a diff between two codes.

    :param code1: first code
    :param code2: second code
    :param output: if not empty, stores the output in this file
    :return: diff
    """
    text = "\n".join(
        difflib.unified_diff(
            code1.strip().splitlines(),
            code2.strip().splitlines(),
            fromfile="original",
            tofile="rewritten",
            lineterm="",
        )
    )
    if output:
        with open(output, "w") as f:
            f.write(text)
    return text


class PatchInfo:
    """
    Stores informations about patches.

    :param function_to_patch: function to pathc
    :param patch: function patched
    :param family: a category, anything to classify the patch
    """

    __slots__ = ("family", "function_to_patch", "patch")

    def __init__(
        self, function_to_patch: Union[str, Callable], patch: Callable, family: str = ""
    ):
        assert callable(function_to_patch) or isinstance(function_to_patch, str), (
            f"function_to_patch is not a function but {type(function_to_patch)} "
            f"- {function_to_patch!r}"
        )
        assert callable(
            patch
        ), f"function_to_patch is not a function but {type(patch)} - {patch!r}"
        self.family = family
        self.function_to_patch = function_to_patch
        self.patch = patch

    def __repr__(self) -> str:
        "usual"
        return (
            (
                f"{self.__class__.__name__}({self.function_to_patch!r}, {self.patch!r}, "
                f"{self.family!r})"
            )
            if self.family
            else f"{self.__class__.__name__}({self.function_to_patch!r}, {self.patch!r})"
        )

    def to_tuple(self) -> Tuple[str, Callable, Callable]:
        "usual"
        return (self.family, self.function_to_patch, self.patch)

    def to_dict(self) -> Dict[str, Any]:
        "usual"
        return {k: getattr(self, k) for k in self.__slots__}

    def make_diff(self) -> str:
        """Returns a diff as a string."""
        if isinstance(self.function_to_patch, str):
            return clean_code_with_black(inspect.getsource(self.patch))
        src1 = clean_code_with_black(inspect.getsource(self.function_to_patch))
        src2 = clean_code_with_black(inspect.getsource(self.patch))
        return make_diff_code(src1, src2)

    @classmethod
    def function_name(cls, f: Callable) -> str:
        return f.__qualname__

    def format_diff(self, format: str = "raw") -> str:
        """
        Format a diff between two function as a string.

        :param format: ``'raw'`` or ``'rst'``
        :return: diff

        .. runpython::
            :showcode:
            :rst:

            import transformers
            import onnx_diagnostic.torch_export_patches.patches.patch_transformers as ptr
            from onnx_diagnostic.torch_export_patches.patch_details import Patchinfo

            diff = Patchinfo(eager_mask, patched_eager_mask).format_diff(format="rst")
            print(diff)
        """
        diff = self.make_diff()
        kind = self.family or ""
        if kind:
            kind = f"{kind}: "
        function_to_pach_name = (
            f"{self.function_to_patch!r}"
            if isinstance(self.function_to_patch, str)
            else self.function_name(self.function_to_patch)
        )
        patch_name = self.function_name(self.patch)
        title = f"{kind}{function_to_pach_name} -> {patch_name}"
        if format == "raw":
            return f"{title}\n{diff}"

        rows = [
            title,
            "=" * len(title),
            "",
            ".. code-block:: diff",
            "    :linenos:",
            "",
            textwrap.indent(diff, prefix="    "),
        ]
        return "\n".join(rows)


class PatchDetails:
    """
    This class is used to store patching information.
    This helps understanding which rewriting was applied to which
    method of functions.

    .. runpython::
        :showcode:
        :rst:

        import torch
        from onnx_diagnostic.torch_export_patches import torch_export_patches
        from onnx_diagnostic.torch_export_patches.patch_inputs import use_dyn_not_str
        from onnx_diagnostic.torch_export_patches.patch_details import PatchDetails
        from onnx_diagnostic.torch_models.hghub import get_untrained_model_with_inputs

        data = get_untrained_model_with_inputs("arnir0/Tiny-LLM", verbose=0)
        model, inputs, ds = data["model"], data["inputs"], data["dynamic_shapes"]
        details = PatchDetails()
        with torch_export_patches(
            patch_transformers=True, patch_details=details, patch_torch=False
        ):
            ep = torch.export.export(
                model, (), kwargs=inputs, dynamic_shapes=use_dyn_not_str(ds)
            )
        patches = details.patches_involded_in_graph(ep.graph)
        report = details.make_report(patches, format="rst")
        print(report)
    """

    def __init__(self):
        self.patched = []

    def append(self, family: str, function_to_patch: Union[str, Callable], patch: Callable):
        """
        Stores a patch.

        :param family: a category, anything to classify the patch
        :param function_to_patch: function to pathc
        :param patch: function patched
        """
        self.patched.append(PatchInfo(function_to_patch, patch, family=family))

    @property
    def n_patches(self) -> int:
        "Returns the number of stored patches."
        # Overwritten __len__ may have an impact on bool(patch_details: PatchDetails)
        return len(self.patched)

    def data(self) -> List[Dict[str, Any]]:
        """Returns the data for a dataframe."""
        return [p.to_dict() for p in self.patched]

    def patches_involded_in_graph(
        self, graph: "torch.fx.Graph"  # noqa: F821
    ) -> List[Tuple[PatchInfo, List["torch.fx.Node"]]]:  # noqa: F821
        """
        Enumerates all patches impacting a graph.
        The function goes through the graph node (only the main graph) and
        looks into the metadata to determine if a listed patch was involved.

        :param graph: fx graph
        :return: list of nodes impacted by a patch
        """
        patches = []
        for patch in self.patched:
            f = patch.patch
            source = inspect.getsourcefile(f)
            lines, lineno = inspect.getsourcelines(f)
            interval = [lineno, lineno + len(lines)]
            patches.append((patch, f, source, interval))

        cst = "onnx_diagnostic"
        node_stack = []
        for node in graph.nodes:
            meta = node.meta
            if "stack_trace" not in meta:
                continue
            stack = meta["stack_trace"]
            if cst not in stack:
                # to reduce the cost of the next iteration
                continue
            node_stack.append((node, stack))

        patch_node = []
        for patch, _f, source, interval in patches:
            exp = 'File "([^"]*?%s[^"]+?)", line (\\d+)' % cst
            reg = re.compile(exp)
            for node, stack in node_stack:
                occ = reg.findall(stack)
                if not occ:
                    continue
                for filename, line_number in occ:
                    if source.replace("\\", "/").strip("/") != filename.replace(
                        "\\", "/"
                    ).strip("/"):
                        continue
                    line = int(line_number)
                    if (
                        line >= interval[0]
                        and line <= interval[1]
                        and self.matching_pair(patch, node)
                    ):
                        patch_node.append((patch, node))

        res = {}
        for patch, node in patch_node:
            if patch not in res:
                res[patch] = []
            res[patch].append(node)
        return list(res.items())

    def matching_pair(cls, patch: PatchInfo, node: "torch.fx.Node") -> bool:  # noqa: F821
        """
        Last validation for a pair. RotaryEmbedding has many rewriting
        and they all end up in the same code line.
        """
        cls_name = patch.function_to_patch.__qualname__.split(".")[0]
        if not cls_name.endswith("RotaryEmbedding"):
            return True
        return cls_name in str(node.meta)

    def make_report(
        cls,
        patches: List[Tuple[PatchInfo, List["torch.fx.Node"]]],  # noqa: F821
        format: str = "raw",
    ) -> str:
        """
        Creates a report based on the involved patches.

        :param patches: from method :meth:`patches_involded_in_graph`
        :param format: format of the report
        :return: report
        """
        rows = []
        for patch, nodes in patches:
            rows.append(patch.format_diff(format=format))
            rows.append("")
            if format == "rst":
                rows.extend(["", "", "**impacted nodes**", "", "", ".. code-block:: raw", ""])
            for node in nodes:
                rows.append(
                    f"    {node.target}({', '.join(map(str,node.args))}) -> {node.name}"
                )
            rows.append("")
        return "\n".join(rows)
