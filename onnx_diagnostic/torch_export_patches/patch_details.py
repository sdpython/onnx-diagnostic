import difflib
import inspect
import textwrap
from typing import Any, Dict, Callable, List, Optional, Union


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


class PatchDetails:
    """
    This class is used to store patching information.
    This helps understanding which rewriting was applied to which
    method of functions.
    """

    def __init__(self):
        self.patched = []

    def append(self, family: str, function_to_patch: Union[str, Callable], patch: Callable):
        assert callable(function_to_patch) or isinstance(function_to_patch, str), (
            f"function_to_patch is not a function but {type(function_to_patch)} "
            f"- {function_to_patch!r}"
        )
        assert callable(
            patch
        ), f"function_to_patch is not a function but {type(patch)} - {patch!r}"
        self.patched.append((family, function_to_patch, patch))

    @property
    def n_patches(self) -> int:
        "Returns the number of stored patches."
        # Overwritten __len__ may have an impact on bool(patch_details: PatchDetails)
        return len(self.patched)

    def data(self) -> List[Dict[str, Any]]:
        """Returns the data for a dataframe."""
        return [dict(zip(["type", "patched", "patch"], v)) for v in self.patched]

    def make_diff(self, function_to_patch: Callable, patch: Callable) -> str:
        """
        Returns a diff as a string.

        :param function_to_patch: function to pathc
        :param patch: function patched
        :return: diff
        """
        assert callable(function_to_patch) or isinstance(function_to_patch, str), (
            f"function_to_patch is not a function but {type(function_to_patch)} "
            f"- {function_to_patch!r}"
        )
        assert callable(patch), (
            f"function_to_patch is not a function but {type(patch)} - {patch!r} "
            f"(function_to_patch={function_to_patch!r})"
        )
        if isinstance(function_to_patch, str):
            return clean_code_with_black(inspect.getsource(patch))
        src1 = clean_code_with_black(inspect.getsource(function_to_patch))
        src2 = clean_code_with_black(inspect.getsource(patch))
        return make_diff_code(src1, src2)

    def format_diff(
        self,
        function_to_patch: Callable,
        patch: Callable,
        kind: Optional[str] = None,
        format: str = "raw",
    ) -> str:
        """
        Format a diff between two function as a string.

        :param function_to_patch: function to pathc
        :param patch: function patched
        :param kind: included in the title
        :param raw: ``'raw'`` or ``'rst'``
        :return: diff

        .. runpython::
            :showcode:
            :rst:

            import transformers
            import onnx_diagnostic.torch_export_patches.patches.patch_transformers as ptr
            from onnx_diagnostic.torch_export_patches.patch_details import PatchDetails

            diff = PatchDetails().format_diff(eager_mask, patched_eager_mask, format="rst")
            print(diff)
        """
        diff = self.make_diff(function_to_patch, patch)
        kind = kind or ""
        if kind:
            kind = f"{kind}: "
        title = (
            f"{kind}{function_to_patch!r} -> {patch.__name__}"
            if isinstance(function_to_patch, str)
            else f"{kind}{function_to_patch.__name__} -> {patch.__name__}"
        )
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
