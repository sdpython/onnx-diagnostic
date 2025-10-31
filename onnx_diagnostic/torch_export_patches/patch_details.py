import difflib
from typing import Any, Dict, Callable, List, Optional


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

    def append(self, family: str, rewritten: Callable, patched: Callable):
        self.patched.append((family, rewritten, patched))

    @property
    def n_patches(self) -> int:
        "Returns the number of stored patches."
        # Overwritten __len__ may have an impact on bool(patch_details: PatchDetails)
        return len(self.patched)

    def data(self) -> List[Dict[str, Any]]:
        """Returns the data for a dataframe."""
        return [dict(zip(["type", "patched", "patch"], v)) for v in self.patched]
