from typing import Callable


class PatchDetails:
    """
    This class is used to store patching information.
    This helps understanding which rewriting was applied to which
    method of functions.
    """

    def __init__(self):
        self.rewritten = []

    def append(self, family: str, rewritten: Callable, patched: Callable):
        self.rewritten.append((family, rewritten, patched))
