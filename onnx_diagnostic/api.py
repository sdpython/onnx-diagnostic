from typing import Any


class TensorLike:
    """Mocks a tensor."""

    @property
    def dtype(self) -> Any:
        "Must be overwritten."
        raise NotImplementedError("dtype must be overwritten.")

    @property
    def shape(self) -> Any:
        "Must be overwritten."
        raise NotImplementedError("shape must be overwritten.")
