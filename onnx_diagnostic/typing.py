from typing import Any, Dict, List, Protocol, Tuple, runtime_checkable


@runtime_checkable
class TensorLike(Protocol):
    @property
    def shape(self) -> Tuple[int, ...]: ...
    @property
    def dtype(self) -> object: ...


@runtime_checkable
class InferenceSessionLike(Protocol):
    def __init__(self, model: Any, **kwargs): ...
    def run(self, feeds: Dict[str, TensorLike]) -> List[TensorLike]: ...
