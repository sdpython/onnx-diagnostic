from .onnx_export_errors import (
    torch_export_patches,
    register_additional_serialization_functions,
)


# bypass_export_some_errors is the first name given to the patches.
bypass_export_some_errors = torch_export_patches  # type: ignore


def register_flattening_functions(verbose: int = 0):
    """
    Registers functions to serialize deserialize cache or other classes
    implemented in :epkg:`transformers` and used as inputs.
    This is needed whenever a model must be exported through
    :func:`torch.export.export`.
    """
    from .onnx_export_serialization import _register_cache_serialization

    return _register_cache_serialization(verbose=verbose)
