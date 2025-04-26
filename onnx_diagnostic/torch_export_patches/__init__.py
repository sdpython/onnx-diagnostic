from .onnx_export_errors import (
    torch_export_patches,
    register_additional_serialization_functions,
)


# bypass_export_some_errors is the first name given to the patches.
bypass_export_some_errors = torch_export_patches  # type: ignore
