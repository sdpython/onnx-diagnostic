import os


def get_data(name: str):
    """Returns data stored in this folder."""
    filename = os.path.join(os.path.dirname(__file__), name)
    assert os.path.exists(
        filename
    ), f"Unable to find a file with {name!r}, looked for {filename!r}"

    from ...helpers.mini_onnx_builder import create_input_tensors_from_onnx_model

    return create_input_tensors_from_onnx_model(filename)
