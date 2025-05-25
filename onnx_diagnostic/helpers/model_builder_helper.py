import importlib.util
import os
import requests
import sys
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

CACHE_SUBDIR = "onnx_diagnostic_cache"


def download_model_builder_to_cache(
    url: str = "https://raw.githubusercontent.com/microsoft/onnxruntime-genai/refs/heads/main/src/python/py/models/builder.py",
):
    """
    Downloads ``builder.py`` from the
    ``https://github.com/microsoft/onnxruntime-genai/blob/main/src/python/py/models/builder.py``.
    """
    filename = os.path.basename(urlparse(url).path)
    cache_dir = Path(os.getenv("HOME", Path.home())) / ".cache" / CACHE_SUBDIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    file_path = cache_dir / filename

    if file_path.exists():
        return file_path

    response = requests.get(url)
    response.raise_for_status()
    with open(file_path, "wb") as f:
        f.write(response.content)
    return file_path


def import_model_builder(module_name: str = "builder") -> object:
    """Imports the downloaded ``model.by``."""
    if module_name in sys.modules:
        return sys.modules[module_name]
    path = Path(os.getenv("HOME", Path.home())) / ".cache" / CACHE_SUBDIR
    module_file = path / f"{module_name}.py"
    assert os.path.exists(module_file), f"Unable to find {module_file!r}"
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None:
        spath = str(path)
        if spath not in sys.path:
            sys.path.append(spath)
        module = importlib.__import__(module_name)
        return module
    assert spec is not None, f"Unable to import module {module_name!r} from {str(path)!r}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def create_model(
    config: Any,
    cache_dir: Optional[str] = None,
    precision: str = "fp32",
    execution_provider: str = "cpu",
    **extra_options,
) -> "Model":  # noqa: F821
    """
    Creates a model based on a configuration.

    :param config: configuration
    :param cache_dir: cache directory
    :param precision: precision
    :param execution_provider: execution provider
    :param extra_options: extra options
    :return: model
    """
    download_model_builder_to_cache()
    builder = import_model_builder()
    extra_kwargs = {}
    io_dtype = builder.set_io_dtype(precision, execution_provider, extra_options)
    onnx_model = builder.Model(
        config,
        io_dtype,
        precision,
        execution_provider,
        cache_dir,
        extra_options,
        **extra_kwargs,
    )
    # onnx_model.make_genai_config(hf_name, extra_kwargs, output_dir)
    # onnx_model.save_processing(hf_name, extra_kwargs, output_dir)
    return onnx_model
