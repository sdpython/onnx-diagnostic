import torch
from typing import Any, Dict, Optional, Tuple, Union
import time
from ..helpers import string_type
from .hghub import get_untrained_model_with_inputs
from .hghub.model_inputs import random_input_kwargs


def _ds_clean(v):
    return (
        str(v)
        .replace("<class 'onnx_diagnostic.torch_models.hghub.model_inputs.", "")
        .replace("'>", "")
        .replace("_DimHint(type=<_DimHintType.DYNAMIC: 3>", "DYNAMIC")
        .replace("_DimHint(type=<_DimHintType.AUTO: 3>", "AUTO")
    )


def get_inputs_for_task(task: str, config: Optional[Any] = None) -> Dict[str, Any]:
    """
    Returns dummy inputs for a specific task.

    :param task: requested task
    :param config: returns dummy inputs for a specific config if available
    :return: dummy inputs and dynamic shapes
    """
    kwargs, f = random_input_kwargs(config, task)
    return f(model=None, config=config, **kwargs)


def validate_model(
    model_id: str,
    task: Optional[str] = None,
    do_run: bool = False,
    exporter: Optional[str] = None,
    do_same: bool = False,
    verbose: int = 0,
    dtype: Optional[Union[str, torch.dtype]] = None,
    device: Optional[Union[str, torch.device]] = None,
    trained: bool = False,
    optimization: Optional[str] = None,
    quiet: bool = False,
) -> Tuple[Dict[str, Union[int, float, str]], Dict[str, Any]]:
    """
    Validates a model.

    :param model_id: model id to validate
    :param task: task used to generate the necessary inputs,
        can be left empty to use the default task for this model
        if it can be determined
    :param do_run: checks the model works with the defined inputs
    :param exporter: exporter the model using this exporter,
        available list: ``export-strict``, ``export-nostrict``, ``onnx``
    :param do_same: checks the discrepancies of the exported model
    :param verbose: verbosity level
    :param dtype: uses this dtype to check the model
    :param device: do the verification on this device
    :param trained: use the trained model, not the untrained one
    :param optimization: optimization to apply to the exported model,
        depend on the the exporter
    :param quiet: if quiet, catches exception if any issue
    :return: two dictionaries, one with some metrics,
        another one with whatever the function produces
    """
    assert not trained, f"trained={trained} not supported yet"
    assert not dtype, f"dtype={dtype} not supported yet"
    assert not device, f"device={device} not supported yet"
    summary: Dict[str, Union[int, float, str]] = {}
    if verbose:
        print(f"[validate_model] validate model id {model_id!r}")
        print("[validate_model] get dummy inputs...")
        summary["model_id"] = model_id
    begin = time.perf_counter()
    if quiet:
        try:
            data = get_untrained_model_with_inputs(model_id, verbose=verbose, task=task)
        except Exception as e:
            summary["ERR_create"] = str(e)
            data["ERR_create"] = e
            summary["time_create"] = time.perf_counter() - begin
            return summary, {}
    else:
        data = get_untrained_model_with_inputs(model_id, verbose=verbose, task=task)
    summary["time_create"] = time.perf_counter() - begin
    for k in ["task", "size", "n_weights"]:
        summary[f"model_{k.replace('_','')}"] = data[k]
        summary["model_inputs"] = string_type(data["inputs"], with_shape=True)
        summary["model_shapes"] = _ds_clean(str(data["dynamic_shapes"]))
        summary["model_class"] = data["model"].__class__.__name__
        summary["model_config_class"] = data["configuration"].__class__.__name__
        summary["model_config"] = str(data["configuration"].to_dict()).replace(" ", "")
    summary["model_id"] = model_id
    if verbose:
        print(f"[validate_model] task={data['task']}")
        print(f"[validate_model] size={data['size']}")
        print(f"[validate_model] n_weights={data['n_weights']}")
        print(f"[validate_model] n_weights={data['n_weights']}")
        for k, v in data["inputs"].items():
            print(f"[validate_model] +INPUT {k}={string_type(v, with_shape=True)}")
        for k, v in data["dynamic_shapes"].items():
            print(f"[validate_model] +SHAPE {k}={_ds_clean(v)}")
    if do_run:
        if verbose:
            print("[validate_model] run the model...")
        begin = time.perf_counter()
        if quiet:
            try:
                expected = data["model"](**data["inputs"])
            except Exception as e:
                summary["ERR_run"] = str(e)
                data["ERR_run"] = e
                summary["time_run"] = time.perf_counter() - begin
                return summary, data
        else:
            expected = data["model"](**data["inputs"])
        summary["time_run"] = time.perf_counter() - begin
        summary["model_expected"] = string_type(expected, with_shape=True)
        if verbose:
            print("[validate_model] run the model")
        data["expected"] = expected
    if verbose:
        print("[validate_model] done.")
    return summary, data
