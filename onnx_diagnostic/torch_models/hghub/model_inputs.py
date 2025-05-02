import inspect
from typing import Any, Dict, Optional, Tuple
import torch
import transformers
from ...helpers.config_helper import update_config
from ...tasks import reduce_model_config, random_input_kwargs
from .hub_api import task_from_arch, task_from_id, get_pretrained_config


def get_untrained_model_with_inputs(
    model_id: str,
    config: Optional[Any] = None,
    task: Optional[str] = "",
    inputs_kwargs: Optional[Dict[str, Any]] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    verbose: int = 0,
    dynamic_rope: Optional[bool] = None,
    same_as_pretrained: bool = False,
    use_preinstalled: bool = True,
    add_second_input: bool = False,
    subfolder: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Gets a non initialized model similar to the original model
    based on the model id given to the function.
    The model size is reduced compare to the original model.
    No weight is downloaded, only the configuration file sometimes.

    :param model_id: model id, ex: :epkg:`arnir0/Tiny-LLM`
    :param config: to overwrite the configuration
    :param task: model task, can be overwritten, otherwise, it is automatically determined
    :param input_kwargs: parameters sent to input generation
    :param model_kwargs: to change the model generation
    :param verbose: display found information
    :param dynamic_rope: use dynamic rope (see :class:`transformers.LlamaConfig`)
    :param same_as_pretrained: if True, do not change the default values
        to get a smaller model
    :param use_preinstalled: use preinstalled configurations
    :param add_second_input: provides a second inputs to check a model
        supports different shapes
    :param subfolder: subfolder to use for this model id
    :return: dictionary with a model, inputs, dynamic shapes, and the configuration

    Example:

    .. runpython::
        :showcode:

        import pprint
        from onnx_diagnostic.helpers import string_type
        from onnx_diagnostic.torch_models.hghub import get_untrained_model_with_inputs

        data = get_untrained_model_with_inputs("arnir0/Tiny-LLM", verbose=1)

        print("-- model size:", data['size'])
        print("-- number of parameters:", data['n_weights'])
        print("-- inputs:", string_type(data['inputs'], with_shape=True))
        print("-- dynamic shapes:", pprint.pformat(data['dynamic_shapes']))
        print("-- configuration:", pprint.pformat(data['configuration']))
    """
    if verbose:
        print(f"[get_untrained_model_with_inputs] model_id={model_id!r}")
        if use_preinstalled:
            print(f"[get_untrained_model_with_inputs] use preinstalled {model_id!r}")
    if config is None:
        config = get_pretrained_config(
            model_id,
            use_preinstalled=use_preinstalled,
            subfolder=subfolder,
            **(model_kwargs or {}),
        )
    if hasattr(config, "architecture") and config.architecture:
        archs = [config.architecture]
    if type(config) is dict:
        assert "_class_name" in config, f"Unable to get the architecture from config={config}"
        archs = [config["_class_name"]]
    else:
        archs = config.architectures  # type: ignore
    task = None
    if archs is None:
        task = task_from_id(model_id)
    assert task is not None or (archs is not None and len(archs) == 1), (
        f"Unable to determine the architecture for model {model_id!r}, "
        f"architectures={archs!r}, conf={config}"
    )
    if verbose:
        print(f"[get_untrained_model_with_inputs] architectures={archs!r}")
        print(f"[get_untrained_model_with_inputs] cls={config.__class__.__name__!r}")
    if task is None:
        task = task_from_arch(archs[0])
    if verbose:
        print(f"[get_untrained_model_with_inputs] task={task!r}")

    # model kwagrs
    if dynamic_rope is not None:
        assert (
            type(config) is not dict
        ), f"Unable to set dynamic_rope if the configuration is a dictionary\n{config}"
        assert hasattr(config, "rope_scaling"), f"Missing 'rope_scaling' in\n{config}"
        config.rope_scaling = (
            {"rope_type": "dynamic", "factor": 10.0} if dynamic_rope else None
        )

    # updating the configuration

    mkwargs = reduce_model_config(config, task) if not same_as_pretrained else {}
    if model_kwargs:
        for k, v in model_kwargs.items():
            if isinstance(v, dict):
                if k in mkwargs:
                    mkwargs[k].update(v)
                else:
                    mkwargs[k] = v
            else:
                mkwargs[k] = v
    if mkwargs:
        update_config(config, mkwargs)

    # input kwargs
    kwargs, fct = random_input_kwargs(config, task)
    if verbose:
        print(f"[get_untrained_model_with_inputs] use fct={fct}")
    if inputs_kwargs:
        kwargs.update(inputs_kwargs)

    if archs is not None:
        model = getattr(transformers, archs[0])(config)
    else:
        assert same_as_pretrained, (
            f"Model {model_id!r} cannot be built, the model cannot be built. "
            f"It must be downloaded. Use same_as_pretrained=True."
        )
        model = None

    # This line is important. Some models may produce different
    # outputs even with the same inputs in training mode.
    model.eval()
    res = fct(model, config, add_second_input=add_second_input, **kwargs)

    res["input_kwargs"] = kwargs
    res["model_kwargs"] = mkwargs

    sizes = compute_model_size(model)
    res["model"] = model
    res["configuration"] = config
    res["size"] = sizes[0]
    res["n_weights"] = sizes[1]
    res["task"] = task

    update = {}
    for k, v in res.items():
        if k.startswith(("inputs", "dynamic_shapes")) and isinstance(v, dict):
            update[k] = filter_out_unexpected_inputs(model, v, verbose=verbose)
    res.update(update)
    return res


def filter_out_unexpected_inputs(
    model: torch.nn.Module, kwargs: Dict[str, Any], verbose: int = 0
):
    """
    Removes input names in kwargs if no parameter names was found in ``model.forward``.
    """
    sig = inspect.signature(model.forward)
    allowed = set(sig.parameters)
    new_kwargs = {k: v for k, v in kwargs.items() if k in allowed}
    diff = set(kwargs) - set(new_kwargs)
    if diff and verbose:
        print(f"[filter_out_unexpected_inputs] removed {diff}")
    return new_kwargs


def compute_model_size(model: torch.nn.Module) -> Tuple[int, int]:
    """Returns the size of the models (weights only) and the number of the parameters."""
    param_size = 0
    nparams = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        nparams += param.nelement()
    return param_size, nparams
