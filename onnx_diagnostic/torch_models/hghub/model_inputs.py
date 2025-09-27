import copy
import inspect
import os
import pprint
import time
from typing import Any, Dict, Optional, Tuple
import torch
import transformers
from ...helpers.config_helper import update_config, build_diff_config
from ...tasks import reduce_model_config, random_input_kwargs
from .hub_api import (
    task_from_arch,
    task_from_id,
    get_pretrained_config,
    download_code_modelid,
    architecture_from_config,
    find_package_source,
)
from .model_specific import HANDLED_MODELS, load_specific_model, instantiate_specific_model


def _code_needing_rewriting(model: Any) -> Any:
    from onnx_diagnostic.torch_export_patches.patch_module_helper import code_needing_rewriting

    return code_needing_rewriting(model)


def get_untrained_model_with_inputs(
    model_id: str,
    config: Optional[Any] = None,
    task: Optional[str] = "",
    inputs_kwargs: Optional[Dict[str, Any]] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    verbose: int = 0,
    dynamic_rope: Optional[bool] = None,
    use_pretrained: bool = False,
    same_as_pretrained: bool = False,
    use_preinstalled: bool = True,
    add_second_input: int = 1,
    subfolder: Optional[str] = None,
    use_only_preinstalled: bool = False,
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
    :param use_pretrained: download the pretrained weights as well
    :param use_preinstalled: use preinstalled configurations
    :param add_second_input: provides a second inputs to check a model
        supports different shapes
    :param subfolder: subfolder to use for this model id
    :param use_only_preinstalled: use only preinstalled version
    :return: dictionary with a model, inputs, dynamic shapes, and the configuration,
        some necessary rewriting as well

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
    assert not use_preinstalled or not use_only_preinstalled, (
        f"model_id={model_id!r}, preinstalled model is only available "
        f"if use_only_preinstalled is False."
    )
    if verbose:
        print(f"[get_untrained_model_with_inputs] model_id={model_id!r}")
        if use_preinstalled:
            print(f"[get_untrained_model_with_inputs] use preinstalled {model_id!r}")
    if config is None:
        config = get_pretrained_config(
            model_id,
            use_preinstalled=use_preinstalled,
            use_only_preinstalled=use_only_preinstalled,
            subfolder=subfolder,
            **(model_kwargs or {}),
        )

    model, task, mkwargs, diff_config = None, None, {}, None
    if use_pretrained and same_as_pretrained:
        if model_id in HANDLED_MODELS:
            model, task, config = load_specific_model(model_id, verbose=verbose)

    if model is None:
        arch = architecture_from_config(config)
        if arch is None:
            task = task_from_id(model_id, subfolder=subfolder)
        assert task is not None or arch is not None, (
            f"Unable to determine the architecture for model {model_id!r}, "
            f"archs={arch!r}, conf={config}"
        )
        if verbose:
            print(f"[get_untrained_model_with_inputs] architecture={arch!r}")
            print(f"[get_untrained_model_with_inputs] cls={config.__class__.__name__!r}")
        if task is None:
            task = task_from_arch(arch, model_id=model_id, subfolder=subfolder)
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
        config0 = copy.deepcopy(config)
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
        try:
            diff_config = build_diff_config(config0, config)
        except (ValueError, AttributeError, TypeError) as e:
            diff_config = f"DIFF CONFIG ERROR {e}"
        if verbose:
            if diff_config:
                print("[get_untrained_model_with_inputs] -- updated config")
                pprint.pprint(diff_config)
                print("[get_untrained_model_with_inputs] --")

        # SDPA
        if model_kwargs and "attn_implementation" in model_kwargs:
            if hasattr(config, "_attn_implementation_autoset"):
                config._attn_implementation_autoset = False
            config._attn_implementation = model_kwargs["attn_implementation"]  # type: ignore[union-attr]
            if verbose:
                print(
                    f"[get_untrained_model_with_inputs] config._attn_implementation="
                    f"{config._attn_implementation!r}"  # type: ignore[union-attr]
                )
        elif verbose:
            print(
                f"[get_untrained_model_with_inputs] default config._attn_implementation="
                f"{getattr(config, '_attn_implementation', '?')!r}"  # type: ignore[union-attr]
            )

        if find_package_source(config) == "diffusers":
            import diffusers

            package_source = diffusers
        else:
            package_source = transformers

        if verbose:
            print(
                f"[get_untrained_model_with_inputs] package_source={package_source.__name__} é"
                f"from {package_source.__file__}"
            )
        if use_pretrained:
            begin = time.perf_counter()
            if verbose:
                print(
                    f"[get_untrained_model_with_inputs] pretrained model_id {model_id!r}, "
                    f"subfolder={subfolder!r}"
                )
            model = transformers.AutoModel.from_pretrained(
                model_id, subfolder=subfolder or "", trust_remote_code=True, **mkwargs
            )
            if verbose:
                print(
                    f"[get_untrained_model_with_inputs] -- done(1) in "
                    f"{time.perf_counter() - begin}s"
                )
        else:
            begin = time.perf_counter()
            if verbose:
                print(
                    f"[get_untrained_model_with_inputs] instantiate model_id {model_id!r}, "
                    f"subfolder={subfolder!r}"
                )
            if arch is not None:
                try:
                    cls_model = getattr(package_source, arch)
                except AttributeError as e:
                    # The code of the models is not in transformers but in the
                    # repository of the model. We need to download it.
                    pyfiles = download_code_modelid(model_id, verbose=verbose)
                    if pyfiles:
                        if "." in arch:
                            cls_name = arch
                        else:
                            modeling = [_ for _ in pyfiles if "/modeling_" in _]
                            assert len(modeling) == 1, (
                                f"Unable to guess the main file implemented class "
                                f"{arch!r} from {pyfiles}, found={modeling}."
                            )
                            last_name = os.path.splitext(os.path.split(modeling[0])[-1])[0]
                            cls_name = f"{last_name}.{arch}"
                        if verbose:
                            print(
                                f"[get_untrained_model_with_inputs] "
                                f"custom code for {cls_name!r}"
                            )
                            print(
                                f"[get_untrained_model_with_inputs] from folder "
                                f"{os.path.split(pyfiles[0])[0]!r}"
                            )
                        cls_model = (
                            transformers.dynamic_module_utils.get_class_from_dynamic_module(
                                cls_name,
                                pretrained_model_name_or_path=os.path.split(pyfiles[0])[0],
                            )
                        )
                    else:
                        raise AttributeError(
                            f"Unable to find class 'tranformers.{arch}'. "
                            f"The code needs to be downloaded, config="
                            f"\n{pprint.pformat(config)}."
                        ) from e
            else:
                assert same_as_pretrained and use_pretrained, (
                    f"Model {model_id!r} cannot be built, the model cannot be built. "
                    f"It must be downloaded. Use same_as_pretrained=True "
                    f"and use_pretrained=True, arch={arch!r}, config={config}"
                )
            if verbose:
                print(
                    f"[get_untrained_model_with_inputs] -- done(2) in "
                    f"{time.perf_counter() - begin}s"
                )

            seed = int(os.environ.get("SEED", "17"))
            torch.manual_seed(seed)

            if verbose:
                begin = time.perf_counter()
                print(
                    f"[get_untrained_model_with_inputs] "
                    f"instantiate_specific_model {cls_model}"
                )

            model = instantiate_specific_model(cls_model, config)

            if verbose:
                print(
                    f"[get_untrained_model_with_inputs] -- done(3) in "
                    f"{time.perf_counter() - begin}s (model is {type(model)})"
                )

            if model is None:

                if verbose:
                    print(
                        f"[get_untrained_model_with_inputs] "
                        f"instantiate_specific_model(2) {cls_model}"
                    )

                try:
                    if type(config) is dict:
                        model = cls_model(**config)
                    else:
                        model = cls_model(config)
                except RuntimeError as e:
                    raise RuntimeError(
                        f"Unable to instantiate class {cls_model.__name__} with\n{config}"
                    ) from e

                if verbose:
                    print(
                        f"[get_untrained_model_with_inputs] -- done(4) in "
                        f"{time.perf_counter() - begin}s (model is {type(model)})"
                    )

    # input kwargs
    seed = int(os.environ.get("SEED", "17")) + 1
    torch.manual_seed(seed)
    kwargs, fct = random_input_kwargs(config, task)  # type: ignore[arg-type]
    if verbose:
        print(f"[get_untrained_model_with_inputs] use fct={fct}")
        if os.environ.get("PRINT_CONFIG") in (1, "1"):
            print(f"-- input kwargs for task {task!r}")
            pprint.pprint(kwargs)
    if inputs_kwargs:
        kwargs.update(inputs_kwargs)

    # This line is important. Some models may produce different
    # outputs even with the same inputs in training mode.
    model.eval()  # type: ignore[union-attr]
    res = fct(model, config, add_second_input=add_second_input, **kwargs)

    res["input_kwargs"] = kwargs
    res["model_kwargs"] = mkwargs
    if diff_config is not None:
        res["dump_info"] = dict(config_diff=diff_config)

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

    rewrite = _code_needing_rewriting(model.__class__.__name__)
    if rewrite:
        res["rewrite"] = rewrite
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
