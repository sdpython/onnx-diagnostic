import functools
import importlib
import inspect
import re
from typing import Any, Callable, Dict, Optional, Tuple, Union
import torch
import transformers
from ...helpers.cache_helper import make_dynamic_cache, make_encoder_decoder_cache
from .hub_api import task_from_arch, get_pretrained_config


@functools.cache
def config_class_from_architecture(arch: str, exc: bool = False) -> Optional[type]:
    """
    Retrieves the configuration class for a given architecture.

    :param arch: architecture (clas name)
    :param exc: raise an exception if not found
    :return: type
    """
    cls = getattr(transformers, arch)
    mod_name = cls.__module__
    mod = importlib.import_module(mod_name)
    source = inspect.getsource(mod)
    reg = re.compile("config: ([A-Za-z0-9]+)")
    fall = reg.findall(source)
    if len(fall) == 0:
        assert not exc, (
            f"Unable to guess Configuration class name for arch={arch!r}, "
            f"module={mod_name!r}, no candidate, source is\n{source}"
        )
        return None
    unique = set(fall)
    assert len(unique) == 1, (
        f"Unable to guess Configuration class name for arch={arch!r}, "
        f"module={mod_name!r}, found={unique} (#{len(unique)}), "
        f"source is\n{source}"
    )
    cls_name = unique.pop()
    return getattr(transformers, cls_name)


def _update_config(config: Any, kwargs: Dict[str, Any]):
    for k, v in kwargs.items():
        if hasattr(config, k):
            setattr(config, k, v)


def reduce_model_config(config: Any, task: str) -> Dict[str, Any]:
    """Reduces a model size."""
    if task == "text-generation":
        check_hasattr(
            config,
            ("head_dim", ("hidden_size", "num_attention_heads")),
            "num_hidden_layers",
            ("num_key_value_heads", "num_attention_heads"),
            "intermediate_size",
            "hidden_size",
        )
        kwargs = dict(
            head_dim=getattr(
                config, "head_dim", config.hidden_size // config.num_attention_heads
            ),
            num_hidden_layers=min(config.num_hidden_layers, 2),
            num_key_value_heads=(
                config.num_key_value_heads
                if hasattr(config, "num_key_value_heads")
                else config.num_attention_heads
            ),
            intermediate_size=(
                min(config.intermediate_size, 24576 // 4)
                if config.intermediate_size % 4 == 0
                else config.intermediate_size
            ),
            hidden_size=(
                min(config.hidden_size, 3072 // 4)
                if config.hidden_size % 4 == 0
                else config.hidden_size
            ),
        )
    elif task == "image-classification":
        check_hasattr(config, ("num_hidden_layers", "hidden_sizes"))
        kwargs = dict(
            num_hidden_layers=(
                min(config.num_hidden_layers, 2)
                if hasattr(config, "num_hidden_layers")
                else len(config.hidden_sizes)
            )
        )
    elif task == "text2text-generation":
        kwargs = {}
        if hasattr(config, "num_decoder_layers"):
            config.num_decoder_layers = min(config.num_decoder_layers, 2)
        if hasattr(config, "num_hidden_layers"):
            config.num_hidden_layers = min(config.num_hidden_layers, 2)
    else:
        raise NotImplementedError(f"Input generation for task {task!r} not implemented yet.")

    for k, v in kwargs.items():
        setattr(config, k, v)
    return kwargs


def check_hasattr(config: Any, *args: Union[str, Tuple[Any, ...]]):
    """
    Checks the confiugation has all the attributes in ``args``.
    Raises an exception otherwise.
    """
    for a in args:
        assert isinstance(a, (str, tuple)), f"unexpected type {type(a)} in {args!r}"
        if isinstance(a, str):
            assert hasattr(config, a), f"Missing attribute {a!r} in\n{config}"
        elif isinstance(a, tuple):
            assert any(
                (isinstance(name, str) and hasattr(config, name))
                or all(hasattr(config, _) for _ in name)
                for name in a
            ), f"All attributes in {a!r} are missing from\n{config}"


def _pick(config, *atts):
    """Returns the first value found in the configuration."""
    for a in atts:
        if isinstance(a, str):
            if hasattr(config, a):
                return getattr(config, a)
        elif isinstance(a, tuple):
            if all(hasattr(config, _) for _ in a[1:]):
                return a[0]([getattr(config, _) for _ in a[1:]])
    raise AssertionError(f"Unable to find any of these {atts!r} in {config}")


def random_input_kwargs(config: Any, task: str) -> Tuple[Dict[str, Any], Callable]:
    """Inputs kwargs"""
    if task == "text-generation":
        check_hasattr(
            config,
            "vocab_size",
            "hidden_size",
            "num_attention_heads",
            ("num_key_value_heads", "num_attention_heads"),
            "intermediate_size",
            "hidden_size",
        )
        kwargs = dict(
            batch_size=2,
            sequence_length=30,
            sequence_length2=3,
            head_dim=getattr(
                config, "head_dim", config.hidden_size // config.num_attention_heads
            ),
            dummy_max_token_id=config.vocab_size - 1,
            num_hidden_layers=config.num_hidden_layers,
            num_key_value_heads=_pick(config, "num_key_value_heads", "num_attention_heads"),
            intermediate_size=config.intermediate_size,
            hidden_size=config.hidden_size,
        )
        fct = get_inputs_for_text_generation
    elif task == "text2text-generation":
        check_hasattr(
            config,
            "vocab_size",
            "hidden_size",
            "num_attention_heads",
            ("num_hidden_layers", "num_layers"),
            ("n_positions", "d_model"),
            (
                "num_key_value_heads",
                "num_heads",
                ("decoder_attention_heads", "encoder_attention_heads"),
            ),
        )
        kwargs = dict(
            batch_size=2,
            sequence_length=30,
            sequence_length2=3,
            head_dim=config.d_kv if hasattr(config, "d_kv") else 1,
            dummy_max_token_id=config.vocab_size - 1,
            num_hidden_layers=_pick(config, "num_hidden_layers", "num_layers"),
            num_key_value_heads=_pick(
                config,
                "num_key_value_heads",
                "num_heads",
                (sum, "encoder_attention_heads", "decoder_attention_heads"),
            ),
            encoder_dim=_pick(config, "n_positions", "d_model"),
        )
        fct = get_inputs_for_text2text_generation  # type: ignore
    elif task == "image-classification":
        check_hasattr(config, "image_size", "num_channels")
        if isinstance(config.image_size, int):
            kwargs = dict(
                batch_size=2,
                input_width=config.image_size,
                input_height=config.image_size,
                input_channels=config.num_channels,
            )
        else:
            kwargs = dict(
                batch_size=2,
                input_width=config.image_size[0],
                input_height=config.image_size[1],
                input_channels=config.num_channels,
            )
        fct = get_inputs_for_image_classification  # type: ignore
    else:
        raise NotImplementedError(f"Input generation for task {task!r} not implemented yet.")

    return kwargs, fct


def get_untrained_model_with_inputs(
    model_id: str,
    config: Optional[Any] = None,
    task: Optional[str] = "",
    inputs_kwargs: Optional[Dict[str, Any]] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    verbose: int = 0,
    dynamic_rope: Optional[bool] = None,
    same_as_pretrained: bool = False,
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
    :return: dictionary with a model, inputs, dynamic shapes, and the configuration

    Example:

    .. runpython::
        :showcode:

        import pprint
        from onnx_diagnostic.helpers import string_type
        from onnx_diagnostic.torch_models.hghub import get_untrained_model_with_inputs

        data = get_untrained_model_with_inputs("arnir0/Tiny-LLM", verbose=1)

        print("-- model size:", data["size"])
        print("-- number of parameters:", data["n_weights"])
        print("-- inputs:", string_type(data["inputs"], with_shape=True))
        print("-- dynamic shapes:", pprint.pformat(data["dynamic_shapes"]))
        print("-- configuration:", pprint.pformat(data["configuration"]))
    """
    if verbose:
        print(f"[get_untrained_model_with_inputs] model_id={model_id!r}")
    if config is None:
        config = get_pretrained_config(model_id)
    archs = config.architectures  # type: ignore
    assert archs is not None and len(archs) == 1, (
        f"Unable to determine the architecture for model {model_id!r}, "
        f"architectures={archs!r}"
    )
    arch = archs[0]
    if verbose:
        print(f"[get_untrained_model_with_inputs] architecture={arch!r}")
    config = get_pretrained_config(model_id)
    if verbose:
        print(f"[get_untrained_model_with_inputs] cls={config.__class__.__name__!r}")
    task = task_from_arch(arch)
    if verbose:
        print(f"[get_untrained_model_with_inputs] task={task!r}")

    # model kwagrs
    if dynamic_rope is not None:
        config.rope_scaling = (
            {"rope_type": "dynamic", "factor": 10.0} if dynamic_rope else None
        )

    # updating the configuration
    if not same_as_pretrained:
        mkwargs = reduce_model_config(config, task)
    else:
        mkwargs = {}
    if model_kwargs:
        for k, v in model_kwargs.items():
            setattr(config, k, v)
            mkwargs[k] = v
    # input kwargs
    kwargs, fct = random_input_kwargs(config, task)
    if inputs_kwargs:
        kwargs.update(inputs_kwargs)

    model = getattr(transformers, arch)(config)
    # This line is important. Some models may produce different
    # outputs even with the same inputs in training mode.
    model.eval()
    res = fct(model, config, **kwargs)
    res["input_kwargs"] = kwargs
    res["model_kwargs"] = mkwargs

    sizes = compute_model_size(model)
    res["model"] = model
    res["configuration"] = config
    res["size"] = sizes[0]
    res["n_weights"] = sizes[1]

    update = {}
    for k, v in res.items():
        if k.startswith(("inputs", "dynamic_shapes")) and isinstance(v, dict):
            update[k] = filter_out_unexpected_inputs(model, v)
    res.update(update)
    return res


def filter_out_unexpected_inputs(model: torch.nn.Module, kwargs: Dict[str, Any]):
    """
    Removes input names in kwargs if no parameter names was found in ``model.forward``.
    """
    sig = inspect.signature(model.forward)
    allowed = set(sig.parameters)
    kwargs = {k: v for k, v in kwargs.items() if k in allowed}
    return kwargs


def compute_model_size(model: torch.nn.Module) -> Tuple[int, int]:
    """Returns the size of the models (weights only) and the number of the parameters."""
    param_size = 0
    nparams = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        nparams += param.nelement()
    return param_size, nparams


def get_inputs_for_image_classification(
    model: torch.nn.Module,
    config: Optional[Any],
    input_width: int,
    input_height: int,
    input_channels: int,
    batch_size: int = 2,
    dynamic_rope: bool = False,
    **kwargs,
):
    """
    Generates inputs for task ``image-classification``.

    :param model: model to get the missing information
    :param config: configuration used to generate the model
    :param batch_size: batch size
    :param input_channel: input channel
    :param input_width: input width
    :param input_height: input height
    :param kwargs: to overwrite the configuration, example ``num_hidden_layers=1``
    :return: dictionary
    """
    assert isinstance(
        input_width, int
    ), f"Unexpected type for input_width {type(input_width)}{config}"
    assert isinstance(
        input_width, int
    ), f"Unexpected type for input_height {type(input_height)}{config}"

    shapes = {
        "pixel_values": {
            0: torch.export.Dim("batch", min=1, max=1024),
            2: torch.export.Dim("width", min=1, max=4096),
            3: torch.export.Dim("height", min=1, max=4096),
        },
    }
    inputs = dict(
        pixel_values=torch.randn(batch_size, input_channels, input_width, input_height).clamp(
            -1, 1
        ),
    )
    return dict(inputs=inputs, dynamic_shapes=shapes)


def get_inputs_for_text_generation(
    model: torch.nn.Module,
    config: Optional[Any],
    dummy_max_token_id: int,
    num_key_value_heads: int,
    num_hidden_layers: int,
    head_dim: int,
    batch_size: int = 2,
    sequence_length: int = 30,
    sequence_length2: int = 3,
    dynamic_rope: bool = False,
    **kwargs,
):
    """
    Generates input for task ``text-generation``.

    :param model: model to get the missing information
    :param config: configuration used to generate the model
    :param head_dim: last dimension of the cache
    :param dummy_max_token_id: dummy max token id
    :param batch_size: batch size
    :param sequence_length: sequence length
    :param sequence_length2: new sequence length
    :param dynamic_rope: use dynamic rope (see :class:`transformers.LlamaConfig`)
    :param kwargs: to overwrite the configuration, example ``num_hidden_layers=1``
    :return: dictionary
    """
    batch = torch.export.Dim("batch", min=1, max=1024)
    seq_length = torch.export.Dim("seq_length", min=1, max=4096)
    cache_length = torch.export.Dim("cache_length", min=1, max=4096)

    shapes = {
        "input_ids": {0: batch, 1: seq_length},
        "attention_mask": {
            0: batch,
            1: torch.export.Dim.DYNAMIC,  # cache_length + seq_length
        },
        "position_ids": {
            0: batch,
            1: torch.export.Dim.DYNAMIC,  # cache_length + seq_length
        },
        "past_key_values": [
            [{0: batch, 2: cache_length} for _ in range(num_hidden_layers)],
            [{0: batch, 2: cache_length} for _ in range(num_hidden_layers)],
        ],
    }
    inputs = dict(
        input_ids=torch.randint(0, dummy_max_token_id, (batch_size, sequence_length2)).to(
            torch.int64
        ),
        attention_mask=torch.ones((batch_size, sequence_length + sequence_length2)).to(
            torch.int64
        ),
        position_ids=torch.arange(sequence_length, sequence_length + sequence_length2)
        .to(torch.int64)
        .expand((batch_size, -1)),
        past_key_values=make_dynamic_cache(
            [
                (
                    torch.randn(batch_size, num_key_value_heads, sequence_length, head_dim),
                    torch.randn(batch_size, num_key_value_heads, sequence_length, head_dim),
                )
                for i in range(num_hidden_layers)
            ]
        ),
    )
    return dict(inputs=inputs, dynamic_shapes=shapes)


def get_inputs_for_text2text_generation(
    model: torch.nn.Module,
    config: Optional[Any],
    dummy_max_token_id: int,
    num_key_value_heads: int,
    num_hidden_layers: int,
    head_dim: int,
    encoder_dim: int,
    batch_size: int = 2,
    sequence_length: int = 30,
    sequence_length2: int = 3,
    **kwargs,
):
    """
    Generates input for task ``text2text-generation``.

    :param model: model to get the missing information
    :param config: configuration used to generate the model
    :param head_dim: last dimension of the cache
    :param dummy_max_token_id: dummy max token id
    :param batch_size: batch size
    :param encoder_dim: last dimension of encoder_last_hidden_state
    :param sequence_length: sequence length
    :param sequence_length2: new sequence length
    :param kwargs: to overwrite the configuration, example ``num_hidden_layers=1``
    :return: dictionary

    Stolen inputs for one model.

    ::

        cache_position:T7s1
        past_key_values:EncoderDecoderCache(
            self_attention_cache=DynamicCache(
                key_cache=#6[T1s1x8x1x64,...],
                value_cache=#6[T1s1x8x1x64,...]),
            cross_attention_cache=DynamicCache(
                key_cache=#6[T1s1x8x16x64,...],
                value_cache=#6[T1s1x8x16x64,...])),
        decoder_input_ids:T7s1x1,
        encoder_outputs:dict(last_hidden_state:T1s1x16x512)
    """
    batch = torch.export.Dim("batch", min=1, max=1024)
    seq_length = torch.export.Dim("seq_length", min=1, max=4096)
    cache_length = torch.export.Dim("cache_length", min=1, max=4096)
    cache_length2 = torch.export.Dim("cache_length2", min=1, max=4096)

    shapes = {
        "input_ids": {0: batch, 1: seq_length},
        "decoder_input_ids": {0: batch, 1: torch.export.Dim.DYNAMIC},
        "attention_mask": {0: batch, 1: torch.export.Dim.DYNAMIC},
        # "cache_position": {0: batch, 1: torch.export.Dim.DYNAMIC},
        "past_key_values": [
            [
                [{0: batch, 2: cache_length} for _ in range(num_hidden_layers)],
                [{0: batch, 2: cache_length} for _ in range(num_hidden_layers)],
            ],
            [
                [{0: batch, 2: cache_length2} for _ in range(num_hidden_layers)],
                [{0: batch, 2: cache_length2} for _ in range(num_hidden_layers)],
            ],
        ],
        # one these is selected based on the forward method signature
        # "encoder_last_hidden_state": {0: batch, 1: torch.export.Dim.DYNAMIC},
        # "encoder_outputs": {0: batch, 1: torch.export.Dim.DYNAMIC},
    }
    inputs = dict(
        input_ids=torch.randint(0, dummy_max_token_id, (batch_size, sequence_length)).to(
            torch.int64
        ),
        decoder_input_ids=torch.randint(
            0, dummy_max_token_id, (batch_size, sequence_length2)
        ).to(torch.int64),
        attention_mask=torch.ones((batch_size, sequence_length)).to(torch.int64),
        # cache_position=torch.arange(sequence_length, sequence_length + sequence_length2)
        # .to(torch.int64)
        # .expand((batch_size, -1)),
        past_key_values=make_encoder_decoder_cache(
            make_dynamic_cache(
                [
                    (
                        torch.randn(
                            batch_size, num_key_value_heads, sequence_length, head_dim
                        ),
                        torch.randn(
                            batch_size, num_key_value_heads, sequence_length, head_dim
                        ),
                    )
                    for i in range(num_hidden_layers)
                ]
            ),
            make_dynamic_cache(
                [
                    (
                        torch.randn(
                            batch_size, num_key_value_heads, sequence_length2, head_dim
                        ),
                        torch.randn(
                            batch_size, num_key_value_heads, sequence_length2, head_dim
                        ),
                    )
                    for i in range(num_hidden_layers)
                ]
            ),
        ),
        # one these is selected based on the forward method signature
        # encoder_last_hidden_state=torch.randn(batch_size, sequence_length2, encoder_dim),
        # encoder_outputs=torch.randn(batch_size, sequence_length2, encoder_dim),
    )
    return dict(inputs=inputs, dynamic_shapes=shapes)
