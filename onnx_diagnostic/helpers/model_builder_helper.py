import copy
import importlib.util
import os
import re
import requests
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse
from onnx import ModelProto, TensorProto, load as load_model

CACHE_SUBDIR = "onnx-diagnostic"


def download_model_builder_to_cache(
    url: str = "https://raw.githubusercontent.com/microsoft/onnxruntime-genai/refs/heads/main/src/python/py/models/builder.py",
) -> Path:
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

    builders = cache_dir / "builders"
    if not builders.exists():
        builders.mkdir(parents=True, exist_ok=True)

    for subfile in [
        "__init__.py",
        "base.py",
        "chatglm.py",
        "ernie.py",
        "gemma.py",
        "gptoss.py",
        "granite.py",
        "llama.py",
        "mistral.py",
        "nemotron.py",
        "olmo.py",
        "phi.py",
        "qwen.py",
        "smollm.py",
    ]:
        u = f"{'/'.join(url.split('/')[:-1])}/builders/{subfile}"
        response = requests.get(u)
        response.raise_for_status()
        with open(builders / subfile, "wb") as f:
            f.write(response.content)

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


def _make_model(self, model, verbose: int = 0):
    # Make inputs and outputs to ONNX model
    import torch

    self.make_inputs_and_outputs()

    # Make pre-processing nodes
    self.make_preprocessing_nodes()

    # Loop through model and map each module to ONNX/ORT ops
    self.layer_id = 0
    for module in model.modules():
        if (
            isinstance(module, torch.nn.Embedding)
            and module.weight.shape[0] == self.vocab_size
        ) or (hasattr(model, "embedding") and module == model.embedding):
            # Checks (Hugging Face logic) or (GGUF logic)
            if not self.exclude_embeds:
                # Embedding layer
                if verbose:
                    print("[_make_model] Reading embedding layer")
                self.make_embedding(module.weight.detach().cpu())
            else:
                # Exclude embedding layer from model
                self.layernorm_attrs["root_input"] = "inputs_embeds"
                self.layernorm_attrs["skip_input"] = "inputs_embeds"

        elif (
            module.__class__.__name__.endswith("DecoderLayer")
            or module.__class__.__name__.endswith("GLMBlock")
        ) and self.layer_id < self.num_layers:
            # Each decoder layer of model
            if verbose:
                print(f"[_make_model] Reading decoder layer {self.layer_id}")
            self.make_layer(self.layer_id, module)
            self.layer_id += 1

        elif self.layer_id == self.num_layers and self.has_final_norm(module, model):
            # SkipLayerNorm after last decoder layer (MatMul --> SkipLayerNorm)
            if verbose:
                print("[_make_model] Reading final norm")
            self.make_layernorm(
                self.layer_id,
                module,
                skip=True,
                simple=self.layernorm_attrs["simple"],
                location="final_norm",
            )

        elif (
            isinstance(module, torch.nn.Linear) and module.out_features == self.vocab_size
        ) or (hasattr(model, "lm_head") and module == model.lm_head):
            # Checks (Hugging Face logic) or (GGUF logic)
            if not self.exclude_lm_head:
                # Language modeling head (SkipLayerNorm --> logits)
                if verbose:
                    print("[_make_model] Reading LM head")
                self.make_lm_head(module)


def save_model_builder(
    self, out_dir: Optional[str] = "", verbose: int = 0
) -> Union[str, ModelProto]:
    """
    Saves a model created by function :func:`create_model_builder`.
    If out_dir is empty or not specified, the function still returns the
    generated model.
    """
    import onnx_ir

    if verbose:
        print(f"[save_model_builder] Saving ONNX model in {out_dir!r}")

    # Skip quantizing `MatMul` in `DequantizeLinear --> Transpose --> MatMul` path
    already_quantized_in_qdq_format = (
        self.quant_type is not None and self.quant_attrs["use_qdq"]
    )
    model = (
        self.to_int4()
        if self.onnx_dtype in {onnx_ir.DataType.INT4, onnx_ir.DataType.UINT4}
        and not already_quantized_in_qdq_format
        else self.model
    )
    model.graph.sort()
    if not out_dir:
        return onnx_ir.to_proto(model)

    out_path = os.path.join(out_dir, self.filename)
    data_path = os.path.join(out_dir, os.path.basename(out_path) + ".data")

    # Save ONNX model with only one external data file and delete any existing duplicate copies
    out_path = os.path.join(out_dir, self.filename)
    data_path = os.path.join(out_dir, os.path.basename(out_path) + ".data")
    if os.path.exists(out_path):
        if verbose:
            print(f"[save_model_builder] Overwriting {out_path!r}")
        os.remove(out_path)
    if os.path.exists(data_path):
        if verbose:
            print(f"[save_model_builder] Overwriting {data_path!r}")
        os.remove(data_path)

    onnx_ir.save(
        model,
        out_path,
        external_data=os.path.basename(data_path),
        size_threshold_bytes=2**10,
    )
    if verbose:
        print(f"[save_model_builder] saved in {out_dir!r}")

    return out_path


def create_model_builder(
    config: Any,
    model: "torch.nn.Module",  # noqa: F821
    cache_dir: str,
    precision: str = "fp32",
    execution_provider: str = "cpu",
    verbose: int = 0,
    **extra_options,
) -> "Model":  # noqa: F821
    """
    Creates a model based on a configuration.
    The onnx model is returned by function :func:`save_model_builder`.

    :param config: configuration
    :param cache_dir: cache directory
    :param precision: precision
    :param execution_provider: execution provider
    :param verbose: verbosity
    :param extra_options: extra options
    :return: model
    """
    assert cache_dir, "create_model_builder does not work without cache_dir."
    assert os.path.exists(cache_dir), f"cache_dir={cache_dir!r} does not exists"
    precision = {"float32": "fp32", "float16": "fp16", "bfloat16": "bfp16"}.get(
        precision, precision
    )
    download_model_builder_to_cache()
    builder = import_model_builder()
    io_dtype = builder.set_io_dtype(precision, execution_provider, extra_options)

    arch_map = {
        "ChatGLMForConditionalGeneration": builder.ChatGLMModel,
        "ChatGLMModel": builder.ChatGLMModel,
        "Ernie4_5_ForCausalLM": builder.ErnieModel,
        "GemmaForCausalLM": builder.Gemma2Model,
        "Gemma2ForCausalLM": builder.Gemma2Model,
        "Gemma3ForCausalLM": builder.Gemma3Model,
        "Gemma3ForConditionalGeneration": builder.Gemma3Model,
        "GraniteForCausalLM": builder.GraniteModel,
        "GptOssForCausalLM": builder.GPTOSSModel,
        "LlamaForCausalLM": builder.LlamaModel,
        "MistralForCausalLM": builder.MistralModel,
        "NemotronForCausalLM": builder.NemotronModel,
        "OlmoForCausalLM": builder.OLMoModel,
        "PhiForCausalLM": builder.PhiModel,
        "Phi3ForCausalLM": (
            lambda config, *args: (
                (
                    builder.Phi3MiniModel
                    if config.max_position_embeddings
                    == config.original_max_position_embeddings
                    else builder.Phi3MiniLongRoPEModel
                )(config, *args)
            )
        ),
        "PhiMoEForCausalLM": builder.Phi3MoELongRoPEModel,
        "Phi3SmallForCausalLM": (
            lambda config, *args: (
                (
                    builder.Phi3SmallModel
                    if config.max_position_embeddings
                    == config.original_max_position_embeddings
                    else builder.Phi3SmallLongRoPEModel
                )(config, *args)
            )
        ),
        "Phi3VForCausalLM": builder.Phi3VModel,
        "Phi4MMForCausalLM": builder.Phi4MMModel,
        "Qwen2ForCausalLM": builder.QwenModel,
        "Qwen3ForCausalLM": builder.Qwen3Model,
        "SmolLM3ForCausalLM": builder.SmolLM3Model,
    }

    assert config.architectures[0] in arch_map, (
        f"Unable find {config.architectures[0]!r} in the supported list "
        f"of architectures: {sorted(arch_map)}"
    )

    # Additional validations.
    post = None
    if config.architectures[0] in ("ChatGLMForConditionalGeneration", "ChatGLMModel"):
        # Quantized ChatGLM model has ChatGLMForConditionalGeneration
        # as architecture whereas HF model as the latter
        config.hidden_act = "swiglu"
    elif config.architectures[0] == "Gemma2ForCausalLM":
        assert precision == "bfp16", (
            f"architecture {config.architectures[0]!r} loses accuracy "
            f"with float16 precision, use bfp16."
        )
    elif config.architectures[0] == "Gemma3ForCausalLM":
        assert precision == "bfp16", (
            f"architecture {config.architectures[0]!r} loses accuracy "
            f"with float16 precision, use bfp16."
        )

        def _post(onnx_model):
            onnx_model.model_type = "gemma3_text"

        post = _post
    elif config.architectures[0] == "Gemma3ForConditionalGeneration":
        assert extra_options.get("exclude_embeds", False), (
            f"This is only generating the text component of architecture "
            f"{config.architectures[0]!r}. Set extra_options exclude_embeds=true."
        )
        assert precision == "bfp16", (
            f"architecture {config.architectures[0]!r} loses accuracy "
            f"with float16 precision, use bfp16."
        )
        text_config = config.text_config
        for key in text_config:
            if not hasattr(config, key):
                setattr(config, key, getattr(text_config, key))
    elif config.architectures[0] == "GptOssForCausalLM":
        if hasattr(config, "quantization_config"):
            delattr(config, "quantization_config")
    elif (
        config.architectures[0] == "PhiMoEForCausalLM"
        and config.max_position_embeddings != config.original_max_position_embeddings
    ):
        assert execution_provider == "cuda", (
            f"architecture {config.architectures[0]!r} works on 'cuda' "
            f"because `MoE` is only supported for CUDA in ONNX Runtime."
        )
        assert precision == "int4", f"architecture {config.architectures[0]!r} supports int4."
    elif config.architectures[0] == "Phi3VForCausalLM":
        assert extra_options.get("exclude_embeds", False), (
            f"This is only generating the text component of architecture "
            f"{config.architectures[0]!r}. Set extra_options exclude_embeds=true."
        )
    elif config.architectures[0] == "Phi4MMForCausalLM":
        assert extra_options.get("exclude_embeds", False), (
            f"This is only generating the text component of architecture "
            f"{config.architectures[0]!r}. Set extra_options exclude_embeds=true."
        )

    cls = arch_map[config.architectures[0]]

    # ModelBuilder does not like None values for some parameters.
    remove = set()
    for c in ["head_dim"]:
        if hasattr(config, c) and getattr(config, c) is None:
            remove.add(c)
    for c in remove:
        delattr(config, c)

    convert = {
        "fp32": TensorProto.FLOAT,
        "fp16": TensorProto.FLOAT16,
        "bfp16": TensorProto.BFLOAT16,
    }
    assert (
        precision in convert
    ), f"Unexpected value for precision={precision!r}, should be in {convert}"
    onnx_model = cls(
        config, io_dtype, convert[precision], execution_provider, cache_dir, extra_options
    )

    if post:
        post(onnx_model)
    _make_model(onnx_model, model, verbose=verbose)

    assert onnx_model.model, (
        f"No node in the model, io_dtype={io_dtype!r}, "
        f"precision={precision!r}, execution_provider={execution_provider!r}, "
        f"extra_options={extra_options!r}, cache_dir={cache_dir!r}, "
        f"\n-- config --\n{config}"
    )
    # onnx_model.make_genai_config(hf_name, extra_kwargs, output_dir)
    # onnx_model.save_processing(hf_name, extra_kwargs, output_dir)
    return onnx_model


def find_names_pattern(names: List[str]) -> str:
    """
    Finds a repeatable patterns in a list of names.
    It tries to locate the figures.

    .. runpython::
        :showcode:

        from onnx_diagnostic.helpers.model_builder_helper import find_names_pattern
        pattern = find_names_pattern(["past_key_values_key_0", "past_key_values_key_1"])
        print(pattern)
    """
    patterns = [re.sub(r"(\d+)", r"%d", t) for t in names]
    unique = set(patterns)
    assert (
        len(unique) == 1
    ), f"Unable to guess a pattern from {names} which led to the unique patterns {unique}"
    return patterns[0]


def make_genai_config(
    config,
    onnx_filename: str,
) -> Dict:
    """
    Creates genai config file for a model.

    :param config: configuration from transformers
    :param onnx_filename: onnx configuration
    :return: configuration
    """
    onx = load_model(onnx_filename, load_external_data=False)
    config = copy.deepcopy(config)
    defaults = {
        "bos_token_id": None,
        "do_sample": False,
        "eos_token_id": None,
        "pad_token_id": None,
        "temperature": 1.0,
        "top_k": 50,
        "top_p": 1.0,
    }
    for key, default_val in defaults.items():
        if not hasattr(config, key):
            setattr(config, key, default_val)

    bos_token_id = (
        config.bos_token_id
        if hasattr(config, "bos_token_id") and config.bos_token_id is not None
        else 1
    )
    eos_token_id = config.eos_token_id
    pad_token_id = (
        config.pad_token_id
        if hasattr(config, "pad_token_id") and config.pad_token_id is not None
        else (
            config.eos_token_id[0]
            if isinstance(config.eos_token_id, list)
            else config.eos_token_id
        )
    )
    input_names = [i.name for i in onx.graph.input]
    output_names = [i.name for i in onx.graph.output]
    past_key_values = [s for s in input_names if s.startswith("past_key_value")]
    first = [i for i in onx.graph.input if i.name == past_key_values[0]][0]  # noqa: RUF015
    shape = tuple(d.dim_value or d.dim_param for d in first.type.tensor_type.shape.dim)
    return {
        "model": {
            "bos_token_id": bos_token_id,
            "context_length": config.max_position_embeddings,
            "decoder": {
                "session_options": {
                    "log_id": "onnxruntime-genai",
                    "provider_options": [],
                },
                "filename": os.path.split(onnx_filename)[-1],
                "head_size": shape[-1],
                "hidden_size": config.hidden_size,
                "inputs": {
                    "input_ids": input_names[0],
                    "attention_mask": input_names[1],
                    "past_key_names": find_names_pattern(input_names[2::2]),
                    "past_value_names": find_names_pattern(input_names[3::2]),
                },
                "outputs": {
                    "logits": output_names[0],
                    "present_key_names": find_names_pattern(output_names[1::2]),
                    "present_value_names": find_names_pattern(output_names[2::2]),
                },
                "num_attention_heads": config.num_attention_heads,
                "num_hidden_layers": len(past_key_values) // 2,
                "num_key_value_heads": shape[1],
            },
            "eos_token_id": eos_token_id,
            "pad_token_id": pad_token_id,
            "type": config.model_type,
            # if "For" in self.model_type else len(self.model_type)].lower(),
            "vocab_size": config.vocab_size,
        },
        "search": {
            "diversity_penalty": (
                config.diversity_penalty if hasattr(config, "diversity_penalty") else 0.0
            ),
            "do_sample": config.do_sample if hasattr(config, "do_sample") else False,
            "early_stopping": True,
            "length_penalty": (
                config.length_penalty if hasattr(config, "length_penalty") else 1.0
            ),
            "max_length": config.max_position_embeddings,
            "min_length": 0,
            "no_repeat_ngram_size": (
                config.no_repeat_ngram_size if hasattr(config, "no_repeat_ngram_size") else 0
            ),
            "num_beams": config.num_beams if hasattr(config, "num_beams") else 1,
            "num_return_sequences": (
                config.num_return_sequences if hasattr(config, "num_return_sequences") else 1
            ),
            "past_present_share_buffer": False,
            "repetition_penalty": (
                config.repetition_penalty if hasattr(config, "repetition_penalty") else 1.0
            ),
            "temperature": config.temperature if hasattr(config, "temperature") else 1.0,
            "top_k": config.top_k if hasattr(config, "top_k") else 50,
            "top_p": config.top_p if hasattr(config, "top_p") else 1.0,
        },
    }
