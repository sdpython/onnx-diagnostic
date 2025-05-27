import importlib.util
import os
import requests
import sys
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse
from onnx import helper, save_model, external_data_helper, ModelProto

CACHE_SUBDIR = "onnx-diagnostic"


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


def save_model_builder(self, out_dir: Optional[str] = "", verbose: int = 0) -> ModelProto:
    """
    Saves a model created by function :func:`create_model_builder`.
    If out_dir is empty or not specified, the function still returns the
    generated model.
    """
    if verbose:
        print(f"[save_model_builder] Saving ONNX model in {out_dir}")

    # Create ONNX model
    model = helper.make_model(
        opset_imports=[
            self.clear_field(
                helper.make_operatorsetid("", 21 if self.quant_attrs["use_qdq"] else 14),
                "domain",
            ),
            helper.make_operatorsetid("com.microsoft", 1),
        ],
        ir_version=7,
        producer_name="onnxruntime-genai",
        producer_version="0.0.0",
        graph=self.make_graph(
            name="main_graph",
            inputs=self.inputs,
            outputs=self.outputs,
            initializer=self.initializers,
            value_info=self.value_infos,
            nodes=self.nodes,
        ),
    )

    # Load external data into ONNX model
    external_data_helper.load_external_data_for_model(model, self.cache_dir)

    # Delete external data files on disk before re-saving
    for path in os.listdir(self.cache_dir):
        if path.endswith(".bin"):
            os.remove(os.path.join(self.cache_dir, path))

    # Delete temporary cache dir if empty
    # if len(os.listdir(self.cache_dir)) == 0:
    #    os.rmdir(self.cache_dir)

    # Quantize ONNX model to desired precision
    already_quantized_in_qdq_format = (
        self.quant_type is not None and self.quant_attrs["use_qdq"]
    )  # Skip quantizing `MatMul` in `DequantizeLinear --> Transpose --> MatMul` path
    if self.onnx_dtype == "int4" and not already_quantized_in_qdq_format:
        model = self.to_int4(model)

    # Save ONNX model with only one external data file and delete any existing duplicate copies
    if out_dir:
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

    if out_dir:
        location = os.path.basename(data_path)
        if os.path.exists(location):
            os.remove(location)
        if verbose:
            print(f"[save_model_builder] out_path={out_path!r}")
            print(f"[save_model_builder] location={location!r}")
        save_model(
            model,
            out_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=location,
            size_threshold=1024,
            convert_attribute=False,
        )
        return None
    return model


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
    download_model_builder_to_cache()
    builder = import_model_builder()
    io_dtype = builder.set_io_dtype(precision, execution_provider, extra_options)

    arch_map = {
        "ChatGLMForConditionalGeneration": builder.ChatGLMModel,
        "ChatGLMModel": builder.ChatGLMModel,
        "GemmaForCausalLM": builder.Gemma2Model,
        "Gemma3ForCausalLM": builder.Gemma3Model,
        "Gemma3ForConditionalGeneration": builder.Gemma3Model,
        "GraniteForCausalLM": builder.GraniteModel,
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

    onnx_model = cls(config, io_dtype, precision, execution_provider, cache_dir, extra_options)

    if post:
        post(onnx_model)
    _make_model(onnx_model, model, verbose=verbose)

    assert onnx_model.nodes, (
        f"No node in the model, io_dtype={io_dtype!r}, "
        f"precision={precision!r}, execution_provider={execution_provider!r}, "
        f"extra_options={extra_options!r}, cache_dir={cache_dir!r}, "
        f"\n-- config --\n{config}"
    )
    # onnx_model.make_genai_config(hf_name, extra_kwargs, output_dir)
    # onnx_model.save_processing(hf_name, extra_kwargs, output_dir)
    return onnx_model
