from typing import Any, Dict, Tuple


def instantiate_specific_model(cls_model: type, config: Any) -> object:
    """
    Instantiates some model requiring some specific code.
    """
    if cls_model.__name__ == "CosmosTransformer3DModel":
        return instantiate_CosmosTransformer3DModel(cls_model, config)
    return None


def instantiate_CosmosTransformer3DModel(cls_model: type, config: Any) -> object:
    kwargs = dict(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        attention_head_dim=config.attention_head_dim,
        mlp_ratio=config.mlp_ratio,
        num_layers=config.num_layers,
        text_embed_dim=config.text_embed_dim,
        adaln_lora_dim=config.adaln_lora_dim,
        max_size=config.max_size,
        patch_size=config.patch_size,
        rope_scale=config.rope_scale,
        concat_padding_mask=config.concat_padding_mask,
        extra_pos_embed_type=config.extra_pos_embed_type,
    )
    return cls_model(**kwargs)


class SpecificConfig:
    """Creates a specific configuration for the loaded model."""

    def __init__(self, **kwargs):
        self._atts = set(kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def to_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self._atts if k != "_atts"}


def load_specific_model(
    model_id: str, verbose: int = 0, **kwargs
) -> Tuple[Any, str, SpecificConfig]:
    """
    Some models do not have any generic to be loaded.
    This functions

    :param model_id: model id
    :param verbose: verbosiy
    :param kwargs: additional parameters
    :return: the model, the task associated to it, a configuration
    """
    assert model_id in HANDLED_MODELS, (
        f"Unable to load model_id={model_id!r}, "
        f"no function is mapped to this id in {sorted(HANDLED_MODELS)}"
    )
    return HANDLED_MODELS[model_id](model_id, verbose=verbose, **kwargs)


def _load_bingsu_adetailer(model_id: str, verbose: int = 0) -> Tuple[Any, str, SpecificConfig]:
    """See `Bingsu/adetailer <https://huggingface.co/Bingsu/adetailer>`_."""
    from huggingface_hub import hf_hub_download
    from ultralytics import YOLO

    path = hf_hub_download("Bingsu/adetailer", "face_yolov8n.pt")
    model = YOLO(path)
    return (
        model,
        "object-detection",
        SpecificConfig(architecture=type(model), image_size=224, num_channels=3),
    )


HANDLED_MODELS = {"Bingsu/adetailer": _load_bingsu_adetailer}
