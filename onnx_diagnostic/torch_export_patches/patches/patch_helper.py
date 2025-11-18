import torch


def _has_transformers(version: str) -> bool:
    import packaging.version as pv
    import transformers

    return pv.Version(transformers.__version__) >= pv.Version(version)


def _is_torchdynamo_exporting() -> bool:
    """
    Tells if :epkg:`torch` is exporting a model.
    Relies on ``torch.compiler.is_exporting()``.
    """
    if not hasattr(torch.compiler, "is_exporting"):
        # torch.compiler.is_exporting requires torch>=2.7
        return False

    try:
        return torch.compiler.is_exporting()
    except Exception:
        try:
            import torch._dynamo as dynamo

            return dynamo.is_exporting()  # type: ignore
        except Exception:
            return False
