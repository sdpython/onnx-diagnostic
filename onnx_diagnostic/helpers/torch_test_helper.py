import contextlib
from collections.abc import Iterable
from typing import Any, Optional, Tuple, Union
import numpy as np
import torch
from .helper import string_type
from .cache_helper import (
    make_dynamic_cache,
    make_encoder_decoder_cache,
    make_sliding_window_cache,
    make_mamba_cache,
)


def _forward_(*args, _f=None, _context=None, **kwargs):
    assert _f is not None, "_f cannot be None"
    assert _context is not None, "_context cannot be None"
    print(
        f"---- stolen forward for class {_context['class_name']} "
        f"-- iteration {_context['iteration']}"
    )
    kws = dict(
        with_shape=_context.get("with_shape", False),
        with_min_max=_context.get("with_min_max", False),
    )
    if not hasattr(torch.compiler, "is_exporting") or not torch.compiler.is_exporting():
        # torch.compiler.is_exporting requires torch>=2.7
        print(f"  <- args={string_type(args, **kws)} --- kwargs={string_type(kwargs, **kws)}")
    res = _f(*args, **kwargs)
    if not hasattr(torch.compiler, "is_exporting") or not torch.compiler.is_exporting():
        print("  --")
        print(f"  -> {string_type(res, **kws)}")
        print(".")
    _context["iteration"] += 1
    return res


@contextlib.contextmanager
def steal_forward(model: torch.nn.Module, with_shape: bool = True, with_min_max: bool = False):
    """
    The necessary modification to steem forward method and prints out inputs
    and outputs. See example :ref:`l-plot-tiny-llm-export`.
    """
    context = dict(
        iteration=0,
        class_name=model.__class__.__name__,
        with_shape=with_shape,
        with_min_max=with_min_max,
    )
    keep_model_forward = model.forward
    model.forward = lambda *args, _f=keep_model_forward, _context=context, **kwargs: _forward_(
        *args, _f=_f, _context=_context, **kwargs
    )
    try:
        yield
    finally:
        model.forward = keep_model_forward


def is_torchdynamo_exporting() -> bool:
    """Tells if torch is exporting a model."""
    import torch

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


def to_numpy(tensor: "torch.Tensor"):  # noqa: F821
    """Converts a torch tensor to numy."""
    try:
        return tensor.numpy()
    except TypeError:
        # We try with ml_dtypes
        pass

    import ml_dtypes

    conv = {torch.bfloat16: ml_dtypes.bfloat16}
    assert tensor.dtype in conv, f"Unsupported type {tensor.dtype}, not in {conv}"
    return tensor.to(torch.float32).numpy().astype(conv[tensor.dtype])


def replace_string_by_dynamic(dynamic_shapes: Any) -> Any:
    """Replaces strings by ``torch.export.Dim.DYNAMIC``."""
    import torch

    if isinstance(dynamic_shapes, torch.export.dynamic_shapes._Dim):
        return dynamic_shapes
    if isinstance(dynamic_shapes, str):
        return torch.export.Dim.DYNAMIC
    if not dynamic_shapes:
        return dynamic_shapes
    if isinstance(dynamic_shapes, (tuple, list)):
        return type(dynamic_shapes)(replace_string_by_dynamic(i) for i in dynamic_shapes)
    if isinstance(dynamic_shapes, dict):
        return {k: replace_string_by_dynamic(v) for k, v in dynamic_shapes.items()}
    raise AssertionError(f"Unexpected type {type(dynamic_shapes)} for dynamic_shapes")


def dummy_llm(
    cls_name: Optional[str] = None,
    dynamic_shapes: bool = False,
) -> Union[
    Tuple[torch.nn.Module, Tuple[torch.Tensor, ...]],
    Tuple[torch.nn.Module, Tuple[torch.Tensor, ...], Any],
]:
    """
    Creates a dummy LLM for test purposes.

    :param cls_name: None for whole model or a piece of it
    :param dynamic_shapes: returns dynamic shapes as well

    .. runpython::
        :showcode:

        from onnx_diagnostic.helpers.torch_test_helper import dummy_llm
        print(dummy_llm())
    """

    class Embedding(torch.nn.Module):
        def __init__(self, vocab_size: int = 1024, embedding_dim: int = 16):
            super().__init__()
            self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
            self.pe = torch.nn.Embedding(vocab_size, embedding_dim)

        def forward(self, x):
            word_emb = self.embedding(x)
            word_pe = self.pe(x)
            return word_emb + word_pe

    class AttentionBlock(torch.nn.Module):

        def __init__(self, embedding_dim: int = 16, context_size: int = 256):
            super().__init__()
            self.query = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)
            self.key = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)
            self.value = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)
            # torch.nn.Buffer are not fully handled by symbolic tracing
            # Buffer(...)[:Prowy()] is not working
            self.mask = torch.nn.Parameter(
                torch.tril(
                    input=torch.ones(size=[context_size, context_size], dtype=torch.float)
                )
            )

        def forward(self, x):
            B, T, C = x.shape

            query = self.query(x)
            key = self.key(x)
            value = self.value(x)

            qk = query @ key.transpose(-2, -1) * C**-0.5
            attention = qk.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
            attention = torch.nn.functional.softmax(input=attention, dim=-1)

            out = attention @ value
            return out

    class MultiAttentionBlock(torch.nn.Module):

        def __init__(
            self, embedding_dim: int = 16, num_heads: int = 2, context_size: int = 256
        ):
            super().__init__()
            self.attention = torch.nn.ModuleList(
                modules=[AttentionBlock(embedding_dim, context_size) for _ in range(num_heads)]
            )
            self.linear = torch.nn.Linear(
                in_features=embedding_dim * num_heads, out_features=embedding_dim
            )

        def forward(self, x):
            out = torch.cat(tensors=[attention(x) for attention in self.attention], dim=-1)
            x = self.linear(out)
            return x

    class FeedForward(torch.nn.Module):

        def __init__(self, embedding_dim: int = 16, ff_dim: int = 128):
            super().__init__()
            self.linear_1 = torch.nn.Linear(embedding_dim, ff_dim)
            self.relu = torch.nn.ReLU()
            self.linear_2 = torch.nn.Linear(ff_dim, embedding_dim)

        def forward(self, x):
            x = self.linear_1(x)
            x = self.relu(x)
            x = self.linear_2(x)
            return x

    class DecoderLayer(torch.nn.Module):

        def __init__(
            self,
            embedding_dim: int = 16,
            num_heads: int = 2,
            context_size: int = 256,
            ff_dim: int = 128,
        ):
            super().__init__()
            self.attention = MultiAttentionBlock(embedding_dim, num_heads, context_size)
            self.feed_forward = FeedForward(embedding_dim, ff_dim)
            self.norm_1 = torch.nn.LayerNorm(normalized_shape=embedding_dim)
            self.norm_2 = torch.nn.LayerNorm(normalized_shape=embedding_dim)

        def forward(self, x):
            x_norm = self.norm_1(x)
            attention = self.attention(x_norm)
            attention = attention + x

            attention_norm = self.norm_2(attention)
            ff = self.feed_forward(attention_norm)
            ff = ff + attention

            return ff

    class LLM(torch.nn.Module):

        def __init__(
            self,
            vocab_size: int = 1024,
            embedding_dim: int = 16,
            num_heads: int = 2,
            context_size: int = 256,
            ff_dim: int = 128,
        ):
            super().__init__()
            self.embedding = Embedding(vocab_size, embedding_dim)
            self.decoder = DecoderLayer(embedding_dim, num_heads, context_size, ff_dim)

        def forward(self, input_ids):
            x = self.embedding(input_ids)
            y = self.decoder(x)
            return y

    if cls_name in (None, "LLM"):
        dec: torch.nn.Module = LLM()
        x = torch.randint(0, 1024, (2 if dynamic_shapes else 1, 30)).to(torch.int64)
        dec(x)
        if dynamic_shapes:
            dyn = {
                "input_ids": {
                    0: torch.export.Dim("batch", min=1, max=1024),
                    1: torch.export.Dim("length", min=1, max=255),
                }
            }
            return dec, (x,), dyn
        return dec, (x,)

    if cls_name == "DecoderLayer":
        LLM()(torch.randint(0, 1024, (2 if dynamic_shapes else 1, 30)).to(torch.int64))

        dec = DecoderLayer()
        x = Embedding()(
            torch.randint(0, 1024, (2 if dynamic_shapes else 1, 30)).to(torch.int64)
        )
        dec(x)
        if dynamic_shapes:
            dyn = {
                "x": {
                    0: torch.export.Dim("batch", min=1, max=1024),
                    1: torch.export.Dim("length", min=1, max=255),
                }
            }
            return dec, (x,), dyn
        return dec, (x,)

    if cls_name == "MultiAttentionBlock":
        dec = MultiAttentionBlock()
        x = torch.rand(2 if dynamic_shapes else 1, 30, 16).to(torch.float32)
        dec(x)
        if dynamic_shapes:
            dyn = {
                "x": {
                    0: torch.export.Dim("batch", min=1, max=1024),
                    1: torch.export.Dim("length", min=1, max=255),
                }
            }
            return dec, (x,), dyn
        return dec, (x,)

    if cls_name == "AttentionBlock":
        dec = AttentionBlock()
        x = torch.rand(2 if dynamic_shapes else 1, 30, 16).to(torch.float32)
        dec(x)
        if dynamic_shapes:
            dyn = {
                "x": {
                    0: torch.export.Dim("batch", min=1, max=1024),
                    1: torch.export.Dim("length", min=1, max=255),
                }
            }
            return dec, (x,), dyn
        return dec, (x,)

    raise NotImplementedError(f"cls_name={cls_name}")


def to_any(value: Any, to_value: Union[torch.dtype, torch.device]) -> Any:
    """
    Applies torch.to is applicable.
    Goes recursively.
    """
    if isinstance(value, (torch.nn.Module, torch.Tensor)):
        return value.to(to_value)
    if isinstance(value, list):
        return [to_any(t, to_value) for t in value]
    if isinstance(value, tuple):
        return tuple(to_any(t, to_value) for t in value)
    if isinstance(value, set):
        return {to_any(t, to_value) for t in value}
    if isinstance(value, dict):
        return {k: to_any(t, to_value) for k, t in value.items()}
    if hasattr(value, "to"):
        return value.to(to_value)
    if value.__class__.__name__ == "DynamicCache":
        return make_dynamic_cache(
            list(
                zip(
                    [t.to(to_value) for t in value.key_cache],
                    [t.to(to_value) for t in value.value_cache],
                )
            )
        )
    if value.__class__ in torch.utils._pytree.SUPPORTED_NODES:
        args, spec = torch.utils._pytree.tree_flatten(value)
        new_args = to_any(args, to_value)
        return torch.utils._pytree.tree_unflatten(new_args, spec)

    assert not isinstance(value, Iterable), f"Unsupported type {type(value)}"
    return value


def torch_deepcopy(value: Any) -> Any:
    """
    Makes a deepcopy.
    """
    if value is None:
        return None
    if isinstance(value, (int, float, str)):
        return value
    if isinstance(value, tuple):
        return tuple(torch_deepcopy(v) for v in value)
    if isinstance(value, list):
        return [torch_deepcopy(v) for v in value]
    if isinstance(value, set):
        return {torch_deepcopy(v) for v in value}
    if isinstance(value, dict):
        if type(value) is dict:
            return {k: torch_deepcopy(v) for k, v in value.items()}
        # for BaseModelOutput
        return value.__class__(**{k: torch_deepcopy(v) for k, v in value.items()})
    if isinstance(value, np.ndarray):
        return value.copy()
    if hasattr(value, "clone"):
        return value.clone()
    if value.__class__.__name__ == "DynamicCache":
        return make_dynamic_cache(
            torch_deepcopy(list(zip(value.key_cache, value.value_cache)))
        )
    if value.__class__.__name__ == "SlidingWindowCache":
        return make_sliding_window_cache(
            torch_deepcopy(list(zip(value.key_cache, value.value_cache)))
        )
    if value.__class__.__name__ == "EncoderDecoderCache":
        return make_encoder_decoder_cache(
            torch_deepcopy(value.self_attention_cache),
            torch_deepcopy(value.cross_attention_cache),
        )
    if value.__class__.__name__ == "MambaCache":
        return make_mamba_cache(list(zip(value.conv_states, value.ssm_states)))

    if value.__class__ in torch.utils._pytree.SUPPORTED_NODES:
        args, spec = torch.utils._pytree.tree_flatten(value)
        new_args = torch_deepcopy(args)
        return torch.utils._pytree.tree_unflatten(new_args, spec)

    # We should have a code using serialization, deserialization assuming a model
    # cannot be exported without them.
    raise NotImplementedError(f"torch_deepcopy not implemented for type {type(value)}")
