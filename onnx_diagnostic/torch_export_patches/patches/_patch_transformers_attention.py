from typing import Optional
import torch
import transformers
from .patch_helper import _has_transformers

patch_sdpa_is_causal = _has_transformers("4.99")


def common_eager_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: Optional[float] = None,
    dropout: float = 0.0,
    head_mask: Optional[torch.Tensor] = None,
    **kwargs,
):
    if scaling is None:
        scaling = query.size(-1) ** -0.5

    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        # PATCHED
        # The two following lines were added.
        if attention_mask is not None and attention_mask.ndim == 4:
            attention_mask = attention_mask[:, :, :, : key.shape[-2]]
        attn_weights = attn_weights + attention_mask

    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)

    if head_mask is not None:
        attn_weights = attn_weights * head_mask.view(1, -1, 1, 1)

    attn_weights = torch.nn.functional.dropout(
        attn_weights, p=dropout, training=module.training
    )
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def patched_sdpa_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """
    manual patch for function
    ``transformers.integrations.sdpa_attention.sdpa_attention_forward``
    """
    assert not kwargs.get("output_attentions", False), (
        "`sdpa` attention does not support `output_attentions=True`."
        " Please set your attention to `eager` if you want any of these features."
    )
    torch._check(
        query.shape[0] == key.shape[0] or query.shape[0] == 1,
        lambda: (
            f"broadcast issue query (1): {query.shape}, key: {key.shape}, "
            f"value: {value.shape}"
        ),
    )
    torch._check(
        key.shape[0] == value.shape[0] or key.shape[0] == 1,
        lambda: (
            f"broadcast issue query (2): {query.shape}, key: {key.shape}, "
            f"value: {value.shape}"
        ),
    )

    sdpa_kwargs = {}
    if hasattr(module, "num_key_value_groups"):
        if not transformers.integrations.sdpa_attention.use_gqa_in_sdpa(attention_mask, key):
            key = transformers.integrations.sdpa_attention.repeat_kv(
                key, module.num_key_value_groups
            )
            value = transformers.integrations.sdpa_attention.repeat_kv(
                value, module.num_key_value_groups
            )
        else:
            sdpa_kwargs = {"enable_gqa": True}

    if attention_mask is not None and attention_mask.ndim == 4:
        attention_mask = attention_mask[:, :, :, : key.shape[-2]]

    torch._check(
        attention_mask is None or attention_mask.shape[3] == key.shape[2],
        lambda: "Attention mask shape incompatible with key shape.",
    )

    if patch_sdpa_is_causal:
        # transformers>=4.55
        is_causal = is_causal if is_causal is not None else getattr(module, "is_causal", True)

        # PATCHED: remove the test query.shape[2] > 1
        # is_causal = query.shape[2] > 1 and attention_mask is None and is_causal
        # and we split the test to keep the minimum in torch.cond
        is_causal = attention_mask is None and is_causal

        if not is_causal:
            torch._check(query.shape[0] > 0)
            torch._check(query.shape[1] > 0)
            torch._check(query.shape[2] > 0)
            torch._check(query.shape[3] > 0)
            torch._check(key.shape[0] > 0)
            torch._check(key.shape[1] > 0)
            torch._check(key.shape[2] > 0)
            torch._check(key.shape[3] > 0)
            torch._check(value.shape[0] > 0)
            torch._check(value.shape[1] > 0)
            torch._check(value.shape[2] > 0)
            torch._check(value.shape[3] > 0)

            return (
                torch.nn.functional.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=attention_mask,
                    dropout_p=dropout,
                    scale=scaling,
                    is_causal=is_causal,
                    **sdpa_kwargs,
                )
                .transpose(1, 2)
                .contiguous(),
                None,
            )
    else:
        # transformers<4.55
        if is_causal is None and attention_mask is not None:
            is_causal = False
        if is_causal is not None:
            return (
                torch.nn.functional.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=attention_mask,
                    dropout_p=dropout,
                    scale=scaling,
                    is_causal=is_causal,
                    **sdpa_kwargs,
                )
                .transpose(1, 2)
                .contiguous(),
                None,
            )

    # To avoid the following errors:
    # is_causal=query.shape[2] > 1
    # TypeError: scaled_dot_product_attention(): argument 'is_causal' must be bool, not SymBool
    # is_causal=torch.tensor(query.shape[2] > 1)
    # TypeError: scaled_dot_product_attention(): argument 'is_causal' must be bool, not Tensor
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    attn_output = torch.cond(
        query.shape[2] > 1,  # distinction between prefill and decoding steps
        lambda query, key, value: torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=dropout,
            scale=scaling,
            is_causal=True,
            **sdpa_kwargs,
        ).contiguous(),
        lambda query, key, value: torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=dropout,
            scale=scaling,
            is_causal=False,
            **sdpa_kwargs,
        ).contiguous(),
        [query, key, value],
    )
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, None


def patched_model_bart_eager_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: Optional[float] = None,
    dropout: float = 0.0,
    head_mask: Optional[torch.Tensor] = None,
    **kwargs,
):
    """[patch:transformers.models.bart.modeling_bart.eager_attention_forward]"""
    return common_eager_attention_forward(
        module,
        query,
        key,
        value,
        attention_mask=attention_mask,
        scaling=scaling,
        dropout=dropout,
        head_mask=head_mask,
        **kwargs,
    )


def patched_modeling_marian_eager_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: Optional[float] = None,
    dropout: float = 0.0,
    head_mask: Optional[torch.Tensor] = None,
    **kwargs,
):
    """[patch:transformers.models.marian.modeling_marian.eager_attention_forward]"""
    return common_eager_attention_forward(
        module,
        query,
        key,
        value,
        attention_mask=attention_mask,
        scaling=scaling,
        dropout=dropout,
        head_mask=head_mask,
        **kwargs,
    )
