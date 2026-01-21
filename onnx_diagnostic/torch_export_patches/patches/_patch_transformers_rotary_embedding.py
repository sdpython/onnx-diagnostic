from functools import wraps
from typing import Callable, Optional, Tuple
import packaging.version as pv
import torch
import transformers


def patched__compute_dynamic_ntk_parameters(
    config: Optional[transformers.PretrainedConfig] = None,
    device: Optional[torch.device] = None,
    seq_len: Optional[int] = None,
    **rope_kwargs,
) -> Tuple[torch.Tensor, float]:
    """
    manual patch:
    ``[patch:transformers.modeling_rope_utils._compute_dynamic_ntk_parameters]``

    Computes the inverse frequencies with NTK scaling.
    Credits to the Reddit users /u/bloc97 and /u/emozilla

    Args:
        config ([`~transformers.PretrainedConfig`]):
            The model configuration.
        device (`torch.device`):
            The device to use for initialization of the inverse frequencies.
        seq_len (`int`, *optional*):
            The current sequence length,
            used to update the dynamic RoPE at inference time.
        rope_kwargs (`Dict`, *optional*):
            BC compatibility with the previous
            RoPE class instantiation, will be removed in v4.45.

    Returns:
        Tuple of (`torch.Tensor`, `float`),
        containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the
        omputed cos/sin (unused in this type of RoPE).
    """
    if config is not None and len(rope_kwargs) > 0:
        raise ValueError(
            "Unexpected arguments: `**rope_kwargs` and `config` are mutually exclusive in "
            f"`_compute_dynamic_ntk_parameters`, got "
            f"`rope_kwargs`={rope_kwargs} and `config`={config}"
        )
    if len(rope_kwargs) > 0:
        base = rope_kwargs["base"]
        dim = rope_kwargs["dim"]
        max_position_embeddings = rope_kwargs["max_position_embeddings"]
        factor = rope_kwargs["factor"]
    elif config is not None:
        if hasattr(config, "rope_theta"):
            # transformers<5
            base = config.rope_theta
            partial_rotary_factor = (
                config.partial_rotary_factor
                if hasattr(config, "partial_rotary_factor")
                else 1.0
            )
            factor = config.rope_scaling["factor"]
        else:
            print("-----")
            print(config)
            base = config.rope_parameters["rope_theta"]
            partial_rotary_factor = config.rope_parameters["partial_rotary_factor"]
            factor = config.rope_parameters["factor"]
        head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        dim = int(head_dim * partial_rotary_factor)
        max_position_embeddings = config.max_position_embeddings

    attention_factor = 1.0  # Unused in this type of RoPE

    # seq_len: default to max_position_embeddings, e.g. at init time
    # seq_len = seq_len if seq_len is not None and
    #       seq_len > max_position_embeddings else max_position_embeddings
    if seq_len is None:
        seq_len = max_position_embeddings
    else:
        # PATCHED: remove the line using max
        torch._check(isinstance(seq_len, torch.Tensor))
        seq_len = torch.maximum(
            seq_len,
            torch.tensor(max_position_embeddings, dtype=seq_len.dtype, device=seq_len.device),
        )

    # Compute the inverse frequencies
    base = base * ((factor * seq_len / max_position_embeddings) - (factor - 1)) ** (
        dim / (dim - 2)
    )
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float)
            / dim
        )
    )
    return inv_freq, attention_factor


def _get_rope_init_fn(self, layer_type=None) -> Callable:
    if hasattr(self, "rope_init_fn"):
        # transformers<=5.0
        rope_init_fn = (
            patched__compute_dynamic_ntk_parameters
            if self.rope_init_fn
            is transformers.modeling_rope_utils._compute_dynamic_ntk_parameters
            else self.rope_init_fn
        )
        return rope_init_fn

    rope_type = self.rope_type if layer_type is None else self.rope_type[layer_type]
    rope_init_fn = self.compute_default_rope_parameters
    if rope_type != "default":
        rope_init_fn = transformers.modeling_rope_utils.ROPE_INIT_FUNCTIONS[self.rope_type]
    if rope_init_fn is transformers.modeling_rope_utils._compute_dynamic_ntk_parameters:
        return patched__compute_dynamic_ntk_parameters
    return rope_init_fn


def patched_dynamic_rope_update(rope_forward):
    """manual patch: ``[patch:transformers.modeling_rope_utils.dynamic_rope_update]``

    ``rope_type`` is determined in the constructor of class
    :class:`transformers.models.phi3.modeling_phi3.Phi3RotaryEmbedding`.

    .. code-block:: python

        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get(
                "rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"

    The original code of the patched function:

    .. code-block:: python

        def dynamic_rope_update(rope_forward):
            def longrope_frequency_update(self, position_ids, device):
                seq_len = torch.max(position_ids) + 1
                if hasattr(self.config, "original_max_position_embeddings"):
                    original_max_position_embeddings =
                        self.config.original_max_position_embeddings
                else:
                    original_max_position_embeddings =
                        self.config.max_position_embeddings
                if seq_len > original_max_position_embeddings:
                    if not hasattr(self, "long_inv_freq"):
                        self.long_inv_freq, _ = self.rope_init_fn(
                            self.config, device, seq_len=original_max_position_embeddings + 1
                        )
                    self.register_buffer("inv_freq", self.long_inv_freq, persistent=False)
                else:
                    self.original_inv_freq = self.original_inv_freq.to(device)
                    self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)

            def dynamic_frequency_update(self, position_ids, device):
                seq_len = torch.max(position_ids) + 1
                if seq_len > self.max_seq_len_cached:  # growth
                    inv_freq, self.attention_scaling = self.rope_init_fn(
                        self.config, device, seq_len=seq_len)
                    self.register_buffer("inv_freq", inv_freq, persistent=False)
                    self.max_seq_len_cached = seq_len

                if seq_len < self.original_max_seq_len and
                        self.max_seq_len_cached > self.original_max_seq_len:
                    self.original_inv_freq = self.original_inv_freq.to(device)
                    self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
                    self.max_seq_len_cached = self.original_max_seq_len

            @wraps(rope_forward)
            def wrapper(self, x, position_ids):
                if "dynamic" in self.rope_type:
                    dynamic_frequency_update(self, position_ids, device=x.device)
                elif self.rope_type == "longrope":
                    longrope_frequency_update(self, position_ids, device=x.device)
                return rope_forward(self, x, position_ids)

            return wrapper

    """

    def longrope_frequency_update(self, position_ids, device, layer_type=None):
        # It is no use to patch the function after the model is created
        # as rope_init_fn is an attribute set to one function when the model
        # is created and when no patch is applied yet.
        # So we select the patched version here.
        rope_init_fn = _get_rope_init_fn(self, layer_type=layer_type)
        seq_len = torch.max(position_ids) + 1
        if hasattr(self.config, "original_max_position_embeddings"):
            original_max_position_embeddings = self.config.original_max_position_embeddings
        else:
            original_max_position_embeddings = self.config.max_position_embeddings

        if layer_type is None:
            # rope_type = self.rope_type
            original_inv_freq = self.original_inv_freq
            prefix = ""
        else:
            # rope_type = self.rope_type[layer_type]
            original_inv_freq = getattr(self, f"{layer_type}_original_inv_freq")
            prefix = f"{layer_type}_"

        # At export time, seq_len is unknown.
        long_inv_freq, _ = rope_init_fn(
            self.config, device, seq_len=original_max_position_embeddings + 1
        )
        original_inv_freq = self.original_inv_freq.to(device)

        # PATCHED: uses torch.cond instead of a test
        cond = (seq_len > original_max_position_embeddings).item()
        inv_freq = torch.cond(
            cond,
            (lambda x, y: x.clone()),
            (lambda x, y: y.clone()),
            [long_inv_freq.to(original_inv_freq.dtype), original_inv_freq],
        )
        setattr(self, f"{prefix}inv_freq", inv_freq)
        # if seq_len > original_max_position_embeddings:
        #    self.inv_freq = self.long_inv_freq
        # else:
        #    self.inv_freq = self.original_inv_freq

    def dynamic_frequency_update(self, position_ids, device, layer_type=None):
        # constructor:
        # - self.max_seq_len_cached = config.max_position_embeddings
        # - self.original_max_seq_len = config.max_position_embeddings
        # - inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)

        # It is no use to patch the function after the model is created
        # as rope_init_fn is an attribute set to one function when the model
        # is created and when no patch is applied yet.
        # So we select the patched version here.
        rope_init_fn = _get_rope_init_fn(self, layer_type=layer_type)

        # This behaviour is difficult to translate.
        # The sequence always grows.
        # The test should always True.
        # So:  self.max_seq_len_cached = max(self.max_seq_len_cached, seq_len) --> seq_len
        #
        # if seq_len > self.max_seq_len_cached:  # growth
        #    inv_freq, self.attention_scaling = self.rope_init_fn(
        #        self.config, device, seq_len=seq_len
        #    )
        #    self.register_buffer("inv_freq", inv_freq, persistent=False)
        #    self.max_seq_len_cached = seq_len
        #
        # So we should not need what follows.
        #
        # cond = (seq_len > self.max_seq_len_cached).item()
        # self.attention_scaling = torch.cond(
        #    cond,
        #    (lambda x, y: x.clone()),
        #    (lambda x, y: y.clone()),
        #    [attention_scaling, self.attention_scaling],
        # )

        seq_len = torch.max(position_ids) + 1
        long_inv_freq, self.attention_scaling = rope_init_fn(
            self.config, device, seq_len=seq_len
        )

        if layer_type is None:
            # rope_type = self.rope_type
            # max_seq_len_cached = self.max_seq_len_cached
            original_inv_freq = self.original_inv_freq
            prefix = ""
        else:
            # rope_type = self.rope_type[layer_type]
            # max_seq_len_cached = getattr(
            #     self, f"{layer_type}_max_seq_len_cached", self.max_seq_len_cached
            # )
            original_inv_freq = getattr(self, f"{layer_type}_original_inv_freq")
            prefix = f"{layer_type}_"

        # Second test to translate.
        # Let's keep in mind, self.max_seq_len_cached = seq_len is likely to be True.
        # But in that case the following condition is a way to restore the original cache.

        # if (
        #    seq_len < self.original_max_seq_len
        #    and self.max_seq_len_cached > self.original_max_seq_len
        # ):
        #    self.original_inv_freq = self.original_inv_freq.to(device)
        #    self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
        #    self.max_seq_len_cached = self.original_max_seq_len

        original_inv_freq = self.original_inv_freq.to(device)
        cond = (seq_len >= self.original_max_seq_len).item()
        # PATCHED: uses torch.cond instead of a test
        inv_freq = torch.cond(
            cond,
            (lambda x, y: x.clone()),
            (lambda x, y: y.clone()),
            [long_inv_freq.to(original_inv_freq.dtype), original_inv_freq],
        )
        setattr(self, f"{prefix}inv_freq", inv_freq)

    @wraps(rope_forward)
    def wrapper(self, x, position_ids, layer_type=None):
        if layer_type is None:
            if "dynamic" in self.rope_type:
                dynamic_frequency_update(self, position_ids, device=x.device)
            elif self.rope_type == "longrope":
                longrope_frequency_update(self, position_ids, device=x.device)
            return rope_forward(self, x, position_ids)

        if "dynamic" in self.rope_type:
            dynamic_frequency_update(
                self, position_ids, device=x.device, layer_type=layer_type
            )
        elif self.rope_type == "longrope":
            longrope_frequency_update(
                self, position_ids, device=x.device, layer_type=layer_type
            )
        return rope_forward(self, x, position_ids, layer_type=layer_type)

    return wrapper


class common_RotaryEmbedding(torch.nn.Module):
    # This may cause some issues.
    # @torch.no_grad()
    # PATCHED: the decorator
    @patched_dynamic_rope_update
    def forward(self, x, position_ids, layer_type=None):
        if layer_type is not None:
            # transformers>=5.0
            inv_freq = getattr(self, f"{layer_type}_inv_freq")
            attention_scaling = getattr(self, f"{layer_type}_attention_scaling")
        else:
            # transformers<5.0
            inv_freq = self.inv_freq
            attention_scaling = self.attention_scaling

        inv_freq_expanded = (
            inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        )
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = (
            x.device.type
            if isinstance(x.device.type, str) and x.device.type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * attention_scaling
            sin = emb.sin() * attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class patched_GemmaRotaryEmbedding(common_RotaryEmbedding):
    _PATCHES_ = ["forward"]
    _PATCHED_CLASS_ = transformers.models.gemma.modeling_gemma.GemmaRotaryEmbedding


if pv.Version(transformers.__version__) >= pv.Version("4.52"):

    class patched_Gemma2RotaryEmbedding(common_RotaryEmbedding):
        _PATCHES_ = ["forward"]
        _PATCHED_CLASS_ = transformers.models.gemma2.modeling_gemma2.Gemma2RotaryEmbedding

    class patched_Gemma3RotaryEmbedding(common_RotaryEmbedding):
        _PATCHES_ = ["forward"]
        _PATCHED_CLASS_ = transformers.models.gemma3.modeling_gemma3.Gemma3RotaryEmbedding


class patched_LlamaRotaryEmbedding(common_RotaryEmbedding):
    _PATCHES_ = ["forward"]
    _PATCHED_CLASS_ = transformers.models.llama.modeling_llama.LlamaRotaryEmbedding


class patched_MistralRotaryEmbedding(common_RotaryEmbedding):
    _PATCHES_ = ["forward"]
    _PATCHED_CLASS_ = transformers.models.mistral.modeling_mistral.MistralRotaryEmbedding


class patched_MixtralRotaryEmbedding(common_RotaryEmbedding):
    _PATCHES_ = ["forward"]
    _PATCHED_CLASS_ = transformers.models.mixtral.modeling_mixtral.MixtralRotaryEmbedding


class patched_PhiRotaryEmbedding(common_RotaryEmbedding):
    _PATCHES_ = ["forward"]
    _PATCHED_CLASS_ = transformers.models.phi.modeling_phi.PhiRotaryEmbedding


if pv.Version(transformers.__version__) >= pv.Version("4.51"):

    class patched_Phi3RotaryEmbedding(common_RotaryEmbedding):
        _PATCHES_ = ["forward"]
        _PATCHED_CLASS_ = transformers.models.phi3.modeling_phi3.Phi3RotaryEmbedding


if pv.Version(transformers.__version__) >= pv.Version("4.52"):

    class patched_Phi4MultimodalRotaryEmbedding(common_RotaryEmbedding):
        _PATCHES_ = ["forward"]
        _PATCHED_CLASS_ = (
            transformers.models.phi4_multimodal.modeling_phi4_multimodal.Phi4MultimodalRotaryEmbedding
        )


if pv.Version(transformers.__version__) >= pv.Version("4.53"):

    class patched_SmolLM3RotaryEmbedding(common_RotaryEmbedding):
        _PATCHES_ = ["forward"]
        _PATCHED_CLASS_ = transformers.models.smollm3.modeling_smollm3.SmolLM3RotaryEmbedding
