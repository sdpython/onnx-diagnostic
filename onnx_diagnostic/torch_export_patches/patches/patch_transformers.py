# transformers
from .patch_helper import _has_transformers

from ._patch_transformers_attention import (
    patched_sdpa_attention_forward,
    patched_model_bart_eager_attention_forward,
    patched_modeling_marian_eager_attention_forward,
)

from ._patch_transformers_cache_utils import patch_parse_processor_args

if patch_parse_processor_args:
    from ._patch_transformers_cache_utils import patched_parse_processor_args

from ._patch_transformers_causal_mask import patched_AttentionMaskConverter

from ._patch_transformers_dynamic_cache import patch_DynamicLayer, patch_DynamicCache

if patch_DynamicLayer:
    from ._patch_transformers_dynamic_cache import patched_DynamicLayer
if patch_DynamicCache:
    from ._patch_transformers_dynamic_cache import patched_DynamicCache

from ._patch_transformers_generation_mixin import patched_GenerationMixin

from ._patch_transformers_masking_utils import patch_masking_utils

if patch_masking_utils:
    from ._patch_transformers_masking_utils import (
        patched__vmap_for_bhqkv,
        patched_eager_mask,
        patched_sdpa_mask_recent_torch,
    )

from ._patch_transformers_rotary_embedding import (
    patched__compute_dynamic_ntk_parameters,
    patched_dynamic_rope_update,
    patched_GemmaRotaryEmbedding,
    patched_LlamaRotaryEmbedding,
    patched_MistralRotaryEmbedding,
    patched_MixtralRotaryEmbedding,
    patched_PhiRotaryEmbedding,
)

if _has_transformers("4.51"):
    from ._patch_transformers_rotary_embedding import patched_Phi3RotaryEmbedding
if _has_transformers("4.52"):
    from ._patch_transformers_rotary_embedding import (
        patched_Gemma2RotaryEmbedding,
        patched_Gemma3RotaryEmbedding,
        patched_Phi4MultimodalRotaryEmbedding,
    )
if _has_transformers("4.53"):
    from ._patch_transformers_rotary_embedding import patched_SmolLM3RotaryEmbedding

# Models

from ._patch_transformers_gemma3 import patch_gemma3

if patch_gemma3:
    from ._patch_transformers_gemma3 import patched_Gemma3Model

from ._patch_transformers_idefics import patched_IdeficsEmbedding, patched_IdeficsAttention


from ._patch_transformers_qwen2 import patch_qwen2

if patch_qwen2:
    from ._patch_transformers_qwen2 import patched_VisionAttention

from ._patch_transformers_qwen2_5 import patch_qwen2_5

if patch_qwen2_5:
    from ._patch_transformers_qwen2_5 import (
        patched_Qwen2_5_VLForConditionalGeneration,
        patched_Qwen2_5_VisionTransformerPretrainedModel,
        patched_Qwen2_5_VLVisionAttentionOneIteration,
        patched_Qwen2_5_VLVisionAttention,
    )

from ._patch_transformers_qwen3 import patch_qwen3

if patch_qwen3:
    from ._patch_transformers_qwen3 import patched_Qwen3MoeSparseMoeBlock


from ._patch_transformers_sam_mask_decoder import patched_SamMaskDecoder
