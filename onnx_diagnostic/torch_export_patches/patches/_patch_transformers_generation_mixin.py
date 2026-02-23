import inspect
from typing import Optional, Tuple
import packaging.version as pv
import torch
import transformers
from transformers.cache_utils import StaticCache, Cache
from .patch_helper import _is_torchdynamo_exporting


class patched_GenerationMixin:
    """
    Applies modifications implemented in PR
    `transformers/#36652 <https://github.com/huggingface/transformers/pull/36652>`_.
    """

    _PATCHES_ = [
        "_cache_dependant_input_preparation",
        "_cache_dependant_input_preparation_exporting",
        (
            None
            if pv.Version(transformers.__version__) >= pv.Version("4.56")
            and pv.Version(transformers.__version__) < pv.Version("5.2.99")
            else "prepare_inputs_for_generation"
        ),
        # (
        #    "_sample"
        #    if pv.Version(transformers.__version__) == pv.Version("4.57.0.dev0")
        #    else None
        # ),
    ]
    _PATCHED_CLASS_ = transformers.generation.utils.GenerationMixin

    def _cache_dependant_input_preparation(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: Optional[torch.FloatTensor],
        cache_position: Optional[torch.LongTensor],
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        """
        Generic cache-dependent input preparation
        The code is put in a separate function to allow granular unit testing
        as it needs a different implementation to be exportable.

        If we have cache: let's slice `input_ids` through `cache_position`,
        to keep only the unprocessed tokens
        - Exception 1: when passing input_embeds,
          input_ids may be missing entries
        - Exception 2: some generation methods do special slicing of input_ids,
          so we don't need to do it here
        - Exception 3: with synced GPUs cache_position may go out of bounds,
          but we only want dummy token in that case.
        - Exception 4: If input_embeds are passed then slice it through
          `cache_position`, to keep only the unprocessed tokens and
          generate the first token for each sequence.
          Later use the generated Input ids for continuation.

        The current implementation does not rely on ``self`` and could be
        a class method. It is left as a standard method to be easily rewritten.
        """
        if _is_torchdynamo_exporting():
            return self._cache_dependant_input_preparation_exporting(
                input_ids, inputs_embeds, cache_position
            )
        if inputs_embeds is not None and input_ids.shape[1] == 0:  # Exception 4
            inputs_embeds = inputs_embeds[:, -cache_position.shape[0] :]
        elif inputs_embeds is not None or (  # Exception 1
            cache_position[-1] >= input_ids.shape[1]
        ):  # Exception 3
            input_ids = input_ids[:, -cache_position.shape[0] :]
        elif (
            input_ids.shape[1] != cache_position.shape[0]
        ):  # Default case (the "else", a no op, is Exception 2)
            input_ids = input_ids[:, cache_position]
        return inputs_embeds, input_ids

    def _cache_dependant_input_preparation_exporting(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: Optional[torch.FloatTensor],
        cache_position: Optional[torch.LongTensor],
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        """
        This method implements method ``_cache_dependant_input_preparation``
        with :func:`torch.cond` to make it exportable with :func:`torch.export.export`.
        The code is put in a separate function to allow granular unit testing.
        """
        if inputs_embeds is None:
            input_ids = input_ids[:, cache_position]
        else:
            # This is the code we need to implemented with torch.cond.
            # if input_ids.shape[1] == 0:
            #     inputs_embeds = inputs_embeds[:, -cache_position.shape[0] :]
            # else:
            #     if cache_position[-1] >= input_ids.shape[1]:
            #         input_ids = input_ids[:, -cache_position.shape[0] :]
            #     else:
            #         if input_ids.shape[1] != cache_position.shape[0]:
            #             input_ids = input_ids[:, cache_position]
            def branch_1(inputs_embeds, cache_position):
                return inputs_embeds[:, -cache_position.shape[0] :].clone()

            def branch_2(input_ids, cache_position):
                return input_ids[:, -cache_position.shape[0] :].clone()

            def branch_3(input_ids, cache_position):
                return input_ids[:, cache_position].clone()

            inputs_embeds, input_ids = torch.cond(
                input_ids.shape[1] == 0,
                (
                    lambda input_ids, inputs_embeds, cache_position: (
                        branch_1(inputs_embeds, cache_position),
                        input_ids.clone(),
                    )
                ),
                (
                    lambda input_ids, inputs_embeds, cache_position: (
                        inputs_embeds,
                        torch.cond(
                            cache_position[-1] >= input_ids.shape[1],
                            branch_2,
                            lambda input_ids, cache_position: (
                                torch.cond(
                                    input_ids.shape[1] != cache_position.shape[0],
                                    branch_3,
                                    (lambda input_ids, cache_position: input_ids),
                                    [input_ids, cache_position],
                                )
                            ),
                            [input_ids, cache_position],
                        ),
                    )
                ),
                [input_ids, inputs_embeds, cache_position],
            )
        return inputs_embeds, input_ids

    def prepare_inputs_for_generation(  # pragma: no cover
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        """
        Prepare the model inputs for generation.
        In includes operations like computing the 4D attention mask or
        slicing inputs given the existing cache.

        See the forward pass in the model documentation
        for expected arguments (different models might have different
        requirements for e.g. `past_key_values`).
        This function should work as is for most LLMs.
        """

        # 1. Handle BC:
        model_inputs = {}
        # - some models don't have `Cache` support
        # (which implies they don't expect `cache_position` in `forward`)
        if getattr(self, "_supports_cache_class", False):
            model_inputs["cache_position"] = cache_position
        # - `cache_position` was not a mandatory input in
        # `prepare_inputs_for_generation` for those models, and this
        #   function may be called outside of `generate`.
        # Handle most use cases by creating `cache_position` on the fly
        #   (this alternative is not as robust as calling
        # `generate` and letting it create `cache_position`)
        elif cache_position is None:
            past_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
            cache_position = torch.arange(
                past_length, input_ids.shape[1], dtype=torch.long, device=input_ids.device
            )

        # 2. Generic cache-dependent input preparation
        if past_key_values is not None:
            model_inputs["past_key_values"] = past_key_values
            inputs_embeds, input_ids = self._cache_dependant_input_preparation(
                input_ids, inputs_embeds, cache_position
            )

        # 3. Prepare base model inputs
        input_ids_key = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
        # if `inputs_embeds` are passed, we only want
        # to use them in the 1st generation step for every prompt.
        if not self.config.is_encoder_decoder:
            if inputs_embeds is not None and len(cache_position) == inputs_embeds.shape[1]:
                model_inputs[input_ids_key] = None
                model_inputs["inputs_embeds"] = inputs_embeds
            else:
                # `clone` calls in this function ensure a consistent stride. See #32227
                model_inputs[input_ids_key] = input_ids.clone(
                    memory_format=torch.contiguous_format
                )
                model_inputs["inputs_embeds"] = None
        else:
            model_inputs[input_ids_key] = input_ids.clone(
                memory_format=torch.contiguous_format
            )

        # 4. Create missing `position_ids` on the fly
        encoder_attention_mask = attention_mask if self.config.is_encoder_decoder else None
        attention_mask = (
            kwargs.pop("decoder_attention_mask", None)
            if self.config.is_encoder_decoder
            else attention_mask
        )
        attention_mask_key = (
            "decoder_attention_mask" if self.config.is_encoder_decoder else "attention_mask"
        )
        position_ids_key = (
            "decoder_position_ids" if self.config.is_encoder_decoder else "position_ids"
        )
        if (
            attention_mask is not None
            and kwargs.get(position_ids_key) is None
            and position_ids_key in set(inspect.signature(self.forward).parameters.keys())
        ):
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            kwargs[position_ids_key] = (
                position_ids  # placed in kwargs for further processing (see below)
            )

        # 5. Slice model inputs if it's an input
        # that should have the same length as `input_ids`
        for model_input_name in ["position_ids", "token_type_ids", "decoder_position_ids"]:
            model_input = kwargs.get(model_input_name)
            if model_input is not None:
                if past_key_values is not None:
                    current_input_length = (
                        model_inputs["inputs_embeds"].shape[1]
                        if model_inputs.get("inputs_embeds") is not None
                        else model_inputs[input_ids_key].shape[1]
                    )
                    model_input = model_input[:, -current_input_length:]
                    model_input = model_input.clone(memory_format=torch.contiguous_format)
                model_inputs[model_input_name] = model_input

        # 6. Create 4D attention mask is we are using a
        # `StaticCache` (important for performant compiled forward pass)
        if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
                device = model_inputs["inputs_embeds"].device
            else:
                batch_size, sequence_length = model_inputs[input_ids_key].shape
                device = model_inputs[input_ids_key].device

            # Create the causal mask with fixed shape in advance,
            # to reduce recompilations. If the function to create
            # the 4D causal mask exists,
            # it should be present in the base model (XXXModel class).
            base_model = getattr(self, self.base_model_prefix, None)
            if base_model is None:
                causal_mask_creation_function = getattr(
                    self, "_prepare_4d_causal_attention_mask_with_cache_position", None
                )
            else:
                causal_mask_creation_function = getattr(
                    base_model, "_prepare_4d_causal_attention_mask_with_cache_position", None
                )
            if causal_mask_creation_function is None:
                pass
                # logger.warning_once(
                #     f"{self.__class__.__name__} has no "
                #     "`_prepare_4d_causal_attention_mask_with_cache_position` method "
                #     "defined in its base modeling class. "
                #     "Compiled forward passes will be sub-optimal. If you're "
                #     "writing code, see Llama for an example implementation. "
                #     "If you're a user, please report this "
                #     "issue on GitHub."
                # )
            else:
                attention_mask = causal_mask_creation_function(
                    attention_mask,
                    sequence_length=sequence_length,
                    target_length=past_key_values.get_max_cache_shape(),
                    dtype=self.dtype,
                    device=device,
                    cache_position=cache_position,
                    batch_size=batch_size,
                    config=self.config,
                    past_key_values=past_key_values,
                )
        if attention_mask is not None:
            model_inputs[attention_mask_key] = attention_mask

        if encoder_attention_mask is not None:
            model_inputs["attention_mask"] = encoder_attention_mask

        # 7. Forward ALL kwargs that are uninitialized (e.g. `use_cache`).
        for key, value in kwargs.items():
            if key not in model_inputs:
                model_inputs[key] = value

        # 8. Remove unexpected `generate` inputs (TODO @joao: fix trainer and examples)
        model_inputs.pop("labels", None)
        return model_inputs
