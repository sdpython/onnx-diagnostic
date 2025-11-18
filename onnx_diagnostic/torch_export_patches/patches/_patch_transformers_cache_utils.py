from typing import Optional
import inspect
import transformers

try:
    from transformers.cache_utils import parse_processor_args  # noqa: F401

    patch_parse_processor_args = True
except ImportError:
    patch_parse_processor_args = False


if patch_parse_processor_args:

    def _init_cache_inspect():
        res = {}
        for processor_class in transformers.cache_utils.PROCESSOR_CLASS_MAP.values():
            try:
                params = list(inspect.signature(processor_class.__init__).parameters)[2:]
                res[processor_class.__init__] = params
            except Exception:
                res[processor_class.__init__] = None
        return res

    _cache_inspect = _init_cache_inspect()

    def patched_parse_processor_args(
        processor_class: Optional[type["CacheProcessor"]], kwargs: dict  # noqa: F821
    ) -> tuple[dict, dict]:
        """[patch:transformers.cache_utils.parse_processor_args]"""
        # If not patched...
        # Fails with transformers>=4.54 because function ``parse_processor_args``
        # relies in inspect and the exporter is not very fond of that.
        # torch._dynamo.exc.Unsupported: id() with unsupported args
        # Explanation: Dynamo doesn't know how to trace id()
        # call with args
        # (GetAttrVariable(ConstantVariable(NoneType: None), __init__),)
        # Hint: Supported args are Tensors, and functions/nn.Modules/user-defined
        # objects from outside the compiled region.
        # Hint: It may be possible to write Dynamo tracing rules for this code.
        #
        # The patch is caching the signature to avoid any call to inspect.
        if processor_class is None:
            return {}, kwargs
        params = _cache_inspect[processor_class.__init__]
        if params is None:
            return {}, kwargs
        processor_kwargs = {k: kwargs[k] for k in params if k in kwargs}
        remaining_kwargs = {k: v for k, v in kwargs.items() if k not in processor_kwargs}
        return processor_kwargs, remaining_kwargs
