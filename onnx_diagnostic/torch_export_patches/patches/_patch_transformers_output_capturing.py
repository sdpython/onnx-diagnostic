try:
    import transformers.utils.output_capturing  # noqa: F401

    patch_output_capturing = True
except ImportError:
    patch_output_capturing = False


if patch_output_capturing:
    # Introduced in 5.2.0
    # https://github.com/huggingface/transformers/pull/43765/
    # changes#diff-b5f9fdbe43ffd89fbdf2b246dc78dd32aa4bdb587e7a53e4dad37b7efd79ab0a
    import torch
    import transformers
    from transformers.utils.import_utils import is_torchdynamo_compiling

    class patched_CompileableContextVar:
        _PATCHES_ = ["set"]
        _PATCHED_CLASS_ = transformers.utils.output_capturing.CompileableContextVar

        def set(self, value):
            if is_torchdynamo_compiling() and not torch.compiler.is_exporting():
                self.global_var = value
                self.compiling = True
                return None
            else:
                return self.context_var.set(value)
