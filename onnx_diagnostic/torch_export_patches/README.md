# Patches to export HuggingFace models

See [Patches Explained](https://sdpython.github.io/doc/onnx-diagnostic/dev/patches.html).

```python
from onnx_diagnostic.torch_export_patches import torch_export_patches

with torch_export_patches(patch_transformers=True) as f:
    ep = torch.export.export(model, args, kwargs=kwargs, dynamic_shapes=dynamic_shapes)
    # ...
```
