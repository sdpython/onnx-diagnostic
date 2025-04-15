import unittest
from typing import Callable
import torch
from onnx_diagnostic.ext_test_case import ExtTestCase, hide_stdout
from onnx_diagnostic.reference import ExtendedReferenceEvaluator
from onnx_diagnostic.helpers.torch_test_helper import is_torchdynamo_exporting


@torch.jit.script_if_tracing
def dummy_loop(padded: torch.Tensor, pos: torch.Tensor):
    copy = torch.zeros(padded.shape)
    for i in range(pos.shape[0]):
        p = pos[i]
        copy[i, :p] = padded[i, :p]
    return copy


def wrap_for_export(f: Callable) -> Callable:

    class _wrapped(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.f = f

        def forward(self, *args, **kwargs):
            return self.f(*args, **kwargs)

    return _wrapped()


def select_when_exporting(mod, f):
    if is_torchdynamo_exporting():
        return mod
    return f


class TestJit(ExtTestCase):
    @hide_stdout()
    def test_export_loop(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.wrapped_f = wrap_for_export(dummy_loop)

            def forward(self, images, position):
                return select_when_exporting(self.wrapped_f, dummy_loop)(images, position)

        model = Model()
        x = torch.randn((5, 6))
        y = torch.arange(5, dtype=torch.int64) + 1
        expected = model(x, y)

        name = self.get_dump_file("test_export_loop.onnx")
        torch.onnx.export(
            model,
            (x, y),
            name,
            dynamic_axes={"images": {0: "batch", 1: "maxdim"}, "position": {0: "batch"}},
            dynamo=False,
        )
        ref = ExtendedReferenceEvaluator(name)
        feeds = dict(images=x.numpy(), position=y.numpy())
        got = ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

        DYN = torch.export.Dim.DYNAMIC
        ep = torch.export.export(
            model,
            (x, y),
            dynamic_shapes={"images": {0: DYN, 1: DYN}, "position": {0: DYN}},
        )
        print(ep)

        name2 = self.get_dump_file("test_export_loop.dynamo.onnx")
        torch.onnx.export(
            model,
            (x, y),
            name2,
            dynamic_axes={"images": {0: "batch", 1: "maxdim"}, "position": {0: "batch"}},
            dynamo=True,
            fallback=False,
        )
        ref = ExtendedReferenceEvaluator(name2)
        feeds = dict(images=x.numpy(), position=y.numpy())
        got = ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got)


if __name__ == "__main__":
    unittest.main(verbosity=2)
