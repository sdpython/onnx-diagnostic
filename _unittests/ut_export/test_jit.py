import unittest
import torch
from onnx_diagnostic.ext_test_case import ExtTestCase, hide_stdout, ignore_warnings
from onnx_diagnostic.reference import ExtendedReferenceEvaluator
from onnx_diagnostic.helpers.torch_test_helper import is_torchdynamo_exporting

try:
    from experimental_experiment.torch_interpreter import to_onnx
except ImportError:
    to_onnx = None


@torch.jit.script_if_tracing
def dummy_loop(padded: torch.Tensor, pos: torch.Tensor):
    copy = torch.zeros(padded.shape)
    for i in range(pos.shape[0]):
        p = pos[i]
        copy[i, :p] = padded[i, :p]
    return copy


def dummy_loop_with_scan(padded: torch.Tensor, pos: torch.Tensor):
    def pad_row(padded, p):
        row = torch.zeros((padded.shape[0],))
        torch._check(p.item() > 0)
        torch._check(p.item() < padded.shape[0])
        # this check is not always true, we add it anyway to make this dimension >= 2
        # and avoid raising an exception about dynamic dimension in {0, 1}
        if is_torchdynamo_exporting():
            torch._check(p.item() > 1)
        row[: p.item()] = padded[: p.item()]
        return (row,)

    return torch.ops.higher_order.scan(
        pad_row,
        [],
        [padded, pos],
        [],
    )


def select_when_exporting(f, f_scan):
    return f_scan if is_torchdynamo_exporting() else f


class TestJit(ExtTestCase):
    def test_dummy_loop(self):
        x = torch.randn((5, 6))
        y = torch.arange(5, dtype=torch.int64) + 1
        res = dummy_loop(x, y)
        res_scan = dummy_loop_with_scan(x, y)
        self.assertEqualArray(res, res_scan[0])

    @hide_stdout()
    @ignore_warnings(UserWarning)
    def test_export_loop_onnxscript(self):
        class Model(torch.nn.Module):
            def forward(self, images, position):
                return select_when_exporting(dummy_loop, dummy_loop_with_scan)(
                    images, position
                )

        model = Model()
        x = torch.randn((5, 6))
        y = torch.arange(5, dtype=torch.int64) + 1
        expected = model(x, y)

        name = self.get_dump_file("test_export_loop_onnxscript.onnx")
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
            strict=False,
        )
        self.assertNotEmpty(ep)

        name2 = self.get_dump_file("test_export_loop_onnxscript.dynamo.onnx")
        torch.onnx.export(
            model,
            (x, y),
            name2,
            dynamic_shapes={"images": {0: "batch", 1: "maxdim"}, "position": {0: "batch"}},
            dynamo=True,
            fallback=False,
        )
        ref = ExtendedReferenceEvaluator(name2)
        feeds = dict(images=x.numpy(), position=y.numpy())
        got = ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got)

    @hide_stdout()
    @ignore_warnings(UserWarning)
    @unittest.skipIf(to_onnx is None, "missing to_onnx")
    def test_export_loop_custom(self):
        class Model(torch.nn.Module):
            def forward(self, images, position):
                return select_when_exporting(dummy_loop, dummy_loop_with_scan)(
                    images, position
                )

        model = Model()
        x = torch.randn((5, 6))
        y = torch.arange(5, dtype=torch.int64) + 1
        expected = model(x, y)

        name2 = self.get_dump_file("test_export_loop.custom.onnx")
        to_onnx(
            model,
            (x, y),
            filename=name2,
            dynamic_shapes={"images": {0: "batch", 1: "maxdim"}, "position": {0: "batch"}},
        )
        ref = ExtendedReferenceEvaluator(name2)
        feeds = dict(images=x.numpy(), position=y.numpy())
        got = ref.run(None, feeds)[0]
        self.assertEqualArray(expected, got)


if __name__ == "__main__":
    unittest.main(verbosity=2)
