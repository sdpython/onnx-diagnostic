import unittest
import torch
from onnx_diagnostic.ext_test_case import ExtTestCase, hide_stdout
from onnx_diagnostic.export import CoupleInputsDynamicShapes, validate_ep


class TestValidate(ExtTestCase):
    @hide_stdout()
    def test_validate_args(self):
        class Model(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        model = Model()
        x = torch.randn((5, 6))
        y = torch.randn((1, 6))
        model(x, y)
        ds = ({0: "a", 1: "b"}, {1: "b"})
        cpl = CoupleInputsDynamicShapes((x, y), {}, ds)
        ep = torch.export.export(model, (x, y), dynamic_shapes=cpl.replace_string_by())
        validate_ep(
            ep,
            model,
            args=(x, y),
            verbose=2,
            copy=True,
            dynamic_shapes=ds,
            values_to_try={"a": [5, 10], "b": [10, 20]},
        )

    @hide_stdout()
    def test_validate_kwargs(self):
        class Model(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        model = Model()
        x = torch.randn((5, 6))
        y = torch.randn((1, 6))
        model(x=x, y=y)
        ds = dict(x={0: "a", 1: "b"}, y={1: "b"})
        cpl = CoupleInputsDynamicShapes((), dict(x=x, y=y), ds)
        ep = torch.export.export(
            model, (), kwargs=dict(x=x, y=y), dynamic_shapes=cpl.replace_string_by()
        )
        validate_ep(
            ep,
            model,
            kwargs=dict(x=x, y=y),
            verbose=2,
            copy=True,
            dynamic_shapes=ds,
            values_to_try={"a": [5, 10], "b": [10, 20]},
        )

    @hide_stdout()
    def test_validate_args_kwargs(self):
        class Model(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        model = Model()
        x = torch.randn((5, 6))
        y = torch.randn((1, 6))
        model(x, y=y)
        ds = dict(x={0: "a", 1: "b"}, y={1: "b"})
        cpl = CoupleInputsDynamicShapes((x,), dict(y=y), ds, args_names=["x"])
        ep = torch.export.export(
            model, (x,), kwargs=dict(y=y), dynamic_shapes=cpl.replace_string_by()
        )
        validate_ep(
            ep,
            model,
            args=(x,),
            kwargs=dict(y=y),
            verbose=2,
            copy=True,
            dynamic_shapes=ds,
            values_to_try={"a": [5, 10], "b": [10, 20]},
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
