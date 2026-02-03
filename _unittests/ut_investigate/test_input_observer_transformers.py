import unittest
import pandas
import torch
from onnx_diagnostic.ext_test_case import ExtTestCase, requires_transformers
from onnx_diagnostic.investigate.input_observer import InputObserver
from onnx_diagnostic.torch_export_patches import (
    register_additional_serialization_functions,
    torch_export_patches,
)
from onnx_diagnostic.export.api import to_onnx
from onnx_diagnostic.torch_models.hghub import get_untrained_model_with_inputs
from onnx_diagnostic.helpers.rt_helper import onnx_generate
from onnx_diagnostic.helpers.cache_helper import make_dynamic_cache, make_encoder_decoder_cache


class TestInputObserverTransformers(ExtTestCase):
    @requires_transformers("4.57")
    def test_input_observer_onnx_generate_tiny_llm(self):
        mid = "arnir0/Tiny-LLM"
        data = get_untrained_model_with_inputs(mid)
        model, inputs, _ds = data["model"], data["inputs"], data["dynamic_shapes"]
        input_ids = inputs["input_ids"][:1]

        observer = InputObserver()
        with (
            register_additional_serialization_functions(patch_transformers=True),
            observer(model),
        ):
            outputs = model.generate(input_ids=input_ids, do_sample=False)

        filenamec = self.get_dump_file("test_input_observer_onnx_generate_tiny_llm.onnx")
        with torch_export_patches(patch_transformers=True):
            to_onnx(
                model,
                (),
                kwargs=observer.infer_arguments(),
                dynamic_shapes=observer.infer_dynamic_shapes(set_batch_dimension_for=True),
                filename=filenamec,
                exporter="custom",
            )

        data = observer.check_discrepancies(filenamec, progress_bar=False)
        df = pandas.DataFrame(data)
        self.assertLess(df["abs"].max(), 1e-4)

        onnx_tokens = onnx_generate(
            filenamec,
            input_ids=input_ids,
            attention_mask=torch.ones(input_ids.shape, dtype=torch.int64),
            eos_token_id=model.config.eos_token_id,
            max_new_tokens=20,
        )
        self.assertEqualArray(outputs, onnx_tokens)

    @requires_transformers("4.55")
    def test_encoder_decoder_cache_args(self):
        class Model(torch.nn.Module):
            def forward(self, cache):
                return cache

        cache1 = make_encoder_decoder_cache(
            make_dynamic_cache(
                [
                    (torch.ones((1, 6, 4, 64)), torch.ones((1, 6, 4, 64))),
                    (torch.ones((1, 6, 4, 64)), torch.ones((1, 6, 4, 64))),
                    (torch.ones((1, 6, 4, 64)), torch.ones((1, 6, 4, 64))),
                    (torch.ones((1, 6, 4, 64)), torch.ones((1, 6, 4, 64))),
                ],
            ),
            make_dynamic_cache(
                [
                    (torch.ones((1, 6, 1500, 64)), torch.ones((1, 6, 1500, 64))),
                    (torch.ones((1, 6, 1500, 64)), torch.ones((1, 6, 1500, 64))),
                    (torch.ones((1, 6, 1500, 64)), torch.ones((1, 6, 1500, 64))),
                    (torch.ones((1, 6, 1500, 64)), torch.ones((1, 6, 1500, 64))),
                ],
            ),
        )
        cache2 = make_encoder_decoder_cache(
            make_dynamic_cache(
                [
                    (torch.ones((1, 6, 5, 64)), torch.ones((1, 6, 5, 64))),
                    (torch.ones((1, 6, 5, 64)), torch.ones((1, 6, 5, 64))),
                    (torch.ones((1, 6, 5, 64)), torch.ones((1, 6, 5, 64))),
                    (torch.ones((1, 6, 5, 64)), torch.ones((1, 6, 5, 64))),
                ],
            ),
            make_dynamic_cache(
                [
                    (torch.ones((1, 6, 1500, 64)), torch.ones((1, 6, 1500, 64))),
                    (torch.ones((1, 6, 1500, 64)), torch.ones((1, 6, 1500, 64))),
                    (torch.ones((1, 6, 1500, 64)), torch.ones((1, 6, 1500, 64))),
                    (torch.ones((1, 6, 1500, 64)), torch.ones((1, 6, 1500, 64))),
                ],
            ),
        )

        model = Model()
        observer = InputObserver()
        with (
            register_additional_serialization_functions(patch_transformers=True),
            observer(model),
        ):
            model(None)
            model(cache1)
            model(cache2)
            dyn_shapes = observer.infer_dynamic_shapes()
            args = observer.infer_arguments(0)
        dyn_shapes_out = observer.infer_dynamic_shapes()
        args0 = observer.infer_arguments(0)
        self.assertEqual(dyn_shapes, dyn_shapes_out)
        cst = torch.export.Dim.DYNAMIC
        self.assertEqual(
            (
                [
                    [
                        {2: cst},
                        {2: cst},
                        {2: cst},
                        {2: cst},
                        {2: cst},
                        {2: cst},
                        {2: cst},
                        {2: cst},
                    ],
                    [{}, {}, {}, {}, {}, {}, {}, {}],
                ],
            ),
            dyn_shapes_out,
        )
        self.assertEqualAny(args, args0)
        self.assertEqual(args[0].self_attention_cache.layers[0].keys.shape, (1, 6, 0, 64))
        self.assertEqual(args[0].cross_attention_cache.layers[0].keys.shape, (1, 6, 1500, 64))

    @requires_transformers("4.55")
    def test_encoder_decoder_cache_kwargs(self):
        class Model(torch.nn.Module):
            def forward(self, cache):
                return cache

        cache1 = make_encoder_decoder_cache(
            make_dynamic_cache(
                [
                    (torch.ones((1, 6, 4, 64)), torch.ones((1, 6, 4, 64))),
                    (torch.ones((1, 6, 4, 64)), torch.ones((1, 6, 4, 64))),
                    (torch.ones((1, 6, 4, 64)), torch.ones((1, 6, 4, 64))),
                    (torch.ones((1, 6, 4, 64)), torch.ones((1, 6, 4, 64))),
                ],
            ),
            make_dynamic_cache(
                [
                    (torch.ones((1, 6, 1500, 64)), torch.ones((1, 6, 1500, 64))),
                    (torch.ones((1, 6, 1500, 64)), torch.ones((1, 6, 1500, 64))),
                    (torch.ones((1, 6, 1500, 64)), torch.ones((1, 6, 1500, 64))),
                    (torch.ones((1, 6, 1500, 64)), torch.ones((1, 6, 1500, 64))),
                ],
            ),
        )
        cache2 = make_encoder_decoder_cache(
            make_dynamic_cache(
                [
                    (torch.ones((1, 6, 5, 64)), torch.ones((1, 6, 5, 64))),
                    (torch.ones((1, 6, 5, 64)), torch.ones((1, 6, 5, 64))),
                    (torch.ones((1, 6, 5, 64)), torch.ones((1, 6, 5, 64))),
                    (torch.ones((1, 6, 5, 64)), torch.ones((1, 6, 5, 64))),
                ],
            ),
            make_dynamic_cache(
                [
                    (torch.ones((1, 6, 1500, 64)), torch.ones((1, 6, 1500, 64))),
                    (torch.ones((1, 6, 1500, 64)), torch.ones((1, 6, 1500, 64))),
                    (torch.ones((1, 6, 1500, 64)), torch.ones((1, 6, 1500, 64))),
                    (torch.ones((1, 6, 1500, 64)), torch.ones((1, 6, 1500, 64))),
                ],
            ),
        )

        model = Model()
        observer = InputObserver()
        with (
            register_additional_serialization_functions(patch_transformers=True),
            observer(model),
        ):
            model(cache=None)
            model(cache=cache1)
            model(cache=cache2)
            dyn_shapes = observer.infer_dynamic_shapes()
            args = observer.infer_arguments(0)
        dyn_shapes_out = observer.infer_dynamic_shapes()
        args0 = observer.infer_arguments(0)
        self.assertEqual(dyn_shapes, dyn_shapes_out)
        cst = torch.export.Dim.DYNAMIC
        self.assertEqual(
            dict(
                cache=[
                    [
                        {2: cst},
                        {2: cst},
                        {2: cst},
                        {2: cst},
                        {2: cst},
                        {2: cst},
                        {2: cst},
                        {2: cst},
                    ],
                    [{}, {}, {}, {}, {}, {}, {}, {}],
                ],
            ),
            dyn_shapes_out,
        )
        self.assertEqualAny(args, args0)
        self.assertEqual(
            args["cache"].self_attention_cache.layers[0].keys.shape, (1, 6, 0, 64)
        )
        self.assertEqual(
            args["cache"].cross_attention_cache.layers[0].keys.shape, (1, 6, 1500, 64)
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
