import os
import unittest
import torch
from onnx_diagnostic.ext_test_case import (
    ExtTestCase,
    hide_stdout,
    has_transformers,
    ignore_warnings,
    requires_transformers,
    requires_experimental_experiment,
)
from onnx_diagnostic.helpers import max_diff
from onnx_diagnostic.helpers.torch_helper import torch_deepcopy
from onnx_diagnostic.helpers.rt_helper import make_feeds
from onnx_diagnostic.helpers.cache_helper import make_dynamic_cache
from onnx_diagnostic.torch_models.hghub import get_untrained_model_with_inputs
from onnx_diagnostic.torch_export_patches import torch_export_patches
from onnx_diagnostic.export.api import to_onnx, method_to_onnx


class TestValidate(ExtTestCase):
    @hide_stdout()
    def test_to_onnx(self):
        class Model(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        x = torch.randn((5, 6))
        y = torch.randn((1, 6))
        ds = ({0: "a", 1: "b"}, {1: "b"})
        to_onnx(
            Model(),
            (x, y),
            dynamic_shapes=ds,
            exporter="custom",
            filename=self.get_dump_file("to_onnx_custom.onnx"),
        )
        to_onnx(
            Model(),
            (x, y),
            dynamic_shapes=ds,
            exporter="onnx-dynamo",
            filename=self.get_dump_file("to_onnx_onnx-dynamo.onnx"),
        )

    @hide_stdout()
    @ignore_warnings(FutureWarning)
    @requires_transformers("4.50")
    def test_tiny_llm_to_onnx(self):
        import onnxruntime

        data = get_untrained_model_with_inputs("arnir0/Tiny-LLM")
        model, inputs, ds = data["model"], data["inputs"], data["dynamic_shapes"]
        b1 = data["inputs_batch1"]
        filenames = {
            "custom": self.get_dump_file("test_tiny_llm_to_onnx-custom.onnx"),
            "onnx-dynamo": self.get_dump_file("test_tiny_llm_to_onnx-dynamo.onnx"),
            "modelbuilder": self.get_dump_file("model.onnx"),
        }
        if not has_transformers("4.55"):
            # <4.55: torch._check(causal_mask.shape[3] != 33)
            #        torch._check(causal_mask.shape[3] == 33)
            del filenames["onnx-dynamo"]
        del inputs["position_ids"]
        del ds["position_ids"]
        del b1["position_ids"]

        expected = model(**torch_deepcopy(b1))

        with torch_export_patches(patch_transformers=True):
            for exporter, filename in filenames.items():
                with self.subTest(exporter=exporter):
                    to_onnx(
                        model,
                        kwargs=inputs,
                        dynamic_shapes=ds,
                        exporter=exporter,
                        filename=filename,
                    )
        for exporter, filename in filenames.items():
            if not os.path.exists(filename):
                continue
            with self.subTest(exporter=f"validate-{exporter}"):
                sess = onnxruntime.InferenceSession(
                    filename, providers=["CPUExecutionProvider"]
                )
                feeds = make_feeds(sess, b1, use_numpy=True)
                got = sess.run(None, feeds)
                diff = max_diff(expected, got)
                assert diff["abs"] <= 1e-5, f"diff={diff}"

        problem = dict(
            input_ids=torch.tensor([[24320]], dtype=torch.int64),
            attention_mask=torch.tensor([[1, 1, 1, 1]], dtype=torch.int64),
            past_key_values=make_dynamic_cache(
                [
                    torch.rand((1, 1, 3, 96), dtype=torch.float32),
                    torch.rand((1, 1, 3, 96), dtype=torch.float32),
                ]
            ),
        )

        expected = model(**torch_deepcopy(problem))
        for exporter, filename in filenames.items():
            if not os.path.exists(filename):
                continue
            with self.subTest(exporter=f"full-mask-{exporter}"):
                sess = onnxruntime.InferenceSession(
                    filename, providers=["CPUExecutionProvider"]
                )
                feeds = make_feeds(sess, problem, use_numpy=True)
                got = sess.run(None, feeds)
                diff = max_diff(expected, got)
                assert diff["abs"] <= 1e-5, f"diff={diff}"

        self.clean_dump()

    @requires_experimental_experiment("0.1")
    def test_method_to_onnx_args(self):
        class Model(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        filename = self.get_dump_file("test_method_to_onnx_args.onnx")
        inputs = [
            (torch.randn((5, 6)), torch.randn((1, 6))),
            (torch.randn((7, 7)), torch.randn((1, 7))),
        ]
        model = Model()
        method_to_call = method_to_onnx(model, exporter="custom", filename=filename)
        expecteds = []
        for args in inputs:
            expecteds.append(method_to_call(*args))
        self.assertExists(filename)
        src = method_to_call._method_src
        self.assertIn("f(self, x, y):", src)
        self.assertIn("return self._method_call(x=x, y=y)", src)
        self.assertEqual(len(list(method_to_call.named_modules())), 2)
        sess = self.check_ort(filename)
        input_names = [i.name for i in sess.get_inputs()]
        for expected, args in zip(expecteds, inputs):
            feeds = make_feeds(input_names, args, use_numpy=True)
            got = sess.run(None, feeds)
            self.assertEqualArray(expected, got[0])
        self.clean_dump()

    @requires_experimental_experiment("0.1")
    def test_method_to_onnx_kwargs(self):
        class Model(torch.nn.Module):
            def forward(self, x=None, y=None):
                return x + y

        filename = self.get_dump_file("test_method_to_onnx_kwargs.onnx")
        inputs = [
            dict(x=torch.randn((5, 6)), y=torch.randn((1, 6))),
            dict(x=torch.randn((7, 7)), y=torch.randn((1, 7))),
        ]
        model = Model()
        method_to_call = method_to_onnx(model, exporter="custom", filename=filename)
        expecteds = []
        for kwargs in inputs:
            expecteds.append(method_to_call(**kwargs))
        self.assertExists(filename)
        src = method_to_call._method_src
        self.assertIn("f(self, x=None, y=None):", src)
        self.assertIn("return self._method_call(x=x, y=y)", src)
        self.assertEqual(len(list(method_to_call.named_modules())), 2)
        sess = self.check_ort(filename)
        input_names = [i.name for i in sess.get_inputs()]
        for expected, kwargs in zip(expecteds, inputs):
            feeds = make_feeds(input_names, kwargs, use_numpy=True)
            got = sess.run(None, feeds)
            self.assertEqualArray(expected, got[0])
        self.clean_dump()

    @requires_experimental_experiment("0.1")
    def test_method_to_onnx_kwargs_patch(self):
        class Model(torch.nn.Module):
            def forward(self, x=None, y=None):
                return x + y

        filename = self.get_dump_file("test_method_to_onnx_kwargs_patch.onnx")
        inputs = [
            dict(x=torch.randn((5, 6)), y=torch.randn((1, 6))),
            dict(x=torch.randn((7, 7)), y=torch.randn((1, 7))),
        ]
        model = Model()
        method_to_call = method_to_onnx(
            model,
            exporter="custom",
            filename=filename,
            patch_kwargs=dict(patch_transformers=True),
        )
        expecteds = []
        for kwargs in inputs:
            expecteds.append(method_to_call(**kwargs))
        self.assertExists(filename)
        src = method_to_call._method_src
        self.assertIn("f(self, x=None, y=None):", src)
        self.assertIn("return self._method_call(x=x, y=y)", src)
        self.assertEqual(len(list(method_to_call.named_modules())), 2)
        sess = self.check_ort(filename)
        input_names = [i.name for i in sess.get_inputs()]
        for expected, kwargs in zip(expecteds, inputs):
            feeds = make_feeds(input_names, kwargs, use_numpy=True)
            got = sess.run(None, feeds)
            self.assertEqualArray(expected, got[0])
        self.clean_dump()

    @requires_experimental_experiment("0.1")
    @hide_stdout()
    def test_method_to_onnx_mixed(self):
        from experimental_experiment.torch_interpreter import ExportOptions

        class Model(torch.nn.Module):
            def forward(self, x, y=None):
                return x + y

        filename = self.get_dump_file("test_method_to_onnx_mixed.onnx")
        inputs = [
            ((torch.randn((5, 6)),), dict(y=torch.randn((1, 6)))),
            ((torch.randn((7, 7)),), dict(y=torch.randn((1, 7)))),
        ]
        model = Model()
        method_to_call = method_to_onnx(
            model,
            exporter="custom",
            filename=filename,
            verbose=10,
            exporter_kwargs=dict(export_options=ExportOptions(backed_size_oblivious=False)),
        )
        expecteds = []
        for args, kwargs in inputs:
            expecteds.append(method_to_call(*args, **kwargs))
        self.assertExists(filename)
        src = method_to_call._method_src
        self.assertIn("f(self, x, y=None):", src)
        self.assertIn("return self._method_call(x=x, y=y)", src)
        self.assertEqual(len(list(method_to_call.named_modules())), 2)
        sess = self.check_ort(filename)
        input_names = [i.name for i in sess.get_inputs()]
        for expected, (args, kwargs) in zip(expecteds, inputs):
            feeds = make_feeds(input_names, (args, kwargs), use_numpy=True)
            got = sess.run(None, feeds)
            self.assertEqualArray(expected, got[0])
        self.clean_dump()


if __name__ == "__main__":
    unittest.main(verbosity=2)
