import unittest
import numpy as np
import ml_dtypes
import onnx
import torch
import transformers
from onnx_diagnostic.ext_test_case import ExtTestCase, hide_stdout, requires_torch
from onnx_diagnostic.helpers import max_diff, string_type
from onnx_diagnostic.helpers.torch_helper import (
    dummy_llm,
    to_numpy,
    is_torchdynamo_exporting,
    model_statistics,
    steal_append,
    steal_forward,
    replace_string_by_dynamic,
    to_any,
    torch_deepcopy,
    torch_tensor_size,
)
from onnx_diagnostic.helpers.cache_helper import (
    make_dynamic_cache,
    make_encoder_decoder_cache,
    make_mamba_cache,
    make_sliding_window_cache,
    CacheKeyValue,
)
from onnx_diagnostic.helpers.mini_onnx_builder import create_input_tensors_from_onnx_model
from onnx_diagnostic.helpers.onnx_helper import from_array_extended, to_array_extended
from onnx_diagnostic.helpers.torch_helper import to_tensor

TFLOAT = onnx.TensorProto.FLOAT


class TestTorchTestHelper(ExtTestCase):

    def test_is_torchdynamo_exporting(self):
        self.assertFalse(is_torchdynamo_exporting())

    def test_dummy_llm(self):
        for cls_name in ["AttentionBlock", "MultiAttentionBlock", "DecoderLayer", "LLM"]:
            model, inputs = dummy_llm(cls_name)
            model(*inputs)

    def test_dummy_llm_ds(self):
        for cls_name in ["AttentionBlock", "MultiAttentionBlock", "DecoderLayer", "LLM"]:
            model, inputs, ds = dummy_llm(cls_name, dynamic_shapes=True)
            model(*inputs)
            self.assertIsInstance(ds, dict)

    def test_dummy_llm_exc(self):
        self.assertRaise(lambda: dummy_llm("LLLLLL"), NotImplementedError)

    def test_to_numpy(self):
        t = torch.tensor([0, 1], dtype=torch.bfloat16)
        a = to_numpy(t)
        self.assertEqual(a.dtype, ml_dtypes.bfloat16)

    @hide_stdout()
    def test_steal_forward(self):
        class Model(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        inputs = torch.rand(3, 4), torch.rand(3, 4)
        model = Model()
        with steal_forward(model):
            model(*inputs)

    @hide_stdout()
    def test_steal_forward_multi(self):
        class SubModel(torch.nn.Module):
            def forward(self, x):
                return x * x

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.s1 = SubModel()
                self.s2 = SubModel()

            def forward(self, x, y):
                return self.s1(x) + self.s2(y)

        inputs = torch.rand(3, 4), torch.rand(3, 4)
        model = Model()
        with steal_forward(
            [
                (
                    "main",
                    model,
                ),
                ("  s1", model.s1),
                ("  s2", model.s2),
            ]
        ):
            model(*inputs)

    @hide_stdout()
    def test_steal_forward_dump_file(self):
        class SubModel(torch.nn.Module):
            def forward(self, x):
                return x * x

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.s1 = SubModel()
                self.s2 = SubModel()

            def forward(self, x, y):
                return self.s1(x) + self.s2(y)

        inputs = torch.rand(3, 4), torch.rand(3, 4)
        model = Model()
        dump_file = self.get_dump_file("test_steal_forward_dump_file.onnx")
        with steal_forward(
            [
                (
                    "main",
                    model,
                ),
                ("  s1", model.s1),
                ("  s2", model.s2),
            ],
            dump_file=dump_file,
        ):
            res1 = model(*inputs)
            res2 = model(*inputs)
        self.assertExists(dump_file)
        restored = create_input_tensors_from_onnx_model(dump_file)
        self.assertEqual(
            [
                ("main", 0, "I"),
                ("main", 0, "O"),
                ("main", 1, "I"),
                ("main", 1, "O"),
                ("s1", 0, "I"),
                ("s1", 0, "O"),
                ("s1", 1, "I"),
                ("s1", 1, "O"),
                ("s2", 0, "I"),
                ("s2", 0, "O"),
                ("s2", 1, "I"),
                ("s2", 1, "O"),
            ],
            sorted(restored),
        )
        self.assertEqualAny(restored["main", 0, "I"], (inputs, {}))
        self.assertEqualAny(restored["main", 1, "I"], (inputs, {}))
        self.assertEqualAny(restored["main", 0, "O"], res1)
        self.assertEqualAny(restored["main", 0, "O"], res2)

    @hide_stdout()
    def test_steal_forward_dump_file_steal_append(self):
        class SubModel(torch.nn.Module):
            def forward(self, x):
                return x * x

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.s1 = SubModel()
                self.s2 = SubModel()

            def forward(self, x, y):
                sx = self.s1(x)
                steal_append("sx", sx)
                return sx + self.s2(y)

        inputs = torch.rand(3, 4), torch.rand(3, 4)
        model = Model()
        dump_file = self.get_dump_file("test_steal_forward_dump_file.onnx")
        with steal_forward(model, dump_file=dump_file):
            model(*inputs)
            model(*inputs)
        self.assertExists(dump_file)
        restored = create_input_tensors_from_onnx_model(dump_file)
        self.assertEqual(
            {("", 1, "I"), ("", 1, "O"), "sx", ("", 0, "O"), "sx_1", ("", 0, "I")},
            set(restored),
        )

    @hide_stdout()
    def test_steal_forward_dump_file_steal_append_drop(self):
        class SubModel(torch.nn.Module):
            def forward(self, x):
                return x * x

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.s1 = SubModel()
                self.s2 = SubModel()

            def forward(self, x, y):
                sx = self.s1(x)
                steal_append("sx", sx)
                return sx + self.s2(y)

        inputs = dict(x=torch.rand(3, 4), y=torch.rand(3, 4))
        model = Model()
        dump_file = self.get_dump_file("test_steal_forward_dump_file_drop.onnx")
        with steal_forward(model, dump_file=dump_file, dump_drop={"x"}):
            model(**inputs)
            model(**inputs)
        self.assertExists(dump_file)
        restored = create_input_tensors_from_onnx_model(dump_file)
        self.assertEqual(
            {("", 1, "I"), ("", 1, "O"), "sx", ("", 0, "O"), "sx_1", ("", 0, "I")},
            set(restored),
        )
        first = restored[("", 0, "I")]
        _a, kws = first
        self.assertNotIn("x", kws)

    @hide_stdout()
    def test_steal_forward_submodules(self):
        class SubModel(torch.nn.Module):
            def forward(self, x):
                return x * x

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.s1 = SubModel()
                self.s2 = SubModel()

            def forward(self, x, y):
                return self.s1(x) + self.s2(y)

        inputs = torch.rand(3, 4), torch.rand(3, 4)
        model = Model()
        dump_file = self.get_dump_file("test_steal_forward_submodules.onnx")
        with steal_forward(model, submodules=True, dump_file=dump_file):
            model(*inputs)
        restored = create_input_tensors_from_onnx_model(dump_file)
        for k, v in sorted(restored.items()):
            if isinstance(v, tuple):
                args, kwargs = v
                print("input", k, args, kwargs)
            else:
                print("output", k, v)
        print(string_type(restored, with_shape=True))
        l1, l2 = 186, 195
        self.assertEqual(
            len(
                [
                    (f"-Model-{l2}", 0, "I"),
                    (f"-Model-{l2}", 0, "O"),
                    (f"s1-SubModel-{l1}", 0, "I"),
                    (f"s1-SubModel-{l1}", 0, "O"),
                    (f"s2-SubModel-{l1}", 0, "I"),
                    (f"s2-SubModel-{l1}", 0, "O"),
                ]
            ),
            len(sorted(restored)),
        )

    def test_replace_string_by_dynamic(self):
        example = {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": ({0: "batch_size", 1: "sequence_length"},),
            "position_ids": [{0: "batch_size", 1: "sequence_length"}],
        }
        proc = replace_string_by_dynamic(example)
        sproc = (
            str(proc)
            .replace("_DimHint(type=<_DimHintType.DYNAMIC: 3>)", "DYN")
            .replace(" ", "")
            .replace("<_DimHint.DYNAMIC:3>", "DYN")
            .replace(
                "_DimHint(type=<_DimHintType.DYNAMIC:3>,min=None,max=None,_factory=True)",
                "DYN",
            )
        )
        self.assertEqual(
            "{'input_ids':{0:DYN,1:DYN},'attention_mask':({0:DYN,1:DYN},),'position_ids':[{0:DYN,1:DYN}]}",
            sproc,
        )

    def test_to_any(self):
        c1 = make_dynamic_cache([(torch.rand((4, 4, 4)), torch.rand((4, 4, 4)))])
        c2 = make_encoder_decoder_cache(
            make_dynamic_cache([(torch.rand((4, 4, 4)), torch.rand((4, 4, 4)))]),
            make_dynamic_cache([(torch.rand((5, 5, 5)), torch.rand((5, 5, 5)))]),
        )
        a = {"t": [(torch.tensor([1, 2]), c1, c2), {4, 5}]}
        at = to_any(a, torch.float16)
        self.assertIn("T10r", string_type(at))

    def test_torch_deepcopy_cache_dce(self):
        c1 = make_dynamic_cache([(torch.rand((4, 4, 4)), torch.rand((4, 4, 4)))])
        c2 = make_encoder_decoder_cache(
            make_dynamic_cache([(torch.rand((4, 4, 4)), torch.rand((4, 4, 4)))]),
            make_dynamic_cache([(torch.rand((5, 5, 5)), torch.rand((5, 5, 5)))]),
        )
        cc = torch_deepcopy(c2)
        self.assertEqual(type(c2), type(c2))
        self.assertEqual(max_diff(c2, cc)["abs"], 0)
        a = {"t": [(torch.tensor([1, 2]), c1, c2), {4, 5}]}
        at = torch_deepcopy(a)
        hash1 = string_type(at, with_shape=True, with_min_max=True)
        ccv = CacheKeyValue(c1)
        ccv.key_cache[0] += 1000
        hash2 = string_type(at, with_shape=True, with_min_max=True)
        self.assertEqual(hash1, hash2)
        self.assertGreater(torch_tensor_size(cc), 1)

    @requires_torch("4.50")
    def test_torch_deepcopy_mamba_cache(self):
        cache = make_mamba_cache(
            [
                (torch.rand((4, 4, 4)), torch.rand((4, 4, 4))),
                (torch.rand((4, 4, 4)), torch.rand((4, 4, 4))),
                (torch.rand((4, 4, 4)), torch.rand((4, 4, 4))),
            ]
        )
        at = torch_deepcopy(cache)
        self.assertEqual(type(cache), type(at))
        self.assertEqual(max_diff(cache, at)["abs"], 0)
        hash1 = string_type(at, with_shape=True, with_min_max=True)
        cache.conv_states[0] += 1000
        hash2 = string_type(at, with_shape=True, with_min_max=True)
        self.assertEqual(hash1, hash2)
        self.assertGreater(torch_tensor_size(cache), 1)

    def test_torch_deepcopy_base_model_outputs(self):
        bo = transformers.modeling_outputs.BaseModelOutput(
            last_hidden_state=torch.rand((4, 4, 4))
        )
        at = torch_deepcopy(bo)
        self.assertEqual(max_diff(bo, at)["abs"], 0)
        self.assertEqual(type(bo), type(at))
        hash1 = string_type(at, with_shape=True, with_min_max=True)
        bo.last_hidden_state[0] += 1000
        hash2 = string_type(at, with_shape=True, with_min_max=True)
        self.assertEqual(hash1, hash2)
        self.assertGreater(torch_tensor_size(bo), 1)

    def test_torch_deepcopy_sliding_windon_cache(self):
        cache = make_sliding_window_cache(
            [
                (torch.rand((4, 5, 6, 7)), torch.rand((4, 5, 6, 7))),
                (torch.rand((4, 5, 6, 7)), torch.rand((4, 5, 6, 7))),
                (torch.rand((4, 5, 6, 7)), torch.rand((4, 5, 6, 7))),
            ]
        )
        at = torch_deepcopy(cache)
        self.assertEqual(type(cache), type(at))
        self.assertEqual(max_diff(cache, at)["abs"], 0)
        hash1 = string_type(at, with_shape=True, with_min_max=True)
        CacheKeyValue(cache).key_cache[0] += 1000
        hash2 = string_type(at, with_shape=True, with_min_max=True)
        self.assertEqual(hash1, hash2)
        self.assertGreater(torch_tensor_size(cache), 1)

    def test_torch_deepcopy_none(self):
        self.assertEmpty(torch_deepcopy(None))
        self.assertEqual(torch_tensor_size(None), 0)

    def test_model_statistics(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.p1 = torch.nn.Parameter(torch.tensor([1], dtype=torch.float32))
                self.b1 = torch.nn.Buffer(torch.tensor([1], dtype=torch.float32))

            def forward(self, x, y=None):
                return x + y + self.p1 + self.b1

        model = Model()
        x, y = torch.rand((3, 4)), torch.rand((3, 4))
        model(x, y)
        stat = model_statistics(model)
        self.assertEqual(
            {
                "type": "Model",
                "n_modules": 1,
                "param_size": 4,
                "buffer_size": 4,
                "float32": 8,
                "size_mb": 0,
            },
            stat,
        )

    def test_to_tensor(self):
        for dtype in [
            np.int8,
            np.uint8,
            np.int16,
            np.uint16,
            np.int32,
            np.uint32,
            np.int64,
            np.uint64,
            np.float16,
            np.float32,
            np.float64,
        ]:
            with self.subTest(dtype=dtype):
                a = np.random.rand(4, 5).astype(dtype)
                proto = from_array_extended(a)
                b = to_array_extended(proto)
                self.assertEqualArray(a, b)
                c = to_tensor(proto)
                self.assertEqualArray(a, c)

        for dtype in [torch.bfloat16]:
            with self.subTest(dtype=dtype):
                a = torch.rand((4, 5), dtype=dtype)
                proto = from_array_extended(a)
                c = to_tensor(proto)
                self.assertEqualArray(a, c)


if __name__ == "__main__":
    unittest.main(verbosity=2)
