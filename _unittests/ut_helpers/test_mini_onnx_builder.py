import unittest
import numpy as np
import ml_dtypes
import onnx
import torch
from onnx_diagnostic.ext_test_case import ExtTestCase
from onnx_diagnostic.reference import ExtendedReferenceEvaluator
from onnx_diagnostic.helpers.mini_onnx_builder import (
    create_onnx_model_from_input_tensors,
    create_input_tensors_from_onnx_model,
    proto_from_array,
    MiniOnnxBuilder,
)
from onnx_diagnostic.helpers.cache_helper import make_dynamic_cache, CacheKeyValue
from onnx_diagnostic.helpers import string_type


class TestMiniOnnxBuilder(ExtTestCase):
    def test_proto_from_array(self):
        self.assertRaise(lambda: proto_from_array(None), TypeError)
        t = torch.tensor([[0, 2.0], [3, 0]]).to_sparse()
        self.assertRaise(lambda: proto_from_array(t), NotImplementedError)
        tp = proto_from_array(torch.tensor([[0, 2.0], [3, 0]]).to(torch.bfloat16))
        self.assertEqual(tp.data_type, onnx.TensorProto.BFLOAT16)

    def test_mini_onnx_builder_sequence_onnx(self):
        builder = MiniOnnxBuilder()
        builder.append_output_sequence("name", [np.array([6, 7])])
        onx = builder.to_onnx()
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, {})
        self.assertEqualAny([np.array([6, 7])], got[0])

    def test_mini_onnx_builder_sequence_ort(self):
        from onnxruntime import InferenceSession

        builder = MiniOnnxBuilder()
        builder.append_output_sequence("name", [np.array([6, 7])])
        onx = builder.to_onnx()
        ref = InferenceSession(onx.SerializeToString(), providers=["CPUExecutionProvider"])
        got = ref.run(None, {})
        self.assertEqualAny([np.array([6, 7])], got[0])

    def test_mini_onnx_builder_sequence_ort_randomize(self):
        from onnxruntime import InferenceSession

        builder = MiniOnnxBuilder()
        builder.append_output_initializer(
            "name1", np.array([6, 7], dtype=np.float32), randomize=True
        )
        builder.append_output_initializer(
            "name2", np.array([-6, 7], dtype=np.float32), randomize=True
        )
        onx = builder.to_onnx()
        ref = InferenceSession(onx.SerializeToString(), providers=["CPUExecutionProvider"])
        got = ref.run(None, {})
        self.assertEqual((2,), got[0].shape)
        self.assertEqual(np.float32, got[0].dtype)
        self.assertGreaterOrEqual(got[0].min(), 0)
        self.assertEqual((2,), got[1].shape)
        self.assertEqual(np.float32, got[1].dtype)

    def test_mini_onnx_builder1(self):
        data = [
            (
                np.array([1, 2], dtype=np.int64),
                torch.tensor([4, 5], dtype=torch.float32),
                {
                    "tt1": np.array([-1, -2], dtype=np.int64),
                    "tt2": torch.tensor([-4, -5], dtype=torch.float32),
                },
                {},
            ),
            {
                "t1": np.array([1, 2], dtype=np.int64),
                "t2": torch.tensor([4, 5], dtype=torch.float32),
                "d1": {
                    "tt1": np.array([-1, -2], dtype=np.int64),
                    "tt2": torch.tensor([-4, -5], dtype=torch.float32),
                },
                "d2": {},
            },
            (
                np.array([1, 2], dtype=np.int64),
                torch.tensor([4, 5], dtype=torch.float32),
                (
                    np.array([-1, -2], dtype=np.int64),
                    torch.tensor([-4, -5], dtype=torch.float32),
                ),
                tuple(),
            ),
            {
                "t1": np.array([1, 2], dtype=np.int64),
                "t2": torch.tensor([4, 5], dtype=torch.float32),
                "l1": (
                    np.array([-1, -2], dtype=np.int64),
                    torch.tensor([-4, -5], dtype=torch.float32),
                ),
                "l2": tuple(),
            },
            # nested
            (
                {
                    "t1": np.array([1, 2], dtype=np.int64),
                    "t2": torch.tensor([4, 5], dtype=torch.float32),
                    "l1": (
                        np.array([-1, -2], dtype=np.int64),
                        torch.tensor([-4, -5], dtype=torch.float32),
                    ),
                    "l2": tuple(),
                },
                (
                    np.array([1, 2], dtype=np.int64),
                    torch.tensor([4, 5], dtype=torch.float32),
                    (
                        np.array([-1, -2], dtype=np.int64),
                        torch.tensor([-4, -5], dtype=torch.float32),
                    ),
                    tuple(),
                ),
            ),
            # simple
            np.array([1, 2], dtype=np.int64),
            torch.tensor([4, 5], dtype=torch.float32),
            (np.array([1, 2], dtype=np.int64), torch.tensor([4, 5], dtype=torch.float32)),
            [np.array([1, 2], dtype=np.int64), torch.tensor([4, 5], dtype=torch.float32)],
            {
                "t1": np.array([1, 2], dtype=np.int64),
                "t2": torch.tensor([4, 5], dtype=torch.float32),
            },
            (
                np.array([1, 2], dtype=np.int64),
                torch.tensor([4, 5], dtype=torch.float32),
                [
                    np.array([-1, -2], dtype=np.int64),
                    torch.tensor([-4, -5], dtype=torch.float32),
                ],
                [],
            ),
            {
                "t1": np.array([1, 2], dtype=np.int64),
                "t2": torch.tensor([4, 5], dtype=torch.float32),
                "l1": [
                    np.array([-1, -2], dtype=np.int64),
                    torch.tensor([-4, -5], dtype=torch.float32),
                ],
                "l2": [],
            },
        ]

        for inputs in data:
            with self.subTest(types=string_type(inputs)):
                model = create_onnx_model_from_input_tensors(inputs)
                restored = create_input_tensors_from_onnx_model(model)
                self.assertEqualAny(inputs, restored)

    def test_mini_onnx_builder2(self):
        data = [
            [],
            {},
            tuple(),
            dict(one=np.array([1, 2], dtype=np.int64)),
            [np.array([1, 2], dtype=np.int64)],
            (np.array([1, 2], dtype=np.int64),),
            [np.array([1, 2], dtype=np.int64), 1],
            dict(one=np.array([1, 2], dtype=np.int64), two=1),
            (np.array([1, 2], dtype=np.int64), 1),
        ]

        for inputs in data:
            with self.subTest(types=string_type(inputs)):
                model = create_onnx_model_from_input_tensors(inputs)
                restored = create_input_tensors_from_onnx_model(model)
                self.assertEqualAny(inputs, restored)

    def test_mini_onnx_builder_type1(self):
        data = [
            np.array([1], dtype=dtype)
            for dtype in [
                np.float32,
                np.double,
                np.float16,
                np.int32,
                np.int64,
                np.int16,
                np.int8,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
                np.bool_,
            ]
        ]

        for inputs in data:
            with self.subTest(types=string_type(inputs)):
                model = create_onnx_model_from_input_tensors(inputs)
                restored = create_input_tensors_from_onnx_model(model)
                self.assertEqualAny(inputs, restored)

    def test_mini_onnx_builder_type2(self):
        data = [
            np.array([1], dtype=dtype) for dtype in [ml_dtypes.bfloat16, ml_dtypes.float8_e3m4]
        ]

        for inputs in data:
            with self.subTest(types=string_type(inputs)):
                model = create_onnx_model_from_input_tensors(inputs)
                restored = create_input_tensors_from_onnx_model(model)
                self.assertEqualAny(inputs, restored)

    def test_mini_onnx_builder_transformers(self):
        cache = make_dynamic_cache([(torch.ones((3, 3)), torch.ones((3, 3)) * 2)])
        dc = CacheKeyValue(cache)
        self.assertEqual(len(dc.key_cache), 1)
        self.assertEqual(len(dc.value_cache), 1)

        data = [(cache,), cache]

        for inputs in data:
            with self.subTest(types=string_type(inputs)):
                model = create_onnx_model_from_input_tensors(inputs)
                restored = create_input_tensors_from_onnx_model(model)
                self.assertEqualAny(inputs, restored)

    def test_mini_onnx_builder_transformers_sep(self):
        cache = make_dynamic_cache([(torch.ones((3, 3)), torch.ones((3, 3)) * 2)])
        dc = CacheKeyValue(cache)
        self.assertEqual(len(dc.key_cache), 1)
        self.assertEqual(len(dc.value_cache), 1)

        data = [(cache,), cache]

        for inputs in data:
            with self.subTest(types=string_type(inputs)):
                model = create_onnx_model_from_input_tensors(inputs, sep="#")
                restored = create_input_tensors_from_onnx_model(model, sep="#")
                self.assertEqualAny(inputs, restored)

    def test_specific_data(self):
        data = {
            ("amain", 0, "I"): (
                (
                    torch.rand((2, 16, 3, 448, 448), dtype=torch.float16),
                    torch.rand((2, 16, 32, 32), dtype=torch.float16),
                    torch.rand((2, 2)).to(torch.int64),
                ),
                {},
            ),
        }
        model = create_onnx_model_from_input_tensors(data)
        shapes = [
            tuple(d.dim_value for d in i.type.tensor_type.shape.dim)
            for i in model.graph.output
        ]
        self.assertEqual(shapes, [(2, 16, 3, 448, 448), (2, 16, 32, 32), (2, 2), (0,)])
        names = [i.name for i in model.graph.output]
        self.assertEqual(
            [
                "dict._((amain,0,I))___tuple_0___tuple_0___tensor",
                "dict._((amain,0,I))___tuple_0___tuple_1___tensor",
                "dict._((amain,0,I))___tuple_0___tuple_2.___tensor",
                "dict._((amain,0,I))___tuple_1.___dict.___empty",
            ],
            names,
        )
        shapes = [tuple(i.dims) for i in model.graph.initializer]
        self.assertEqual(shapes, [(2, 16, 3, 448, 448), (2, 16, 32, 32), (2, 2), (0,)])
        names = [i.name for i in model.graph.initializer]
        self.assertEqual(
            [
                "t_dict._((amain,0,I))___tuple_0___tuple_0___tensor",
                "t_dict._((amain,0,I))___tuple_0___tuple_1___tensor",
                "t_dict._((amain,0,I))___tuple_0___tuple_2.___tensor",
                "t_dict._((amain,0,I))___tuple_1.___dict.___empty",
            ],
            names,
        )
        restored = create_input_tensors_from_onnx_model(model)
        self.assertEqualAny(data, restored)


if __name__ == "__main__":
    unittest.main(verbosity=2)
