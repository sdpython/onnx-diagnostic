import inspect
import unittest
import numpy as np
import ml_dtypes
import onnx
import onnx.helper as oh
import torch
from onnx_diagnostic.ext_test_case import ExtTestCase, skipif_ci_windows, hide_stdout
from onnx_diagnostic.helpers import (
    string_type,
    string_sig,
    pretty_onnx,
    get_onnx_signature,
    flatten_object,
    max_diff,
    type_info,
    size_type,
    onnx_dtype_name,
    string_signature,
    make_hash,
    onnx_dtype_to_torch_dtype,
    np_dtype_to_tensor_dtype,
    torch_dtype_to_onnx_dtype,
    from_array_extended,
    convert_endian,
    from_array_ml_dtypes,
    dtype_to_tensor_dtype,
    string_diff,
    rename_dynamic_dimensions,
    rename_dynamic_expression,
)
from onnx_diagnostic.cache_helpers import make_dynamic_cache

TFLOAT = onnx.TensorProto.FLOAT


class TestHelpers(ExtTestCase):
    @skipif_ci_windows("numpy does not choose the same default type on windows and linux")
    def test_string_type(self):
        a = np.array([1])
        obj = {"a": a, "b": [5.6], "c": (1,)}
        s = string_type(obj)
        self.assertEqual(s, "dict(a:A7r1,b:#1[float],c:(int,))")

    def test_string_dict(self):
        a = np.array([1], dtype=np.float32)
        obj = {"a": a, "b": {"r": 5.6}, "c": {1}}
        s = string_type(obj)
        self.assertEqual(s, "dict(a:A1r1,b:dict(r:float),c:{int})")

    def test_string_type_array(self):
        a = np.array([1], dtype=np.float32)
        t = torch.tensor([1])
        obj = {"a": a, "b": t}
        s = string_type(obj, with_shape=False)
        self.assertEqual(s, "dict(a:A1r1,b:T7r1)")
        s = string_type(obj, with_shape=True)
        self.assertEqual(s, "dict(a:A1s1,b:T7s1)")

    def test_string_sig_f(self):
        def f(a, b=3, c=4, e=5):
            pass

        ssig = string_sig(f, {"a": 1, "c": 8, "b": 3})
        self.assertEqual(ssig, "f(a=1, c=8)")

    def test_string_sig_cls(self):
        class A:
            def __init__(self, a, b=3, c=4, e=5):
                self.a, self.b, self.c, self.e = a, b, c, e

        ssig = string_sig(A(1, c=8))
        self.assertEqual(ssig, "A(a=1, c=8)")

    def test_pretty_onnx(self):
        proto = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Sigmoid", ["Y"], ["sy"]),
                    oh.make_node("Mul", ["Y", "sy"], ["ysy"]),
                    oh.make_node("Mul", ["X", "ysy"], ["final"]),
                ],
                "nd",
                [
                    oh.make_tensor_value_info("X", TFLOAT, [1, "b", "c"]),
                    oh.make_tensor_value_info("Y", TFLOAT, ["a", "b", "c"]),
                ],
                [oh.make_tensor_value_info("final", TFLOAT, ["a", "b", "c"])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        pretty_onnx(proto, shape_inference=True)
        pretty_onnx(proto.graph.input[0])
        pretty_onnx(proto.graph)
        pretty_onnx(proto.graph.node[0])

    @hide_stdout()
    def test_print_pretty_onnx(self):
        proto = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Sigmoid", ["Y"], ["sy"]),
                    oh.make_node("Mul", ["Y", "sy"], ["ysy"]),
                    oh.make_node("Mul", ["X", "ysy"], ["final"]),
                ],
                "nd",
                [
                    oh.make_tensor_value_info("X", TFLOAT, [1, "b", "c"]),
                    oh.make_tensor_value_info("Y", TFLOAT, ["a", "b", "c"]),
                ],
                [oh.make_tensor_value_info("final", TFLOAT, ["a", "b", "c"])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        self.print_onnx(proto)
        self.print_model(proto)
        self.dump_onnx("test_print_pretty_onnx", proto)
        self.check_ort(proto)
        self.assertNotEmpty(proto)
        self.assertEmpty(None)

    def test_get_onnx_signature(self):
        proto = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Sigmoid", ["Y"], ["sy"]),
                    oh.make_node("Mul", ["Y", "sy"], ["ysy"]),
                    oh.make_node("Mul", ["X", "ysy"], ["final"]),
                ],
                "nd",
                [
                    oh.make_tensor_value_info("X", TFLOAT, [1, "b", "c"]),
                    oh.make_tensor_value_info("Y", TFLOAT, ["a", "b", "c"]),
                ],
                [oh.make_tensor_value_info("final", TFLOAT, ["a", "b", "c"])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )
        sig = get_onnx_signature(proto)
        self.assertEqual(sig, (("X", 1, (1, "b", "c")), ("Y", 1, ("a", "b", "c"))))

    @hide_stdout()
    def test_flatten(self):
        inputs = (
            torch.rand((3, 4), dtype=torch.float16),
            [
                torch.rand((5, 6), dtype=torch.float16),
                torch.rand((5, 6, 7), dtype=torch.float16),
                {
                    "a": torch.rand((2,), dtype=torch.float16),
                    "cache": make_dynamic_cache(
                        [(torch.rand((4, 4, 4)), torch.rand((4, 4, 4)))]
                    ),
                },
            ],
        )
        flat = flatten_object(inputs, drop_keys=True)
        diff = max_diff(inputs, flat, flatten=True, verbose=10)
        self.assertEqual(diff["abs"], 0)
        d = string_diff(diff)
        self.assertIsInstance(d, str)

    @hide_stdout()
    def test_max_diff_verbose(self):
        inputs = (
            torch.rand((3, 4), dtype=torch.float16),
            [
                torch.rand((5, 6), dtype=torch.float16),
                torch.rand((5, 6, 7), dtype=torch.float16),
            ],
        )
        flat = flatten_object(inputs)
        diff = max_diff(inputs, flat, flatten=True, verbose=10)
        self.assertEqual(diff["abs"], 0)
        d = string_diff(diff)
        self.assertIsInstance(d, str)

    def test_type_info(self):
        for tt in [
            onnx.TensorProto.FLOAT,
            onnx.TensorProto.FLOAT16,
            onnx.TensorProto.DOUBLE,
            onnx.TensorProto.BFLOAT16,
            onnx.TensorProto.INT32,
            onnx.TensorProto.INT64,
        ]:
            type_info(tt, "min")
            type_info(tt, "max")

    def test_size_type_onnx(self):
        for i in range(1, 40):
            with self.subTest(i=i):
                try:
                    name = onnx_dtype_name(i)
                except ValueError:
                    continue
                if name in {"NAME_FIELD_NUMBER"}:
                    continue
                if name not in {"STRING", "UINT4", "INT4", "FLOAT4E2M1"}:
                    size_type(i)

                if name not in {
                    "STRING",
                    "UINT4",
                    "INT4",
                    "FLOAT4E2M1",
                    "FLOAT8E5M2FNUZ",
                    "FLOAT8E5M2",
                    "FLOAT8E4M3FN",
                    "FLOAT8E4M3FNUZ",
                }:
                    onnx_dtype_to_torch_dtype(i)

    def test_size_type_numpy(self):
        for dt in {
            np.float32,
            np.float64,
            np.float16,
            np.int32,
            np.int64,
            np.int8,
            np.int16,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
        }:
            size_type(dt)
            np_dtype_to_tensor_dtype(dt)

    def test_from_array(self):
        for dt in {
            np.float32,
            np.float64,
            np.float16,
            np.int32,
            np.int64,
            np.int8,
            np.int16,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
        }:
            t = np.random.rand(4, 3).astype(dt)
            proto = from_array_extended(t)
            self.assertIsInstance(proto, onnx.TensorProto)
            convert_endian(proto)
            dtype_to_tensor_dtype(dt)

    def test_from_array_ml_dtypes(self):
        for dt in {
            ml_dtypes.bfloat16,
        }:
            t = np.random.rand(4, 3).astype(dt)
            from_array_ml_dtypes(t)
            from_array_extended(t)

    def test_size_type_mldtypes(self):
        for dt in {
            ml_dtypes.bfloat16,
        }:
            size_type(dt)
            np_dtype_to_tensor_dtype(dt)
            dtype_to_tensor_dtype(dt)

    def test_size_type_torch(self):
        for dt in {
            torch.float32,
            torch.float64,
            torch.float16,
            torch.int32,
            torch.int64,
            torch.int8,
            torch.int16,
            torch.uint8,
            torch.uint16,
            torch.uint32,
            torch.uint64,
        }:
            size_type(dt)
            torch_dtype_to_onnx_dtype(dt)
            dtype_to_tensor_dtype(dt)

    def test_string_signature(self):
        sig = string_signature(inspect.signature(string_signature))
        self.assertIn("sig: typing.Any", sig)

    def test_make_hash(self):
        self.assertIsInstance(make_hash([]), str)

    def test_string_type_one(self):
        self.assertEqual(string_type(None), "None")
        self.assertEqual(string_type([4]), "#1[int]")
        self.assertEqual(string_type((4, 5)), "(int,int)")
        self.assertEqual(string_type([4] * 100), "#100[int,...]")
        self.assertEqual(string_type((4,) * 100), "#100(int,...)")

    def test_string_type_one_with_min_max_int(self):
        self.assertEqual(string_type(None, with_min_max=True), "None")
        self.assertEqual(string_type([4], with_min_max=True), "#1[int=4]")
        self.assertEqual(string_type((4, 5), with_min_max=True), "(int=4,int=5)")
        self.assertEqual(string_type([4] * 100, with_min_max=True), "#100[int=4,...][4,4:4.0]")
        self.assertEqual(
            string_type((4,) * 100, with_min_max=True), "#100(int=4,...)[4,4:A[4.0]]"
        )

    def test_string_type_one_with_min_max_bool(self):
        self.assertEqual(string_type(None, with_min_max=True), "None")
        self.assertEqual(string_type([True], with_min_max=True), "#1[bool=True]")
        self.assertEqual(string_type((True, True), with_min_max=True), "(bool=True,bool=True)")
        self.assertEqual(
            string_type([True] * 100, with_min_max=True), "#100[bool=True,...][True,True:1.0]"
        )
        self.assertEqual(
            string_type((True,) * 100, with_min_max=True),
            "#100(bool=True,...)[True,True:A[1.0]]",
        )

    def test_string_type_one_with_min_max_float(self):
        self.assertEqual(string_type(None, with_min_max=True), "None")
        self.assertEqual(string_type([4.5], with_min_max=True), "#1[float=4.5]")
        self.assertEqual(string_type((4.5, 5.5), with_min_max=True), "(float=4.5,float=5.5)")
        self.assertEqual(
            string_type([4.5] * 100, with_min_max=True), "#100[float=4.5,...][4.5,4.5:4.5]"
        )
        self.assertEqual(
            string_type((4.5,) * 100, with_min_max=True), "#100(float=4.5,...)[4.5,4.5:A[4.5]]"
        )

    def test_string_type_at(self):
        self.assertEqual(string_type(None), "None")
        a = np.array([4, 5], dtype=np.float32)
        t = torch.tensor([4, 5], dtype=torch.float32)
        self.assertEqual(string_type([a]), "#1[A1r1]")
        self.assertEqual(string_type([t]), "#1[T1r1]")
        self.assertEqual(string_type((a,)), "(A1r1,)")
        self.assertEqual(string_type((t,)), "(T1r1,)")
        self.assertEqual(string_type([a] * 100), "#100[A1r1,...]")
        self.assertEqual(string_type([t] * 100), "#100[T1r1,...]")
        self.assertEqual(string_type((a,) * 100), "#100(A1r1,...)")
        self.assertEqual(string_type((t,) * 100), "#100(T1r1,...)")

    def test_string_type_at_with_shape(self):
        self.assertEqual(string_type(None), "None")
        a = np.array([4, 5], dtype=np.float32)
        t = torch.tensor([4, 5], dtype=torch.float32)
        self.assertEqual(string_type([a], with_shape=True), "#1[A1s2]")
        self.assertEqual(string_type([t], with_shape=True), "#1[T1s2]")
        self.assertEqual(string_type((a,), with_shape=True), "(A1s2,)")
        self.assertEqual(string_type((t,), with_shape=True), "(T1s2,)")
        self.assertEqual(string_type([a] * 100, with_shape=True), "#100[A1s2,...]")
        self.assertEqual(string_type([t] * 100, with_shape=True), "#100[T1s2,...]")
        self.assertEqual(string_type((a,) * 100, with_shape=True), "#100(A1s2,...)")
        self.assertEqual(string_type((t,) * 100, with_shape=True), "#100(T1s2,...)")

    def test_string_type_at_with_shape_min_max(self):
        self.assertEqual(string_type(None), "None")
        a = np.array([4, 5], dtype=np.float32)
        t = torch.tensor([4, 5], dtype=torch.float32)
        self.assertEqual(
            string_type([a], with_shape=True, with_min_max=True), "#1[A1s2[4.0,5.0:A4.5]]"
        )
        self.assertEqual(
            string_type([t], with_shape=True, with_min_max=True), "#1[T1s2[4.0,5.0:A4.5]]"
        )
        self.assertEqual(
            string_type((a,), with_shape=True, with_min_max=True), "(A1s2[4.0,5.0:A4.5],)"
        )
        self.assertEqual(
            string_type((t,), with_shape=True, with_min_max=True), "(T1s2[4.0,5.0:A4.5],)"
        )
        self.assertEqual(
            string_type([a] * 100, with_shape=True, with_min_max=True),
            "#100[A1s2[4.0,5.0:A4.5],...]",
        )
        self.assertEqual(
            string_type([t] * 100, with_shape=True, with_min_max=True),
            "#100[T1s2[4.0,5.0:A4.5],...]",
        )
        self.assertEqual(
            string_type((a,) * 100, with_shape=True, with_min_max=True),
            "#100(A1s2[4.0,5.0:A4.5],...)",
        )
        self.assertEqual(
            string_type((t,) * 100, with_shape=True, with_min_max=True),
            "#100(T1s2[4.0,5.0:A4.5],...)",
        )

    def test_pretty_onnx_att(self):
        node = oh.make_node("Cast", ["xm2c"], ["xm2"], to=1)
        pretty_onnx(node.attribute[0])

    def test_rename_dimension(self):
        res = rename_dynamic_dimensions(
            {"a": {"B", "C"}},
            {
                "B",
            },
        )
        self.assertEqual(res, {"B": "B", "a": "B", "C": "B"})

    def test_rename_dynamic_expression(self):
        text = rename_dynamic_expression("a * 10 - a", {"a": "x"})
        self.assertEqual(text, "x * 10 - x")


if __name__ == "__main__":
    unittest.main(verbosity=2)
