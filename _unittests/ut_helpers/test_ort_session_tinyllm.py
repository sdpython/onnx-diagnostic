import copy
import unittest
import numpy as np
import onnx
import torch
import onnxruntime
from onnxruntime.capi import _pybind_state as ORTC
from onnx_diagnostic.ext_test_case import ExtTestCase, hide_stdout, ignore_warnings
from onnx_diagnostic.helpers import max_diff
from onnx_diagnostic.rt_helper import make_feeds
from onnx_diagnostic.helpers.ort_session import (
    InferenceSessionForNumpy,
    InferenceSessionForTorch,
)
from onnx_diagnostic.torch_export_patches import bypass_export_some_errors
from onnx_diagnostic.torch_models.llms import get_tiny_llm
from onnx_diagnostic.reference import ExtendedReferenceEvaluator
from onnx_diagnostic.helpers.onnx_helper import np_dtype_to_tensor_dtype


class TestOrtSessionTinyLLM(ExtTestCase):

    def test_ort_value(self):
        val = np.array([30, 31, 32], dtype=np.int64)
        ort = ORTC.OrtValue.ortvalue_from_numpy_with_onnx_type(val, onnx.TensorProto.INT64)
        self.assertEqual(np_dtype_to_tensor_dtype(val.dtype), onnx.TensorProto.INT64)
        val2 = ort.numpy()
        self.assertEqualArray(val, val2)
        ort = ORTC.OrtValue.ortvalue_from_numpy_with_onnx_type(
            val, np_dtype_to_tensor_dtype(val.dtype)
        )
        val2 = ort.numpy()
        self.assertEqualArray(val, val2)

    def test_ort_value_py(self):
        data = get_tiny_llm()
        inputs = data["inputs"]
        feeds = make_feeds(
            ["input_ids", "attention_mask", "position_ids", "key0", "value0"],
            inputs,
            use_numpy=True,
            copy=True,
        )
        new_feeds = {}
        for k, v in feeds.items():
            new_feeds[k] = onnxruntime.OrtValue.ortvalue_from_numpy_with_onnx_type(
                v, np_dtype_to_tensor_dtype(v.dtype)
            )
        other_feeds = {k: v.numpy() for k, v in new_feeds.items()}
        self.assertEqualAny(feeds, other_feeds)

    def test_ort_value_more(self):
        data = get_tiny_llm()
        inputs = data["inputs"]
        feeds = make_feeds(
            ["input_ids", "attention_mask", "position_ids", "key0", "value0"],
            inputs,
            use_numpy=True,
            copy=True,
        )
        feeds = {
            k: feeds[k].copy()
            for k in ["input_ids", "attention_mask", "key0", "value0", "position_ids"]
        }
        new_feeds = {}
        for k, v in feeds.items():
            new_feeds[k] = ORTC.OrtValue.ortvalue_from_numpy_with_onnx_type(
                v, np_dtype_to_tensor_dtype(v.dtype)
            )
        other_feeds = {k: v.numpy() for k, v in new_feeds.items()}
        self.assertEqualAny(feeds, other_feeds)

    @ignore_warnings((UserWarning, DeprecationWarning, FutureWarning))
    @hide_stdout()
    def test_check_allruntimes_on_tiny_llm(self):
        data = get_tiny_llm()
        model, inputs, ds = data["model"], data["inputs"], data["dynamic_shapes"]
        expected = model(**copy.deepcopy(inputs))

        with bypass_export_some_errors(patch_transformers=True):
            ep = torch.onnx.export(
                model, (), kwargs=copy.deepcopy(inputs), dynamic_shapes=ds, dynamo=True
            )

        proto = ep.model_proto
        self.dump_onnx("test_check_allruntimes_on_tiny_llm.onnx", proto)
        feeds = make_feeds(proto, inputs, use_numpy=True, copy=True)
        sess = onnxruntime.InferenceSession(
            proto.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, feeds)
        self.assertLess(max_diff(expected, got, flatten=True)["abs"], 1e-5)

        sess = ExtendedReferenceEvaluator(proto)
        got = sess.run(None, feeds)
        self.assertLess(max_diff(expected, got, flatten=True)["abs"], 1e-5)
        all_outputs = sess.run(None, feeds, intermediate=True)
        if "linear_7" in all_outputs:
            self.assertEqualArray(got[0], all_outputs["linear_7"])

        sess = InferenceSessionForNumpy(proto)
        got = sess.run(None, feeds)
        self.assertLess(max_diff(expected, got, flatten=True)["abs"], 1e-5)

        feeds = make_feeds(proto, inputs, copy=True)
        sess = InferenceSessionForTorch(proto)
        got = sess.run(None, feeds)
        self.assertLess(max_diff(expected, got, flatten=True)["abs"], 1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
