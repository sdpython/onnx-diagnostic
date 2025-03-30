import copy
import unittest
import torch
import onnxruntime
from onnx_diagnostic.ext_test_case import ExtTestCase, hide_stdout, ignore_warnings
from onnx_diagnostic.helpers import max_diff
from onnx_diagnostic.helpers.ort_session import (
    InferenceSessionForNumpy,
    InferenceSessionForTorch,
    make_feeds,
)
from onnx_diagnostic.torch_export_patches import bypass_export_some_errors
from onnx_diagnostic.torch_models.llms import get_tiny_llm
from onnx_diagnostic.reference import ExtendedReferenceEvaluator


class TestOrtSessionTinyLLM(ExtTestCase):

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
        feeds = make_feeds(proto, inputs, use_numpy=True)
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
        got = sess.run(None, feeds, expected=all_outputs)
        self.assertLess(max_diff(expected, got, flatten=True)["abs"], 1e-5)

        feeds = make_feeds(proto, inputs)
        sess = InferenceSessionForTorch(proto)
        got = sess.run(None, feeds)
        self.assertLess(max_diff(expected, got, flatten=True)["abs"], 1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
