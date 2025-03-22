import sys
import unittest
import warnings
from typing import Any
import numpy
import onnx.backend.base
import onnx.backend.test
import onnx.shape_inference
import onnx.version_converter
from onnx import ModelProto
from onnx.backend.base import Device, DeviceType
from onnx.backend.test.runner.item import TestItem
from onnx_diagnostic.reference import ExtendedReferenceEvaluator


class ExtendedReferenceEvaluatorBackendRep(onnx.backend.base.BackendRep):
    def __init__(self, session):
        self._session = session

    def run(self, inputs, **kwargs):
        if isinstance(inputs, numpy.ndarray):
            inputs = [inputs]
        if isinstance(inputs, list):
            if len(inputs) == len(self._session.input_names):
                feeds = dict(zip(self._session.input_names, inputs))
            else:
                feeds = {}
                pos_inputs = 0
                for inp, tshape in zip(self._session.input_names, self._session.input_types):
                    shape = tuple(d.dim_value for d in tshape.tensor_type.shape.dim)
                    if shape == inputs[pos_inputs].shape:
                        feeds[inp] = inputs[pos_inputs]
                        pos_inputs += 1
                        if pos_inputs >= len(inputs):
                            break
        elif isinstance(inputs, dict):
            feeds = inputs
        else:
            raise TypeError(f"Unexpected input type {type(inputs)!r}.")
        outs = self._session.run(None, feeds)
        return outs


class ExtendedReferenceEvaluatorBackend(onnx.backend.base.Backend):
    @classmethod
    def is_opset_supported(cls, model):  # pylint: disable=unused-argument
        return True, ""

    @classmethod
    def supports_device(cls, device: str) -> bool:
        d = Device(device)
        return d.type == DeviceType.CPU  # type: ignore[no-any-return]

    @classmethod
    def create_inference_session(cls, model):
        return ExtendedReferenceEvaluator(model)

    @classmethod
    def prepare(
        cls, model: Any, device: str = "CPU", **kwargs: Any
    ) -> ExtendedReferenceEvaluatorBackendRep:
        if isinstance(model, ExtendedReferenceEvaluator):
            return ExtendedReferenceEvaluatorBackendRep(model)
        if isinstance(model, (str, bytes, ModelProto)):
            inf = cls.create_inference_session(model)
            return cls.prepare(inf, device, **kwargs)
        raise TypeError(f"Unexpected type {type(model)} for model.")

    @classmethod
    def run_model(cls, model, inputs, device=None, **kwargs):
        rep = cls.prepare(model, device, **kwargs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return rep.run(inputs, **kwargs)

    @classmethod
    def run_node(cls, node, inputs, device=None, outputs_info=None, **kwargs):
        raise NotImplementedError("Unable to run the model node by node.")


class RunnerTest(onnx.backend.test.BackendTest):
    @property
    def _filtered_test_items(self) -> dict[str, dict[str, TestItem]]:
        filtered: dict[str, dict[str, TestItem]] = {}
        for category, items_map in self._test_items.items():
            filtered[category] = {}
            for name, item in items_map.items():
                if name.endswith("_cuda"):
                    continue
                if not name.startswith(
                    (
                        "test_cast_like",
                        "test_concat",
                        "test_constant",
                        "test_gather",
                        "test_scatter_elements",
                        "test_slice",
                        "test_softmax",
                    )
                ):
                    continue
                filtered[category][name] = item
        return filtered


dft_atol = 1e-3 if sys.platform != "linux" else 1e-5
backend_test = RunnerTest(
    ExtendedReferenceEvaluatorBackend,
    __name__,
    test_kwargs={
        "test_dft": {"atol": dft_atol},
        "test_dft_axis": {"atol": dft_atol},
        "test_dft_axis_opset19": {"atol": dft_atol},
        "test_dft_inverse": {"atol": dft_atol},
        "test_dft_inverse_opset19": {"atol": dft_atol},
        "test_dft_opset19": {"atol": dft_atol},
    },
)


# import all test cases at global scope to make them visible to python.unittest
globals().update(backend_test.test_cases)

if __name__ == "__main__":
    res = unittest.main(verbosity=2, exit=False)
    tests_run = res.result.testsRun
    errors = len(res.result.errors)
    skipped = len(res.result.skipped)
    unexpected_successes = len(res.result.unexpectedSuccesses)
    expected_failures = len(res.result.expectedFailures)
    print("---------------------------------")
    print(
        f"tests_run={tests_run} errors={errors} skipped={skipped} "
        f"unexpected_successes={unexpected_successes} "
        f"expected_failures={expected_failures}"
    )
