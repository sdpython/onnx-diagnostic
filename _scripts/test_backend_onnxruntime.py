"""
This file runs through the backend test and evaluates onnxruntime.
"""

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
from onnx.defs import onnx_opset_version
import onnxruntime

ORT_OPSET = max(23, onnx_opset_version() - 2)


class OnnxruntimeBackendRep(onnx.backend.base.BackendRep):
    def __init__(self, session):
        self._session = session

    def run(self, inputs, **kwargs):
        if isinstance(inputs, numpy.ndarray):
            inputs = [inputs]
        if isinstance(inputs, list):
            if len(inputs) == len(self._session.get_inputs()):
                feeds = dict(zip([i.name for i in self._session.get_inputs()], inputs))
            else:
                input_names = [i.name for i in self._session.get_inputs()]
                feeds = {}
                pos_inputs = 0
                for inp, tshape in zip(input_names, self._session.input_types):
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


class OnnxruntimeBackend(onnx.backend.base.Backend):
    @classmethod
    def is_compatible(cls, model) -> bool:
        return all(not (d.domain == "" and d.version > ORT_OPSET) for d in model.opset_import)

    @classmethod
    def supports_device(cls, device: str) -> bool:
        d = Device(device)
        if d.type == DeviceType.CPU:
            return True
        # if d.type == DeviceType.CUDA:
        #     import torch
        #
        #     return torch.cuda.is_available()
        return False

    @classmethod
    def create_inference_session(cls, model, device):
        d = Device(device)
        if d.type == DeviceType.CUDA:
            providers = ["CUDAExecutionProvider"]
        elif d.type == DeviceType.CPU:
            providers = ["CPUExecutionProvider"]
        else:
            raise ValueError(f"Unrecognized device {device!r} or {d!r}")
        return onnxruntime.InferenceSession(model.SerializeToString(), providers=providers)

    @classmethod
    def prepare(cls, model: Any, device: str = "CPU", **kwargs: Any) -> OnnxruntimeBackendRep:
        if isinstance(model, onnxruntime.InferenceSession):
            return OnnxruntimeBackendRep(model)
        if isinstance(model, (str, bytes, ModelProto)):
            inf = cls.create_inference_session(model, device)
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


dft_atol = 1e-3
stft_atol = 1e-4
ql_atol = 1e-5
backend_test = onnx.backend.test.BackendTest(
    OnnxruntimeBackend,
    __name__,
    test_kwargs={
        "test_dft": {"atol": dft_atol, "rtol": numpy.inf},
        "test_dft_axis": {"atol": dft_atol, "rtol": numpy.inf},
        "test_dft_axis_opset19": {"atol": dft_atol, "rtol": numpy.inf},
        "test_dft_inverse": {"atol": dft_atol, "rtol": numpy.inf},
        "test_dft_inverse_opset19": {"atol": dft_atol, "rtol": numpy.inf},
        "test_dft_opset19": {"atol": dft_atol, "rtol": numpy.inf},
        "test_stft": {"atol": stft_atol, "rtol": numpy.inf},
        "test_stft_with_window": {"atol": stft_atol, "rtol": numpy.inf},
        "test_qlinearmatmul_2D_int8_float32": {"atol": ql_atol},
        "test_qlinearmatmul_3D_int8_float32": {"atol": ql_atol},
    },
)

# The following tests are too slow with the reference implementation (Conv).
backend_test.exclude(
    "(test_bvlc_alexnet"
    "|test_densenet121"
    "|test_inception_v1"
    "|test_inception_v2"
    "|test_resnet50"
    "|test_shufflenet"
    "|test_squeezenet"
    "|test_vgg19"
    "|test_zfnet512)"
)

# The following tests cannot pass because they consists in generating random number.
backend_test.exclude("(test_bernoulli|test_PoissonNLLLLoss)")

# The following tests are not supported.
backend_test.exclude("test_gradient")

backend_test.exclude("(test_adagrad|test_adam|test_add_uint8)")


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
