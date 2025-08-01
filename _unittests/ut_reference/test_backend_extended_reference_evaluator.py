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
from onnx.defs import onnx_opset_version
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
    def is_compatible(cls, model) -> bool:
        return True

    @classmethod
    def supports_device(cls, device: str) -> bool:
        d = Device(device)
        return d.type == DeviceType.CPU

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


dft_atol = 1e-3 if sys.platform != "linux" else 1e-5
backend_test = onnx.backend.test.BackendTest(
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
backend_test.exclude("(test_bernoulli)")

# The following tests are not supported.
backend_test.exclude(
    "(test_gradient"
    "|test_if_opt"
    "|test_loop16_seq_none"
    "|test_range_float_type_positive_delta_expanded"
    "|test_range_int32_type_negative_delta_expanded"
    "|test_scan_sum)"
)

# The following tests fail due to discrepancies (small but still higher than 1e-7).
backend_test.exclude("test_adam_multiple")  # 1e-2


if onnx_opset_version() < 19:
    backend_test.exclude(
        "(test_argm[ai][nx]_default_axis_example"
        "|test_argm[ai][nx]_default_axis_random"
        "|test_argm[ai][nx]_keepdims_example"
        "|test_argm[ai][nx]_keepdims_random"
        "|test_argm[ai][nx]_negative_axis_keepdims_example"
        "|test_argm[ai][nx]_negative_axis_keepdims_random"
        "|test_argm[ai][nx]_no_keepdims_example"
        "|test_argm[ai][nx]_no_keepdims_random"
        "|test_col2im_pads"
        "|test_gru_batchwise"
        "|test_gru_defaults"
        "|test_gru_seq_length"
        "|test_gru_with_initial_bias"
        "|test_layer_normalization_2d_axis1_expanded"
        "|test_layer_normalization_2d_axis_negative_1_expanded"
        "|test_layer_normalization_3d_axis1_epsilon_expanded"
        "|test_layer_normalization_3d_axis2_epsilon_expanded"
        "|test_layer_normalization_3d_axis_negative_1_epsilon_expanded"
        "|test_layer_normalization_3d_axis_negative_2_epsilon_expanded"
        "|test_layer_normalization_4d_axis1_expanded"
        "|test_layer_normalization_4d_axis2_expanded"
        "|test_layer_normalization_4d_axis3_expanded"
        "|test_layer_normalization_4d_axis_negative_1_expanded"
        "|test_layer_normalization_4d_axis_negative_2_expanded"
        "|test_layer_normalization_4d_axis_negative_3_expanded"
        "|test_layer_normalization_default_axis_expanded"
        "|test_logsoftmax_large_number_expanded"
        "|test_lstm_batchwise"
        "|test_lstm_defaults"
        "|test_lstm_with_initial_bias"
        "|test_lstm_with_peepholes"
        "|test_mvn"
        "|test_mvn_expanded"
        "|test_softmax_large_number_expanded"
        "|test_operator_reduced_mean"
        "|test_operator_reduced_mean_keepdim)"
    )

if onnx_opset_version() < 21:
    backend_test.exclude(
        "(test_averagepool_2d_dilations"
        "|test_if*"
        "|test_loop*"
        "|test_scan*"
        "|test_sequence_map*"
        ")"
    )

# The following tests are using types not supported by NumPy.
# They could be if method to_array is extended to support custom
# types the same as the reference implementation does
# (see onnx.reference.op_run.to_array_extended).
backend_test.exclude(
    "(test_cast_FLOAT_to_BFLOAT16"
    "|test_cast_BFLOAT16_to_FLOAT"
    "|test_cast_BFLOAT16_to_FLOAT"
    "|test_castlike_BFLOAT16_to_FLOAT"
    "|test_castlike_FLOAT_to_BFLOAT16"
    "|test_castlike_FLOAT_to_BFLOAT16_expanded"
    "|test_cast_no_saturate_"
    "|_to_FLOAT8"
    "|_FLOAT8"
    "|test_quantizelinear_e4m3fn"
    "|test_quantizelinear_e5m2"
    ")"
)

# Disable test about float 8
backend_test.exclude(
    "(test_castlike_BFLOAT16*"
    "|test_cast_BFLOAT16*"
    "|test_cast_no_saturate*"
    "|test_cast_FLOAT_to_FLOAT8*"
    "|test_cast_FLOAT16_to_FLOAT8*"
    "|test_cast_FLOAT8_to_*"
    "|test_castlike_BFLOAT16*"
    "|test_castlike_no_saturate*"
    "|test_castlike_FLOAT_to_FLOAT8*"
    "|test_castlike_FLOAT16_to_FLOAT8*"
    "|test_castlike_FLOAT8_to_*"
    "|test_quantizelinear_e*)"
)

# Disable test about INT 4
backend_test.exclude(
    "(test_cast_FLOAT_to_INT4*"
    "|test_cast_FLOAT16_to_INT4*"
    "|test_cast_INT4_to_*"
    "|test_castlike_INT4_to_*"
    "|test_cast_FLOAT_to_UINT4*"
    "|test_cast_FLOAT16_to_UINT4*"
    "|test_cast_UINT4_to_*"
    "|test_castlike_UINT4_to_*)"
)

backend_test.exclude("(test_regex_full_match*)")

backend_test.exclude("(test_scatter_with_axis*|test_scatter_without_axis*)")

if onnx_opset_version() < 21:
    # The following tests fail due to a bug in the backend test comparison.
    backend_test.exclude(
        "(test_cast_FLOAT_to_STRING|test_castlike_FLOAT_to_STRING|test_strnorm)"
    )

    # The following tests fail due to a shape mismatch.
    backend_test.exclude(
        "(test_center_crop_pad_crop_axes_hwc_expanded|test_lppool_2d_dilations)"
    )

    # The following tests fail due to a type mismatch.
    backend_test.exclude("(test_eyelike_without_dtype)")

if onnx_opset_version() <= 24:
    backend_test.exclude("(attention_3d|causal_expanded)")


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
