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
from onnx_diagnostic.reference import OnnxruntimeEvaluator

ORT_OPSET = max(21, onnx_opset_version() - 2)


class OnnxruntimeEvaluatorBackendRep(onnx.backend.base.BackendRep):
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


class OnnxruntimeEvaluatorBackend(onnx.backend.base.Backend):
    @classmethod
    def is_compatible(cls, model) -> bool:
        return all(not (d.domain == "" and d.version > ORT_OPSET) for d in model.opset_import)

    @classmethod
    def supports_device(cls, device: str) -> bool:
        d = Device(device)
        if d.type == DeviceType.CPU:
            return True
        if d.type == DeviceType.CUDA:
            import torch

            return torch.cuda.is_available()
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
        return OnnxruntimeEvaluator(model, providers=providers)

    @classmethod
    def prepare(
        cls, model: Any, device: str = "CPU", **kwargs: Any
    ) -> OnnxruntimeEvaluatorBackendRep:
        if isinstance(model, OnnxruntimeEvaluator):
            return OnnxruntimeEvaluatorBackendRep(model)
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
    OnnxruntimeEvaluatorBackend,
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

# rtol=inf does not work
backend_test.exclude("(test_dft|test_stft)")

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
backend_test.exclude(
    "(test_gradient"
    "|test_if_opt"
    "|test_loop16_seq_none"
    "|test_range_float_type_positive_delta_expanded"
    "|test_range_int32_type_negative_delta_expanded"
    "|test_scan_sum)"
)

if onnx_opset_version() < 21:
    backend_test.exclude(
        "(test_averagepool_2d_dilations"
        "|test_if*"
        "|test_loop*"
        "|test_scan*"
        "|test_sequence_map*"
        "|test_cast_FLOAT_to_STRING|"
        "test_castlike_FLOAT_to_STRING|test_strnorm|"
        "test_center_crop_pad_crop_axes_hwc_expanded|"
        "test_lppool_2d_dilations|test_eyelike_without_dtype)"
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
    "(test_cast_FLOAT_to_INT4"
    "|test_cast_FLOAT16_to_INT4"
    "|test_cast_INT4_to_"
    "|test_castlike_INT4_to_"
    "|test_cast_FLOAT_to_UINT4"
    "|test_cast_FLOAT16_to_UINT4"
    "|test_cast_UINT4_to_"
    "|test_castlike_UINT4_to_)"
)

backend_test.exclude(
    "(test_regex_full_match|"
    "test_adagrad|"
    "test_adam|"
    "test_add_uint8|"
    "test_ai_onnx_ml_label_encoder_string|"
    "test_ai_onnx_ml_label_encoder_tensor_mapping|"
    "test_ai_onnx_ml_label_encoder_tensor_value_only_mapping|"
    "test_AvgPool|"
    "test_BatchNorm|"
    "test_bitshift_[a-z]+_uint16|"
    "test_center_crop_pad_crop|"
    "test_clip_[0-9a-z_]*expanded|"
    "test_elu_[0-9a-z_]*expanded|"
    "test_equal_string|"
    "test_GLU_|"
    "test_identity_opt|"
    "test_if|"
    "test_image|"
    "test_leakyrelu|"
    "test_((less)|(greater))_equal_bcast|"
    "test_((less)|(greater))[a-z_]*expanded|"
    "test_Linear|"
    "test_loop13|"
    "test_momentum|"
    "test_nesterov|"
    "test_((mul)|(min)|(max)|(div))_u?int((8)|(16))|"
    "test_operator|"
    "test_optional_|"
    "test_pow_types_float32_uint|"
    "test_qlinearmatmul|"
    "test_prelu|"
    "test_PReLU|"
    "test_reduce_max_empty|"
    "test_resize_downsample_scales|"
    "test_scatter_with_axis|"
    "test_scatter_without_axis"
    "|test_selu"
    "|test_sequence"
    "|test_shrink_"
    "|test_Softsign"
    "|test_split_to_sequence"
    "|test_string_concat"
    "|test_string_split"
    "|test_strnorm_model"
    "|test_strnormalizer"
    "|test_sub_uint8"
    "|test_thresholdedrelu"
    "|test_top_k_uint64"
    ")"
)

# failing on CI only
backend_test.exclude(
    "(_to_STRING|to_BFLOAT16|STRING_to|BFLOAT16_to|"
    "test_constant|test_(de)?quantizelinear_u?int4"
    "|test_identity_sequence"
    ")"
)

if onnx_opset_version() <= 24:
    backend_test.exclude(
        "(deform_conv"
        "|gru"
        "|lstm"
        "|l1normalization"
        "|l2normalization"
        "|lpnormalization"
        "|maxunpool"
        "|attention_3d"
        "|causal_expanded"
        ")"
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
