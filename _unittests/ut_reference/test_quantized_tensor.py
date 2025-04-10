import unittest
import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh
from onnx_diagnostic.ext_test_case import ExtTestCase
from onnx_diagnostic.reference import ExtendedReferenceEvaluator
from onnx_diagnostic.reference.quantized_tensor import QuantizedTensor
from onnx_diagnostic.reference.ops.op_qlinear_conv import QLinearConv

TFLOAT = onnx.TensorProto.FLOAT
TINT64 = onnx.TensorProto.INT64
_mkv_ = oh.make_tensor_value_info


class TestQuantizedOps(ExtTestCase):
    def test_qlinear_conv_2d(self):
        onnx.defs.register_schema(QLinearConv.op_schema)

        _x = np.array(
            [
                0.45246148109436035,
                0.15498268604278564,
                0.11199361085891724,
                -0.39421093463897705,
                0.2626858949661255,
                0.13414543867111206,
                -0.27184486389160156,
                -0.43028733134269714,
                -0.26825493574142456,
                0.3893144130706787,
                -0.13631996512413025,
                -0.009590476751327515,
                -0.48771554231643677,
                -0.25256502628326416,
                -0.2812897562980652,
                0.4043201804161072,
                0.07795023918151855,
                0.326981782913208,
                0.13114392757415771,
                -0.4416425824165344,
                0.12446999549865723,
                0.36739975214004517,
                0.1698915958404541,
                0.2008744478225708,
                0.23339951038360596,
                0.38613730669021606,
                0.11117297410964966,
                0.3877097964286804,
                0.20812749862670898,
                -0.34297940135002136,
                -0.029246658086776733,
                -0.20483523607254028,
                -0.19244328141212463,
                -0.11104947328567505,
                -0.32830488681793213,
                -0.01800677180290222,
                0.3618946671485901,
                -0.40949052572250366,
                -0.18248388171195984,
                -0.3349453806877136,
                -0.34091079235076904,
                0.006497859954833984,
                0.4537564516067505,
                0.08006560802459717,
                -0.14788749814033508,
                0.034442365169525146,
                -0.33322954177856445,
                0.06049239635467529,
                0.42619407176971436,
            ],
            dtype=np.float32,
        ).reshape((1, 1, 7, 7))

        _w = np.array([0.4406261742115021], dtype=np.float32).reshape((1, 1, 1, 1))

        _y = -np.array(
            [
                -0.19936637580394745,
                -0.06828942894935608,
                -0.04934731498360634,
                0.17369966208934784,
                -0.11574628204107285,
                -0.05910799279808998,
                0.1197819635272026,
                0.18959586322307587,
                0.1182001456618309,
                -0.17154212296009064,
                0.06006614491343498,
                0.0042258151806890965,
                0.21490024030208588,
                0.11128675937652588,
                0.12394362688064575,
                -0.17815405130386353,
                -0.034346915781497955,
                -0.14407673478126526,
                -0.05778544768691063,
                0.19459928572177887,
                -0.05484473705291748,
                -0.16188594698905945,
                -0.07485868036746979,
                -0.08851054310798645,
                -0.10284193605184555,
                -0.17014220356941223,
                -0.04898572340607643,
                -0.17083507776260376,
                -0.09170642495155334,
                0.1511256992816925,
                0.012886842712759972,
                0.09025576710700989,
                0.08479554951190948,
                0.0489313043653965,
                0.14465972781181335,
                0.007934254594147205,
                -0.15946026146411896,
                0.1804322451353073,
                0.08040717244148254,
                0.1475857049226761,
                0.15021422505378723,
                -0.0028631272725760937,
                -0.19993697106838226,
                -0.03527900204062462,
                0.06516310572624207,
                -0.015176207758486271,
                0.14682966470718384,
                -0.02665453404188156,
                -0.18779225647449493,
            ],
            dtype=np.float32,
        ).reshape(_x.shape)

        _x_ = _x
        _y_ = _y

        for channels_last in [0, 1]:
            with self.subTest(channels_last=channels_last):
                if channels_last:
                    _x = _x_.transpose((0, 2, 3, 1))
                    _y = _y_.transpose((0, 2, 3, 1))
                else:
                    _x = _x_
                    _y = _y_
                x = QuantizedTensor(_x)
                w = QuantizedTensor(_w)
                y = QuantizedTensor(_y)

                got = QLinearConv.eval(
                    x.qtensor,
                    x.scale,
                    x.zero_point,
                    w.qtensor,
                    w.scale,
                    w.zero_point,
                    y.scale,
                    y.zero_point,
                    auto_pad="NOTSET",
                    group=1,
                    channels_last=channels_last,
                )
                self.assertEqualArray(y.qtensor, got)

                # model part
                model = oh.make_model(
                    oh.make_graph(
                        [
                            oh.make_node(
                                "QuantizeLinear", ["x", "x_scale", "x_zero_point"], ["dqx"]
                            ),
                            oh.make_node(
                                "QuantizeLinear", ["w", "w_scale", "w_zero_point"], ["dqw"]
                            ),
                            oh.make_node(
                                "QLinearConv",
                                [
                                    "dqx",
                                    "x_scale",
                                    "x_zero_point",
                                    "dqw",
                                    "w_scale",
                                    "w_zero_point",
                                    "y_scale",
                                    "y_zero_point",
                                ],
                                ["qy"],
                                auto_pad="NOTSET",
                                group=1,
                                channels_last=channels_last,
                                domain="com.microsoft",
                            ),
                            oh.make_node(
                                "DequantizeLinear", ["qy", "y_scale", "y_zero_point"], ["y"]
                            ),
                        ],
                        "test",
                        [_mkv_("x", TFLOAT, _x.shape), _mkv_("w", TFLOAT, [1, 1, 1, 1])],
                        [_mkv_("y", TFLOAT, _y.shape)],
                        [
                            onh.from_array(x.scale, name="x_scale"),
                            onh.from_array(x.zero_point, name="x_zero_point"),
                            onh.from_array(w.scale, name="w_scale"),
                            onh.from_array(w.zero_point, name="w_zero_point"),
                            onh.from_array(y.scale, name="y_scale"),
                            onh.from_array(y.zero_point, name="y_zero_point"),
                        ],
                    ),
                    opset_imports=[
                        oh.make_opsetid("", 19),
                        oh.make_opsetid("com.microsoft", 1),
                    ],
                    ir_version=9,
                )
                ref = ExtendedReferenceEvaluator(model)
                feeds = dict(x=_x, w=_w)
                got = ref.run(None, feeds)
                self.assertEqualArray(_y, got[0], atol=1e-3)

                from onnxruntime import InferenceSession

                sess = InferenceSession(
                    model.SerializeToString(), providers=["CPUExecutionProvider"]
                )
                got = sess.run(None, feeds)
                self.assertEqualArray(_y, got[0], atol=1e-3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
