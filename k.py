import onnx
import onnx.helper as oh
import torch
from onnx_diagnostic.helpers import string_type
from onnx_diagnostic.reference import TorchOnnxEvaluator

TFLOAT = onnx.TensorProto.FLOAT

proto = oh.make_model(
    oh.make_graph(
        [
            oh.make_node("Sigmoid", ["Y"], ["sy"]),
            oh.make_node("Mul", ["Y", "sy"], ["ysy"]),
            oh.make_node("Mul", ["X", "ysy"], ["final"]),
        ],
        "-nd-",
        [
            oh.make_tensor_value_info("X", TFLOAT, [1, "b", "c"]),
            oh.make_tensor_value_info("Y", TFLOAT, ["a", "b", "c"]),
        ],
        [oh.make_tensor_value_info("final", TFLOAT, ["a", "b", "c"])],
    ),
    opset_imports=[oh.make_opsetid("", 18)],
    ir_version=9,
)

sess = TorchOnnxEvaluator(proto, verbose=1)
feeds = dict(X=torch.rand((4, 5)), Y=torch.rand((4, 5)))
result = sess.run(None, feeds)
print(string_type(result, with_shape=True, with_min_max=True))
