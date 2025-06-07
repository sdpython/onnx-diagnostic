from typing import Dict, Optional, Tuple
import onnx
import onnx.helper as oh
import torch
from .torch_helper import onnx_dtype_to_torch_dtype, torch_dtype_to_onnx_dtype
from ..reference.torch_ops import OpRunKernel, OpRunTensor


class LayerNormalizationOrt(OpRunKernel):
    "LayerNormalization with onnxruntime"

    @classmethod
    def device_dependent(cls) -> bool:
        "Needs device."
        return False

    def __init__(
        self,
        node: onnx.NodeProto,
        version=None,
        device: Optional[torch.device] = None,
        verbose=0,
    ):
        super().__init__(node, version, verbose=verbose)
        self.axis = self.get_attribute_int(node, "axis", -1)
        self.epsilon = self.get_attribute_float(node, "epsilon", 1e-5)
        self.device = device
        self.stash_type = onnx_dtype_to_torch_dtype(
            self.get_attribute_int(node, "stash_type", onnx.TensorProto.FLOAT)  # type: ignore[arg-type]
        )
        self.compute_std = len(node.output) > 1
        assert not self.compute_std, (
            f"This kernel implementation only work when only one output "
            f"is required but {node.output} were."
        )
        self._cache: Dict[Tuple[int, int], onnx.ModelProto] = {}
        self.is_cpu = torch.device("cpu") == self.device

    def _make_model(self, itype: int, rank: int) -> onnx.ModelProto:
        shape = [*["d{i}" for i in range(rank - 1)], "last"]
        layer_model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node(
                        "LayerNormalization",
                        ["X", "W", "B"],
                        ["Z"],
                        axis=self.axis,
                        epsilon=self.epsilon,
                    )
                ],
                "dummy",
                [
                    oh.make_tensor_value_info("X", itype, shape),
                    oh.make_tensor_value_info("W", itype, ["last"]),
                    oh.make_tensor_value_info("B", itype, ["last"]),
                ],
                [oh.make_tensor_value_info("Z", itype, shape)],
            ),
            ir_version=9,
            opset_imports=[oh.make_opsetid("", 18)],
        )
        import onnxruntime

        provider = "CPUExecutionProvider" if self.is_cpu else "CUDAExecutionProvider"
        return onnxruntime.InferenceSession(
            layer_model.SerializeToString(), providers=[provider]
        )

    def run(self, x, scale, bias=None):
        itype = torch_dtype_to_onnx_dtype(x.dtype)
        rank = len(x.shape)
        key = itype, rank
        if key not in self._cache:
            self._cache[key] = self._make_model(itype, rank)
        sess = self._cache[key]
        feeds = dict(X=x, W=scale)
        if bias is not None:
            feeds["B"] = bias
        feeds = {k: v.tensor.detach().cpu().numpy() for k, v in feeds.items()}
        got = sess.run(None, feeds)[0]
        return OpRunTensor(torch.from_numpy(got).to(x.dtype).to(x.device))


class MatMulOrt(OpRunKernel):
    "MatMul with onnxruntime"

    @classmethod
    def device_dependent(cls) -> bool:
        "Needs device."
        return False

    def __init__(
        self,
        node: onnx.NodeProto,
        version=None,
        device: Optional[torch.device] = None,
        verbose=0,
    ):
        super().__init__(node, version, verbose=verbose)
        self.device = device
        self._cache: Dict[Tuple[int, int, int], onnx.ModelProto] = {}
        self.is_cpu = torch.device("cpu") == self.device

    def _make_model(self, itype: int, ranka: int, rankb: int) -> onnx.ModelProto:
        shapea = ["a{i}" for i in range(ranka)]
        shapeb = ["b{i}" for i in range(rankb)]
        shapec = ["c{i}" for i in range(max(ranka, rankb))]
        model = oh.make_model(
            oh.make_graph(
                [oh.make_node("MatMul", ["A", "B"], ["C"])],
                "dummy",
                [
                    oh.make_tensor_value_info("A", itype, shapea),
                    oh.make_tensor_value_info("B", itype, shapeb),
                ],
                [oh.make_tensor_value_info("C", itype, shapec)],
            ),
            ir_version=9,
            opset_imports=[oh.make_opsetid("", 17)],
        )
        import onnxruntime

        provider = "CPUExecutionProvider" if self.is_cpu else "CUDAExecutionProvider"
        return onnxruntime.InferenceSession(model.SerializeToString(), providers=[provider])

    def run(self, a, b):
        itype = torch_dtype_to_onnx_dtype(a.dtype)
        ranka, rankb = len(a.shape), len(b.shape)
        key = itype, ranka, rankb
        if key not in self._cache:
            self._cache[key] = self._make_model(itype, ranka, rankb)
        sess = self._cache[key]
        feeds = dict(A=a, B=b)
        feeds = {k: v.tensor.detach().cpu().numpy() for k, v in feeds.items()}
        got = sess.run(None, feeds)[0]
        return OpRunTensor(torch.from_numpy(got).to(a.dtype).to(a.device))
