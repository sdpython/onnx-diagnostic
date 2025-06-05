from typing import Optional, Tuple
import onnx
import torch
from ...helpers.torch_helper import onnx_dtype_to_torch_dtype
from . import OpRun, OpRunTensor


class AveragePool_11(OpRun):
    "AveragePool"

    def __init__(self, node: onnx.NodeProto, version: Optional[int] = None):
        super().__init__(node, version)
        self.auto_pad = self.get_attribute_string(node, "auto_pad", "NOTSET")
        self.ceil_mode = bool(self.get_attribute_int(node, "ceil_mode", 0))
        self.count_include_pad = bool(self.get_attribute_int(node, "count_include_pad", 0))
        self.dilations = self.get_attribute_ints(node, "dilations", None)
        self.kernel_shape: Tuple[int, ...] = (
            self.get_attribute_ints(node, "kernel_shape") or tuple()
        )
        self.pads = self.get_attribute_ints(node, "pads", None)
        self.strides = self.get_attribute_ints(node, "strides", None)

    def run(self, x):
        kernel_shape = self.kernel_shape
        dilations = self.dilations or [1 for _ in x.shape[2:]]
        strides = self.strides or [1 for _ in x.shape[2:]]
        pads = self.pads or ([0 for _ in x.shape[2:]] * 2)
        assert (
            self.auto_pad == "NOTSET"
        ), f"conv not implemented for auto_pad={self.auto_pad!r}"
        assert len(set(pads)) == 1, f"conv not implemented for pads={pads}"
        assert set(dilations) == {1}, f"conv not implemented for dilations={dilations}"
        avg_pool = getattr(torch.nn.functional, f"avg_pool{len(kernel_shape)}d")
        return OpRunTensor(
            avg_pool(
                x.tensor,
                kernel_size=tuple(kernel_shape),
                stride=tuple(strides),
                padding=pads[0],
                ceil_mode=self.ceil_mode,
                count_include_pad=self.count_include_pad,
                # dilation=tuple(dilations),
            )
        )


class Conv_11(OpRun):
    "Conv"

    def __init__(self, node: onnx.NodeProto, version: Optional[int] = None):
        super().__init__(node, version)
        self.auto_pad = self.get_attribute_string(node, "auto_pad", "NOTSET")
        self.dilations = self.get_attribute_ints(node, "dilations", None)
        self.group = self.get_attribute_int(node, "group", 1)
        self.kernel_shape: Tuple[int, ...] = (
            self.get_attribute_ints(node, "kernel_shape") or tuple()
        )
        self.pads = self.get_attribute_ints(node, "pads", None)
        self.strides = self.get_attribute_ints(node, "strides", None)

    def run(self, x, w, b=None):
        kernel_shape = self.kernel_shape or w.shape[2:]
        assert (
            tuple(kernel_shape) == w.shape[-len(kernel_shape) :]
        ), f"conv not implemented for kernel_shape={kernel_shape} and w.shape={w.shape}"
        dilations = self.dilations or [1 for _ in x.shape[2:]]
        strides = self.strides or [1 for _ in x.shape[2:]]
        pads = self.pads or ([0 for _ in x.shape[2:]] * 2)
        assert (
            self.auto_pad == "NOTSET"
        ), f"conv not implemented for auto_pad={self.auto_pad!r}"
        assert len(set(pads)) == 1, f"conv not implemented for pads={pads}"
        if b is None:
            bias = None
        else:
            bias = b.tensor.squeeze()
            if not bias.shape:
                bias = bias.unsqueeze(0)
        return OpRunTensor(
            torch.nn.functional.conv2d(
                x.tensor,
                w.tensor,
                bias=bias,
                stride=tuple(strides),
                padding=pads[0],
                dilation=tuple(dilations),
                groups=self.group,
            )
        )


class LayerNormalization_17(OpRun):
    "LayerNormalization"

    def __init__(self, node: onnx.NodeProto, version: Optional[int] = None):
        super().__init__(node, version)
        self.axis = self.get_attribute_int(node, "axis", -1)
        self.epsilon = self.get_attribute_float(node, "epsilon", 1e-5)
        self.stash_type = onnx_dtype_to_torch_dtype(
            self.get_attribute_int(node, "stash_type", onnx.TensorProto.FLOAT)  # type: ignore[arg-type]
        )
        self.compute_std = len(node.output) > 1

    def run(self, x, scale, bias=None):
        original_dtype = x.dtype
        xt = x.tensor.to(self.stash_type)
        res = torch.nn.functional.layer_norm(
            xt,
            xt.shape[self.axis :],
            weight=scale.tensor,
            bias=None if bias is None else bias.tensor,
            eps=self.epsilon,
        )
        if not self.compute_std:
            return OpRunTensor(res.to(original_dtype))
        axes = tuple(range(len(xt.shape)))[self.axis :]
        mean, var = torch.var(xt, dim=axes, keepdim=False)
        x_inv_std_dev = torch.reciprocal(torch.sqrt(var + self.epsilon))
        return (
            OpRunTensor(res.to(original_dtype)),
            OpRunTensor(mean),
            OpRunTensor(x_inv_std_dev),
        )


class Softmax_13(OpRun):
    "Softmax"

    def __init__(self, node: onnx.NodeProto, version: Optional[int] = None):
        super().__init__(node, version)
        self.axis = self.get_attribute_int(node, "axis", -1)
        assert isinstance(self.axis, int), f"Unexpected value for attribute axis={self.axis!r}"
        # this is out of spec
        stash_type = self.get_attribute_int(node, "stash_type", None)
        self.stash_type = None if stash_type is None else onnx_dtype_to_torch_dtype(stash_type)

    def run(self, data: OpRunTensor) -> OpRunTensor:
        return OpRunTensor(
            torch.nn.functional.softmax(data.tensor, dim=self.axis, dtype=self.stash_type)
        )


class Tanh_6(OpRun):
    "Tanh"

    def run(self, data: OpRunTensor) -> OpRunTensor:
        return OpRunTensor(torch.nn.functional.tanh(data.tensor))
