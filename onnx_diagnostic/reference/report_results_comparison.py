from typing import Any, Dict, List, Tuple, Union


ReportKeyNameType = Union[str, Tuple[str, int, str]]
ReportKeyValueType = Tuple[int, Tuple[int, ...]]


class ReportResultComparison:
    """
    Holds tensors a runtime can use as a reference to compare
    intermediate results.
    See :meth:`onnx_diagnostic.reference.TorchOnnxEvaluator.run`.

    :param tensors: tensor
    """

    def __init__(self, tensors: Dict[ReportKeyNameType, "torch.Tensor"]):  # noqa: F821
        from ..helpers.onnx_helper import dtype_to_tensor_dtype
        from ..helpers import max_diff, string_type

        assert all(
            hasattr(v, "shape") and hasattr(v, "dtype") for v in tensors.values()
        ), f"One of the tensors is not: {string_type(tensors, with_shape=True)}"
        self.dtype_to_tensor_dtype = dtype_to_tensor_dtype
        self.max_diff = max_diff
        self.tensors = tensors
        self._build_mapping()

    def key(self, tensor: "torch.Tensor") -> ReportKeyValueType:  # noqa: F821
        "Returns a key for a tensor, (onnx dtype, shape)."
        return self.dtype_to_tensor_dtype(tensor.dtype), tuple(map(int, tensor.shape))

    def _build_mapping(self):
        mapping = {}
        for k, v in self.tensors.items():
            key = self.key(v)
            if key not in mapping:
                mapping[key] = []
            mapping[key].append(k)
        self.mapping = mapping
        self.clear()

    def clear(self):
        """Clears the last report."""
        self.report_cmp = {}
        self.unique_run_names = set()

    @property
    def value(
        self,
    ) -> Dict[Tuple[Tuple[int, str], ReportKeyNameType], Dict[str, Union[float, str]]]:
        "Returns the report."
        return self.report_cmp

    @property
    def data(self) -> List[Dict[str, Any]]:
        "Returns data which can be consumed by a dataframe."
        rows = []
        for k, v in self.value.items():
            (i_run, run_name), ref_name = k
            d = dict(run_index=i_run, run_name=run_name, ref_name=ref_name)
            d.update(v)
            rows.append(d)
        return rows

    def report(
        self, outputs: Dict[str, "torch.Tensor"]  # noqa: F821
    ) -> List[Tuple[Tuple[int, str], ReportKeyNameType, Dict[str, Union[float, str]]]]:
        """
        For every tensor in outputs, compares it to every tensor held by
        this class if it shares the same type and shape. The function returns
        the results of the comparison. The function also collects the results
        into a dictionary the user can retrieve later.
        """
        res: List[Tuple[Tuple[int, str], ReportKeyNameType, Dict[str, Union[float, str]]]] = []
        for name, tensor in outputs.items():
            i_run = len(self.unique_run_names)
            self.unique_run_names.add(name)
            key = self.key(tensor)
            if key not in self.mapping:
                continue
            cache: Dict["torch.device", "torch.Tensor"] = {}  # noqa: F821, UP037
            for held_key in self.mapping[key]:
                t2 = self.tensors[held_key]
                if hasattr(t2, "device") and hasattr(tensor, "device"):
                    if t2.device in cache:
                        t = cache[t2.device]
                    else:
                        cache[t2.device] = t = tensor.to(t2.device)
                    diff = self.max_diff(t, t2)
                else:
                    diff = self.max_diff(tensor, t2)
                res.append((i_run, name, held_key, diff))  # type: ignore[arg-type]
                self.report_cmp[(i_run, name), held_key] = diff
        return res
