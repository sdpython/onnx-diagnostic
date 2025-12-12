import unittest
from typing import Tuple
import torch
from onnx_diagnostic.ext_test_case import ExtTestCase, requires_torch
from onnx_diagnostic.export.control_flow_onnx import (
    enable_code_export_control_flow,
)
from onnx_diagnostic.export.cf_simple_loop_for import simple_loop_for, SimpleLoopForOp


class TestCfSimpleLoopFor(ExtTestCase):
    @requires_torch("2.9.99")
    def test_simple_loop_for_int(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                def body(i: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor]:
                    return (x[: i.item() + 1].unsqueeze(1),)

                return simple_loop_for(4, body, (x,))

        model = Model()
        x = torch.arange(10, dtype=torch.float32)
        expected = torch.tensor([0, 0, 1, 0, 1, 2, 0, 1, 2, 3], dtype=x.dtype).unsqueeze(1)
        got = model(x)
        self.assertEqualArray(expected, got)

        with enable_code_export_control_flow():
            got = model(x)
        self.assertEqualArray(expected, got)

        ep = torch.export.export(
            model, (x,), dynamic_shapes=(({0: torch.export.Dim.DYNAMIC},))
        )
        check = []
        for node in ep.graph.nodes:
            if isinstance(node.target, SimpleLoopForOp):
                check.append(node)
        # Loop should be unrolled.
        self.assertEqual(len(check), 0)

    @requires_torch("2.9.99")
    def test_simple_loop_for_no_inputs(self):
        class Model(torch.nn.Module):
            def forward(self, n_iter, x):
                def body(i: torch.Tensor) -> Tuple[torch.Tensor]:
                    return (torch.arange(i + 1, dtype=torch.int64),)

                y = simple_loop_for(n_iter, body)
                torch._check(isinstance(y, torch.Tensor), lambda: f"y is {type(y)}")
                return x.unsqueeze(1) + y.unsqueeze(0).to(x.device)

        model = Model()
        n_iter = torch.tensor(4, dtype=torch.int64)
        x = torch.arange(8, dtype=torch.float32)
        expected = x.reshape((-1, 1)) + torch.tensor(
            [[0, 0, 1, 0, 1, 2, 0, 1, 2, 3]], dtype=x.dtype
        )
        got = model(n_iter, x)
        self.assertEqualArray(expected, got)

        with enable_code_export_control_flow():
            got = model(n_iter, x)
        self.assertEqualArray(expected, got)

        ep = torch.export.export(
            model, (n_iter, x), dynamic_shapes=({}, ({0: torch.export.Dim.DYNAMIC}))
        )
        check = []
        for node in ep.graph.nodes:
            if isinstance(node.target, SimpleLoopForOp):
                check.append(node)
        self.assertEqual(len(check), 1)

    @requires_torch("2.9.99")
    def test_simple_loop_for_1(self):
        class Model(torch.nn.Module):
            def forward(self, n_iter, x):
                def body(i: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor]:
                    return (x[: i.item() + 1].unsqueeze(1),)

                return simple_loop_for(n_iter, body, (x,))

        model = Model()
        n_iter = torch.tensor(4, dtype=torch.int64)
        x = torch.arange(10, dtype=torch.float32)
        expected = torch.tensor([0, 0, 1, 0, 1, 2, 0, 1, 2, 3], dtype=x.dtype).unsqueeze(1)
        got = model(n_iter, x)
        self.assertEqualArray(expected, got)

        with enable_code_export_control_flow():
            got = model(n_iter, x)
        self.assertEqualArray(expected, got)

        ep = torch.export.export(
            model, (n_iter, x), dynamic_shapes=({}, ({0: torch.export.Dim.DYNAMIC}))
        )
        check = []
        for node in ep.graph.nodes:
            if isinstance(node.target, SimpleLoopForOp):
                check.append(node)
        self.assertEqual(len(check), 1)

    @requires_torch("2.9.99")
    def test_simple_loop_for_2(self):
        class Model(torch.nn.Module):
            def forward(self, n_iter, x):
                def body(i: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor]:
                    return (x[: i.item() + 1].unsqueeze(1), x[i.item() + 1 :].unsqueeze(1))

                return simple_loop_for(n_iter, body, (x,))

        model = Model()
        n_iter = torch.tensor(4, dtype=torch.int64)
        x = torch.arange(10, dtype=torch.float32)
        expected = (
            torch.tensor([0, 0, 1, 0, 1, 2, 0, 1, 2, 3], dtype=x.dtype).unsqueeze(1),
            torch.tensor(
                [
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                ],
                dtype=x.dtype,
            ).unsqueeze(1),
        )
        got = model(n_iter, x)
        self.assertEqualArray(expected[0], got[0])
        self.assertEqualArray(expected[1], got[1])

        with enable_code_export_control_flow():
            got = model(n_iter, x)
        self.assertEqualArray(expected[0], got[0])
        self.assertEqualArray(expected[1], got[1])

        ep = torch.export.export(
            model, (n_iter, x), dynamic_shapes=({}, ({0: torch.export.Dim.DYNAMIC}))
        )
        check = []
        for node in ep.graph.nodes:
            if isinstance(node.target, SimpleLoopForOp):
                check.append(node)
        self.assertEqual(len(check), 1)

    @requires_torch("2.9.99")
    def test_simple_loop_for_2_concatenation_dims(self):
        class Model(torch.nn.Module):
            def forward(self, n_iter, x):
                def body(i: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor]:
                    return (x[: i.item() + 1].unsqueeze(1), x[i.item() + 1 :].unsqueeze(0))

                return simple_loop_for(n_iter, body, (x,), (0, 1))

        model = Model()
        n_iter = torch.tensor(4, dtype=torch.int64)
        x = torch.arange(10, dtype=torch.float32)
        expected = (
            torch.tensor([0, 0, 1, 0, 1, 2, 0, 1, 2, 3], dtype=x.dtype).unsqueeze(1),
            torch.tensor(
                [
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                ],
                dtype=x.dtype,
            ).unsqueeze(0),
        )
        got = model(n_iter, x)
        self.assertEqualArray(expected[0], got[0])
        self.assertEqualArray(expected[1], got[1])

        with enable_code_export_control_flow():
            got = model(n_iter, x)
        self.assertEqualArray(expected[0], got[0])
        self.assertEqualArray(expected[1], got[1])

        ep = torch.export.export(
            model, (n_iter, x), dynamic_shapes=({}, ({0: torch.export.Dim.DYNAMIC}))
        )
        check = []
        for node in ep.graph.nodes:
            if isinstance(node.target, SimpleLoopForOp):
                check.append(node)
        self.assertEqual(len(check), 1)

    @requires_torch("2.9.99")
    def test_simple_loop_for_1_with_concatenation_dims(self):
        class Model(torch.nn.Module):
            def forward(self, n_iter, x):
                def body(i: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor]:
                    return (x[: i.item() + 1].unsqueeze(0),)

                return simple_loop_for(n_iter, body, (x,), 1)

        model = Model()
        n_iter = torch.tensor(4, dtype=torch.int64)
        x = torch.arange(10, dtype=torch.float32)
        expected = torch.tensor([0, 0, 1, 0, 1, 2, 0, 1, 2, 3], dtype=x.dtype).unsqueeze(0)
        got = model(n_iter, x)
        self.assertEqualArray(expected, got)

        with enable_code_export_control_flow():
            got = model(n_iter, x)
        self.assertEqualArray(expected, got)

        ep = torch.export.export(
            model, (n_iter, x), dynamic_shapes=({}, ({0: torch.export.Dim.DYNAMIC}))
        )
        check = []
        for node in ep.graph.nodes:
            if isinstance(node.target, SimpleLoopForOp):
                check.append(node)
        self.assertEqual(len(check), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
