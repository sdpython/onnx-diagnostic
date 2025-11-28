import unittest
import pandas
import onnx
from onnx_diagnostic.ext_test_case import (
    ExtTestCase,
    hide_stdout,
    ignore_warnings,
    ignore_errors,
    requires_cuda,
)
from onnx_diagnostic.helpers.rt_helper import make_feeds
from onnx_diagnostic.reference import ExtendedReferenceEvaluator, OnnxruntimeEvaluator
from onnx_diagnostic.torch_export_patches.patch_inputs import use_dyn_not_str
from onnx_diagnostic.torch_export_patches.patches.patch_transformers import patch_qwen2_5
from onnx_diagnostic.torch_onnx.sbs import run_aligned
from onnx_diagnostic.torch_onnx.sbs_dataclasses import RunAlignedRecord, ReplayConfiguration
from onnx_diagnostic.export.api import to_onnx


class TestSideBySide(ExtTestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        cls.torch = torch

    def test_run_aligned_record(self):
        r = RunAlignedRecord(
            ep_id_node=1,
            onnx_id_node=-1,
            ep_name="A",
            onnx_name="B",
            ep_target="C",
            onnx_op_type="D",
            ep_shape_type="E",
            err_abs=0.1,
            err_rel=0.2,
            err_dev=0.3,
            err_nan=0.4,
        )
        sr = str(r)
        self.assertIn("RunAlignedRecord(", sr)
        self.assertIn("ep_shape_type='E'", sr)

    @hide_stdout()
    @unittest.skipIf(to_onnx is None, "to_onnx not installed")
    @ignore_errors(OSError)  # connectivity issues
    @ignore_warnings((UserWarning,))
    def test_ep_onnx_sync_exp(self):
        class Model(self.torch.nn.Module):
            def forward(self, x):
                ry = x.abs()
                rz = ry.exp()
                rw = rz + 1
                ru = rw.log() + rw
                return ru

        x = self.torch.randn((5, 4))
        Model()(x)
        ep = self.torch.export.export(
            Model(), (x,), dynamic_shapes=({0: self.torch.export.Dim("batch")},)
        )
        onx = to_onnx(ep, exporter="custom").model_proto
        results = list(
            run_aligned(
                ep,
                onx,
                args=(x,),
                run_cls=ExtendedReferenceEvaluator,
                atol=1e-5,
                rtol=1e-5,
                verbose=10,
            ),
        )
        self.assertEqual(len(results), 7)

    @hide_stdout()
    @ignore_warnings((DeprecationWarning, FutureWarning, UserWarning))
    def test_ep_onnx_sync_a(self):
        class Model(self.torch.nn.Module):
            def forward(self, x):
                ry = x.abs()
                rz = ry.exp()
                rw = rz + 1
                ru = rw.log() + rw
                return ru

        x = self.torch.randn((5, 4))
        Model()(x)
        ep = self.torch.export.export(
            Model(), (x,), dynamic_shapes=({0: self.torch.export.Dim("batch")},)
        )
        onx = to_onnx(
            ep,
            (x,),
            dynamic_shapes=({0: self.torch.export.Dim("batch")},),
            exporter="onnx-dynamo",
        ).model_proto
        results = list(
            run_aligned(
                ep,
                onx,
                args=(x,),
                run_cls=ExtendedReferenceEvaluator,
                atol=1e-5,
                rtol=1e-5,
                verbose=10,
            ),
        )
        self.assertEqual(len(results), 6)

    @hide_stdout()
    @ignore_warnings((DeprecationWarning, FutureWarning, UserWarning))
    def test_ep_onnx_sync_a_verbose1(self):
        class Model(self.torch.nn.Module):
            def forward(self, x):
                ry = x.abs()
                rz = ry.exp()
                rw = rz + 1
                ru = rw.log() + rw
                return ru

        x = self.torch.randn((5, 4))
        Model()(x)
        ep = self.torch.export.export(
            Model(), (x,), dynamic_shapes=({0: self.torch.export.Dim("batch")},)
        )
        onx = to_onnx(
            ep,
            (x,),
            dynamic_shapes=({0: self.torch.export.Dim("batch")},),
            exporter="onnx-dynamo",
        ).model_proto
        results = list(
            run_aligned(
                ep,
                onx,
                args=(x,),
                run_cls=ExtendedReferenceEvaluator,
                atol=1e-5,
                rtol=1e-5,
                verbose=1,
            ),
        )
        self.assertEqual(len(results), 6)

    @hide_stdout()
    @ignore_warnings((DeprecationWarning, FutureWarning, UserWarning))
    def test_sbs_dict(self):
        class Model(self.torch.nn.Module):
            def forward(self, x):
                ry = x.abs()
                rz = ry.exp()
                rw = rz + 1
                ru = rw.log() + rw
                return ru

        inputs = dict(x=self.torch.randn((5, 4)))
        ds = dict(x={0: "batch"})
        Model()(**inputs)
        ep = self.torch.export.export(
            Model(), (), kwargs=inputs, dynamic_shapes=use_dyn_not_str(ds)
        )
        epo = to_onnx(Model(), (), kwargs=inputs, dynamic_shapes=ds, exporter="onnx-dynamo")
        onx = epo.model_proto
        results = list(
            run_aligned(
                ep,
                onx,
                kwargs=inputs,
                run_cls=ExtendedReferenceEvaluator,
                atol=1e-5,
                rtol=1e-5,
                verbose=10,
            ),
        )
        self.assertEqual(len(results), 6)

    @hide_stdout()
    @ignore_warnings((DeprecationWarning, FutureWarning, UserWarning))
    def test_sbs_dict_onnxruntime(self):
        class Model(self.torch.nn.Module):
            def forward(self, x):
                ry = x.abs()
                rz = ry.exp()
                rw = rz + 1
                ru = rw.log() + rw
                return ru

        inputs = dict(x=self.torch.randn((5, 4)))
        ds = dict(x={0: "batch"})
        Model()(**inputs)
        ep = self.torch.export.export(
            Model(), (), kwargs=inputs, dynamic_shapes=use_dyn_not_str(ds)
        )
        onx = to_onnx(ep, exporter="custom").model_proto
        results = list(
            run_aligned(
                ep,
                onx,
                kwargs=inputs,
                run_cls=OnnxruntimeEvaluator,
                atol=1e-5,
                rtol=1e-5,
                verbose=11,
            ),
        )
        self.assertEqual(len(results), 7)

    @hide_stdout()
    @ignore_warnings((DeprecationWarning, FutureWarning, UserWarning))
    def test_sbs_dict_tensor(self):
        class Model(self.torch.nn.Module):
            def forward(self, x):
                ry = x.abs()
                rz = ry.exp()
                rw = rz + 1
                ru = rw.log() + rw + ry
                return ru

        inputs = dict(x=self.torch.randn((5, 4)))
        ds = dict(x={0: "batch"})
        Model()(**inputs)
        ep = self.torch.export.export(
            Model(), (), kwargs=inputs, dynamic_shapes=use_dyn_not_str(ds)
        )
        onx = to_onnx(ep, exporter="custom").model_proto
        results = list(
            run_aligned(
                ep,
                onx,
                kwargs=inputs,
                run_cls=OnnxruntimeEvaluator,
                atol=1e-5,
                rtol=1e-5,
                verbose=11,
                use_tensor=True,
            ),
        )
        self.assertEqual(len(results), 8)

    @hide_stdout()
    @ignore_warnings((DeprecationWarning, FutureWarning, UserWarning))
    @requires_cuda()
    def test_sbs_dict_tensor_cuda(self):
        class Model(self.torch.nn.Module):
            def forward(self, x):
                ry = x.abs()
                rz = ry.exp()
                rw = rz + 1
                ru = rw.log() + rw + ry
                return ru

        inputs = dict(x=self.torch.randn((5, 4)).to("cuda"))
        ds = dict(x={0: "batch"})
        Model()(**inputs)
        ep = self.torch.export.export(
            Model(), (), kwargs=inputs, dynamic_shapes=use_dyn_not_str(ds)
        )
        onx = to_onnx(ep, exporter="custom").model_proto
        results = list(
            run_aligned(
                ep,
                onx,
                kwargs=inputs,
                run_cls=OnnxruntimeEvaluator,
                atol=1e-5,
                rtol=1e-5,
                verbose=11,
                use_tensor=True,
            ),
        )
        self.assertEqual(len(results), 8)

    @hide_stdout()
    @ignore_warnings((DeprecationWarning, FutureWarning, UserWarning))
    @requires_cuda()
    def test_sbs_dict_tensor_cuda_reshape(self):
        class Model(self.torch.nn.Module):
            def forward(self, x):
                ry = x.abs()
                ry1 = ry.reshape((-1, 1))
                ry2 = ry.reshape((1, -1))
                prod = ry1 * ry2
                shape = prod.shape
                resh = prod.reshape((-1, shape[0] // 2, shape[1] // 2))
                return resh.transpose(2, 1)

        inputs = dict(x=self.torch.randn((16, 16)).to("cuda"))
        ds = dict(x={0: "batch"})
        Model()(**inputs)
        ep = self.torch.export.export(
            Model(), (), kwargs=inputs, dynamic_shapes=use_dyn_not_str(ds)
        )
        onx = to_onnx(ep, exporter="custom").model_proto
        results = list(
            run_aligned(
                ep,
                onx,
                kwargs=inputs,
                run_cls=OnnxruntimeEvaluator,
                atol=1e-5,
                rtol=1e-5,
                verbose=11,
                use_tensor=True,
            ),
        )
        self.assertEqual(len(results), 14)

    @hide_stdout()
    @ignore_warnings((DeprecationWarning, FutureWarning, UserWarning))
    def test_sbs_dict_tensor_cpu_reshape(self):
        class Model(self.torch.nn.Module):
            def forward(self, x):
                ry = x.abs()
                ry1 = ry.reshape((-1, 1))
                ry2 = ry.reshape((1, -1))
                prod = ry1 * ry2
                shape = prod.shape
                resh = prod.reshape((-1, shape[0] // 2, shape[1] // 2))
                return resh.transpose(2, 1)

        inputs = dict(x=self.torch.randn((16, 16)))
        ds = dict(x={0: "batch"})
        Model()(**inputs)
        ep = self.torch.export.export(
            Model(), (), kwargs=inputs, dynamic_shapes=use_dyn_not_str(ds)
        )
        onx = to_onnx(ep, exporter="custom").model_proto
        results = list(
            run_aligned(
                ep,
                onx,
                kwargs=inputs,
                run_cls=OnnxruntimeEvaluator,
                atol=1e-5,
                rtol=1e-5,
                verbose=11,
                use_tensor=True,
            ),
        )
        self.assertEqual(len(results), 14)
        self.assertEqual(
            [None, None, None, None, None, None, None, None, 0, 0, 0, 0, 0, 0],
            [r.err_dev for r in results],
        )

    @hide_stdout()
    @ignore_warnings((DeprecationWarning, FutureWarning, UserWarning))
    def test_sbs_model_with_weights_custom(self):
        torch = self.torch

        class Model(self.torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.fc1 = torch.nn.Linear(10, 32)  # input size 10 → hidden size 32
                self.relu = torch.nn.ReLU()
                self.fc2 = torch.nn.Linear(32, 1)  # hidden → output

            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.fc2(x)
                return x

        inputs = dict(x=self.torch.randn((5, 10)))
        ds = dict(x={0: "batch"})
        Model()(**inputs)
        ep = self.torch.export.export(
            Model(), (), kwargs=inputs, dynamic_shapes=use_dyn_not_str(ds)
        )
        filename = self.get_dump_file("test_sbs_model_with_weights_custom.onnx")
        to_onnx(ep, exporter="custom", filename=filename)
        onx = onnx.load(filename)
        results = list(
            run_aligned(
                ep,
                onx,
                kwargs=inputs,
                run_cls=OnnxruntimeEvaluator,
                verbose=11,
                use_tensor=True,
            ),
        )
        df = pandas.DataFrame(list(results)).dropna(axis=1, how="all")
        df.to_excel(self.get_dump_file("test_sbs_model_with_weights_custom.xlsx"))
        self.assertEqual(
            [
                "ep_id_node",
                "ep_name",
                "ep_shape_type",
                "ep_target",
                "ep_time_run",
                "err_abs",
                "err_dev",
                "err_h001",
                "err_h01",
                "err_rel",
                "onnx_id_node",
                "onnx_id_output",
                "onnx_name",
                "onnx_op_type",
                "onnx_shape_type",
                "onnx_time_run",
            ],
            sorted(df.columns),
        )
        self.assertEqual(len(results), 8)
        self.assertEqual([0, 0, 0, 0, None, 0, 0, 0], [r.err_dev for r in results])
        self.assertEqual(
            [-1, -1, -1, -1, -1, 0, 1, 2], df["onnx_id_node"].fillna(-10).tolist()
        )
        self.clean_dump()

    @hide_stdout()
    @ignore_warnings((DeprecationWarning, FutureWarning, UserWarning))
    def test_sbs_model_with_weights_dynamo(self):
        torch = self.torch

        class Model(self.torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.fc1 = torch.nn.Linear(10, 32)  # input size 10 → hidden size 32
                self.relu = torch.nn.ReLU()
                self.fc2 = torch.nn.Linear(32, 1)  # hidden → output

            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.fc2(x)
                return x

        inputs = dict(x=self.torch.randn((5, 10)))
        ds = dict(x={0: "batch"})
        Model()(**inputs)
        ep = self.torch.export.export(
            Model(), (), kwargs=inputs, dynamic_shapes=use_dyn_not_str(ds)
        )
        filename = self.get_dump_file("test_sbs_model_with_weights_dynamo.onnx")
        to_onnx(ep, exporter="onnx-dynamo", filename=filename)
        onx = onnx.load(filename)
        results = list(
            run_aligned(
                ep,
                onx,
                kwargs=inputs,
                run_cls=OnnxruntimeEvaluator,
                verbose=11,
                use_tensor=True,
            ),
        )
        df = pandas.DataFrame(list(results)).dropna(axis=1, how="all")
        df.to_excel(self.get_dump_file("test_sbs_model_with_weights_dynamo.xlsx"))
        self.assertEqual(
            [
                "ep_id_node",
                "ep_name",
                "ep_shape_type",
                "ep_target",
                "ep_time_run",
                "err_abs",
                "err_dev",
                "err_h001",
                "err_h01",
                "err_rel",
                "onnx_id_node",
                "onnx_id_output",
                "onnx_name",
                "onnx_op_type",
                "onnx_shape_type",
                "onnx_time_run",
            ],
            sorted(df.columns),
        )
        self.assertEqual(len(results), 8)
        self.assertEqual([0, 0, 0, 0, None, 0, 0, 0], [r.err_dev for r in results])
        self.assertEqual(
            [-1, -1, -1, -1, -1, 0, 1, 2], df["onnx_id_node"].fillna(-10).tolist()
        )
        self.clean_dump()

    @hide_stdout()
    def test_sbs_unique_consecutive(self):
        torch = self.torch

        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.unique_consecutive(x)

        model = Model()
        inputs = (torch.tensor([0, 1, 2, 2, 3, 3, 0, 0], dtype=torch.int64),)
        ds = ({0: "length"},)
        ep = torch.export.export(model, inputs, dynamic_shapes=use_dyn_not_str(ds))
        onx = to_onnx(model, inputs, dynamic_shapes=ds, exporter="custom").model_proto
        results = list(
            run_aligned(
                ep,
                onx,
                kwargs=inputs,
                run_cls=OnnxruntimeEvaluator,
                verbose=11,
                use_tensor=True,
            ),
        )
        self.assertEqual(len(results), 5)

    @hide_stdout()
    @ignore_warnings((DeprecationWarning, FutureWarning, UserWarning))
    def test_sbs_model_with_weights_custom_reset(self):
        torch = self.torch

        class Model(self.torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.fc1 = torch.nn.Linear(10, 3200)  # input size 10 → hidden size 32
                self.relu = torch.nn.ReLU()
                self.fc2 = torch.nn.Linear(3200, 1)  # hidden → output
                with torch.no_grad():
                    self.fc2.bias += 1999
                    self.fc1.bias += 999

            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.fc2(x)
                return x

        inputs = dict(x=self.torch.randn((5, 10), dtype=torch.float16))
        ds = dict(x={0: "batch"})
        model = Model()
        model = model.to(torch.float16)
        model(**inputs)
        ep = self.torch.export.export(
            model, (), kwargs=inputs, dynamic_shapes=use_dyn_not_str(ds)
        )
        filename = self.get_dump_file("test_sbs_model_with_weights_custom_reset.onnx")
        to_onnx(ep, exporter="custom", filename=filename)
        onx = onnx.load(filename)
        results = list(
            run_aligned(
                ep,
                onx,
                kwargs=inputs,
                run_cls=OnnxruntimeEvaluator,
                verbose=11,
                use_tensor=True,
                reset_names=["linear"],
            ),
        )
        df = pandas.DataFrame(list(results)).dropna(axis=1, how="all")
        df.to_excel(self.get_dump_file("test_sbs_model_with_weights_custom_reset.xlsx"))
        onnx_op_type = df["onnx_op_type"].tolist()
        self.assertEqual(onnx_op_type.count("reset"), 1)
        self.clean_dump()

    @hide_stdout()
    @ignore_warnings((DeprecationWarning, FutureWarning, UserWarning))
    def test_sbs_replay(self):
        torch = self.torch

        class Model(self.torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.fc1 = torch.nn.Linear(10, 3200)  # input size 10 → hidden size 32
                self.relu = torch.nn.ReLU()
                self.fc2 = torch.nn.Linear(3200, 1)  # hidden → output
                with torch.no_grad():
                    self.fc2.bias += 1999
                    self.fc1.bias += 999

            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.fc2(x)
                return x

        inputs = dict(x=self.torch.randn((5, 10), dtype=torch.float16))
        ds = dict(x={0: "batch"})
        model = Model()
        model = model.to(torch.float16)
        model(**inputs)
        ep = self.torch.export.export(
            model, (), kwargs=inputs, dynamic_shapes=use_dyn_not_str(ds)
        )
        filename = self.get_dump_file("test_sbs_replay.onnx")
        dump_folder = self.get_dump_folder("test_sbs_replay_linear")
        to_onnx(ep, exporter="custom", filename=filename)
        onx = onnx.load(filename)
        results = list(
            run_aligned(
                ep,
                onx,
                kwargs=inputs,
                run_cls=OnnxruntimeEvaluator,
                verbose=11,
                use_tensor=True,
                replay_configuration=ReplayConfiguration(
                    dump_folder=dump_folder, selected_op_types={"Gemm"}
                ),
            ),
        )
        df = pandas.DataFrame(list(results)).dropna(axis=1, how="all")
        df.to_excel(self.get_dump_file("test_sbs_replay.xlsx"))
        self.assertEqual(df.shape, (8, 16))
        self.clean_dump()

    @hide_stdout()
    @ignore_warnings((DeprecationWarning, FutureWarning, UserWarning))
    def test_sbs_run_onnx_with_torch_inputs(self):
        torch = self.torch

        class Model(self.torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.fc1 = torch.nn.Linear(10, 32)  # input size 10 → hidden size 32
                self.relu = torch.nn.ReLU()
                self.fc2 = torch.nn.Linear(32, 1)  # hidden → output

            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.fc2(x)
                return x

        inputs = dict(x=self.torch.randn((5, 10)))
        ds = dict(x={0: "batch"})
        Model()(**inputs)
        ep = self.torch.export.export(
            Model(), (), kwargs=inputs, dynamic_shapes=use_dyn_not_str(ds)
        )
        filename = self.get_dump_file("test_sbs_run_onnx_with_torch_inputs.onnx")
        to_onnx(ep, exporter="custom", filename=filename)
        onx = onnx.load(filename)
        results = list(
            run_aligned(
                ep,
                onx,
                kwargs=inputs,
                run_cls=OnnxruntimeEvaluator,
                verbose=11,
                use_tensor=True,
                run_onnx_with_torch_inputs=True,
            ),
        )
        df = pandas.DataFrame(list(results)).dropna(axis=1, how="all")
        df.to_excel(self.get_dump_file("test_sbs_run_onnx_with_torch_inputs.xlsx"))
        self.assertEqual(
            [
                "comment",
                "ep_id_node",
                "ep_name",
                "ep_shape_type",
                "ep_target",
                "ep_time_run",
                "err_abs",
                "err_abs2",
                "err_dev",
                "err_dev2",
                "err_h001",
                "err_h0012",
                "err_h01",
                "err_h012",
                "err_rel",
                "err_rel2",
                "onnx_id_node",
                "onnx_id_output",
                "onnx_name",
                "onnx_op_type",
                "onnx_shape_type",
                "onnx_time_run",
            ],
            sorted(df.columns),
        )
        self.assertEqual(len(results), 8)
        self.assertEqual([0, 0, 0, 0, None, 0, 0, 0], [r.err_dev for r in results])
        self.assertEqual(
            [-1, -1, -1, -1, -1, 0, 1, 2], df["onnx_id_node"].fillna(-10).tolist()
        )
        self.clean_dump()

    @unittest.skipIf(not patch_qwen2_5, "Qwen25 not part of this transformers")
    @hide_stdout()
    def test_sbs_with_loops(self):
        import torch
        from onnx_diagnostic.torch_export_patches.patches.patch_transformers import (
            PLUGS_Qwen25,
        )
        from onnx_diagnostic.torch_export_patches.patches._patch_transformers_qwen2_5 import (
            qwen_sdpa_attention_loopmha_versatile,
        )

        class Model(torch.nn.Module):
            def forward(self, query, key, value, seq_lens):
                rg1 = torch.arange(4, dtype=torch.int32).unsqueeze(0)
                rg0 = torch.arange(4, dtype=torch.int32).unsqueeze(1)
                mask = (rg0 <= rg1).flatten().reshape((1, -1, 1, 1)).to(query.dtype)
                qs = query * mask
                ks = key * mask
                vs = value * mask
                attn_output = qwen_sdpa_attention_loopmha_versatile(
                    qs,
                    ks,
                    vs,
                    seq_lens,
                    0.11,
                    16,
                    (
                        onnx.TensorProto.FLOAT
                        if query.dtype == torch.float32
                        else (
                            onnx.TensorProto.FLOAT16
                            if query.dtype == torch.float16
                            else onnx.TensorProto.BFLOAT16
                        )
                    ),
                )
                red = attn_output.mean(dim=-1, keepdim=True)
                return attn_output - red

        model = Model()
        inputs = (
            torch.rand((1, 16, 1292, 80), dtype=torch.float16),
            torch.rand((1, 16, 1292, 80), dtype=torch.float16),
            torch.rand((1, 16, 1292, 80), dtype=torch.float16),
            torch.tensor(
                [
                    0,
                    64,
                    128,
                    192,
                    256,
                    304,
                    368,
                    432,
                    496,
                    560,
                    608,
                    672,
                    736,
                    800,
                    864,
                    912,
                    976,
                    1040,
                    1104,
                    1168,
                    1216,
                    1232,
                    1248,
                    1264,
                    1280,
                    1292,
                ],
                dtype=torch.int64,
            ),
        )
        expected = model(*inputs)
        ds = ({2: "seq_length"}, {2: "seq_length"}, {2: "seq_length"}, {0: "num_patches"})
        onnx_file = self.get_dump_file("test_sbs_with_loops.onnx")
        ep_file = self.get_dump_file("test_sbs_with_loops")
        to_onnx(
            model,
            inputs,
            dynamic_shapes=ds,
            filename=onnx_file,
            save_ep=(ep_file, 2**28),
            exporter="custom",
            onnx_plugs=PLUGS_Qwen25,
            target_opset=22,
        )
        input_file = ep_file + ".input.pt"
        ep_file = ep_file + ".ep.pt2"
        self.assertExists(onnx_file)
        self.assertExists(ep_file)
        self.assertExists(input_file)
        sess = self.check_ort(onnx_file)
        input_names = [i.name for i in sess.get_inputs()]
        feeds = make_feeds(input_names, inputs, use_numpy=True)
        got = sess.run(None, feeds)
        self.assertEqualArray(expected, got[0], atol=1e-3)
        # sbs
        ep = torch.export.load(ep_file)
        onx = onnx.load(onnx_file)
        kwargs = make_feeds(input_names, inputs, use_numpy=False)
        results = list(
            run_aligned(
                ep,
                onx,
                kwargs=kwargs,
                run_cls=OnnxruntimeEvaluator,
                verbose=11,
                use_tensor=True,
            ),
        )
        df = pandas.DataFrame(list(results)).dropna(axis=1, how="all")
        df.to_excel(self.get_dump_file("test_sbs_with_loops.xlsx"))
        # self.clean_dump()


if __name__ == "__main__":
    unittest.main(verbosity=2)
