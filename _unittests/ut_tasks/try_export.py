import os
import time
import unittest
import onnx
import textwrap
import torch
from onnx_diagnostic.ext_test_case import (
    ExtTestCase,
    never_test,
    ignore_warnings,
    has_onnxruntime,
)
from onnx_diagnostic.torch_export_patches import torch_export_patches
from onnx_diagnostic.torch_models.hghub.model_inputs import get_untrained_model_with_inputs
from onnx_diagnostic.export.api import to_onnx

# from onnx_diagnostic.export.shape_helper import make_fake_with_dynamic_dimensions


class TestTryExportHuggingFaceHubModel(ExtTestCase):
    @never_test()
    @ignore_warnings(UserWarning)
    def test_qwen25_vli_visual(self):
        """
        unittest::

            UNITTEST_GOING=1 NEVERTEST=1 TESTDTYPE=float16 TESTDEVICE=cpu python \\
                _unittests/ut_tasks/try_export.py -f -k test_qwen25_vli_visual

        # task: imagetext2text
        clear&&NEVERTEST=1 python _unittests/ut_tasks/try_export.py -k qwen_2_5

        possible prefix: ``TEXTDEVICE=cuda TESTDTYPE=float16 EXPORTER=onnx-dynamo

        ::

            kwargs=dict(
                cache_position:T7s3602,
                input_ids:T7s1x3602,
                inputs_embeds:None,
                attention_mask:T7s1x3602,
                position_ids:T7s4x1x3602,
                pixel_values:T1s14308x1176,
                pixel_values_videos:None,
                image_grid_thw:T7s1x3,
                video_grid_thw:None,
                second_per_grid_ts:None,
                use_cache:bool,
                return_dict:bool
            )

        .. code-block:: bash

            NEVERTEST=1 \\
            QWEN25ATTENTION=BIGMASK \\
            PRETRAINED=1 \\
            TESTDEVICE=cuda \\
            TESTDTYPE=float16 \\
            EXPORTER=custom \\
            python _unittests/ut_tasks/try_export.py -k qwen25_vli_visual

        .. code-block:: bash

            python -m onnx_diagnostic sbs \\
                -i qwen25_vli_visual.inputs.pt \\
                -e test_qwen25_vli_visual.cuda.float16.PACKED.custom.graph.ep \\
                -m test_qwen25_vli_visual.cuda.float16.PACKED.custom.onnx \\
                -o test_qwen25_vli_visual.cuda.float16.PACKED.custom.xlsx \\
                -v 1 --atol 0.1 --rtol 1000

        """
        begin = time.perf_counter()
        device = os.environ.get("TESTDEVICE", "cpu")
        dtype = os.environ.get("TESTDTYPE", "float32")
        torch_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[dtype]
        exporter = os.environ.get("EXPORTER", "custom")

        from transformers import AutoModel, AutoProcessor
        from onnx_diagnostic.torch_export_patches.patches._patch_transformers_qwen2_5 import (
            PLUGS,
        )

        model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
        # model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
        if os.environ.get("PRETRAINED", ""):
            print("-- pretrained model")
            model = AutoModel.from_pretrained(
                model_id, device_map=device, dtype=torch_dtype, attn_implementation="sdpa"
            ).eval()
        else:
            print("-- random model")

            def _config_reduction(config, task):
                return {
                    # "num_hidden_layers": 2,
                    "text_config": {
                        "num_hidden_layers": 2,
                        "layer_types": ["full_attention", "full_attention"],
                    },
                    # "_attn_implementation": "flash_attention_2",
                    "_attn_implementation": "sdpa",
                    "dtype": "float16",
                }

            config_reduction = _config_reduction
            data = get_untrained_model_with_inputs(
                model_id, verbose=1, add_second_input=False, config_reduction=config_reduction
            )
            model = data["model"]

        print(f"-- MODEL LOADED IN {time.perf_counter() - begin}")
        begin = time.perf_counter()
        model = model.to(device).to(getattr(torch, dtype))
        print(f"-- MODEL MOVED IN {time.perf_counter() - begin}")

        print(f"-- config._attn_implementation={model.config._attn_implementation}")
        print(f"-- model.dtype={model.dtype}")
        print(f"-- model.device={model.device}")
        begin = time.perf_counter()
        processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
        print(f"-- processor={type(processor)}")
        print(f"-- PROCESSOR LOADED IN {time.perf_counter() - begin}")

        big_inputs = dict(
            hidden_states=torch.rand((14308, 1176), dtype=torch_dtype).to(device),
            grid_thw=torch.tensor([[1, 98, 146]], dtype=torch.int64).to(device),
        )
        print("-- save inputs")
        inputs = dict(
            hidden_states=torch.rand((1292, 1176), dtype=torch_dtype).to(device),
            grid_thw=torch.tensor([[1, 34, 38]], dtype=torch.int64).to(device),
        )
        if not self.unit_test_going():
            print("-- save inputs")
            torch.save(big_inputs, self.get_dump_file("qwen25_vli_visual.inputs.big.pt"))
            torch.save(inputs, self.get_dump_file("qwen25_vli_visual.inputs.pt"))

        print(f"-- inputs: {self.string_type(inputs, with_shape=True)}")
        # this is too long
        model_to_export = model.visual if hasattr(model, "visual") else model.model.visual
        begin = time.perf_counter()
        expected = model_to_export(**inputs)
        print(f"-- MODEL RUN IN {time.perf_counter() - begin}")
        print(f"-- expected: {self.string_type(expected, with_shape=True)}")

        dynamic_shapes = dict(
            hidden_states={0: "hidden_width", 1: "hidden_height"},
            grid_thw={},  # {0: "n_images"}, # TODO: fix
        )

        qwen25_attention = os.environ.get("QWEN25ATTENTION", "")
        if qwen25_attention:
            attention_options = [qwen25_attention]
        elif device == "cuda" and dtype in ("float16", "bfloat16"):
            attention_options = ["PACKED", "BIGMASK"]
        else:
            attention_options = ["LOOPMHA", "LOOPA24", "BIGMASK"]

        # fake_inputs = make_fake_with_dynamic_dimensions(inputs, dynamic_shapes)[0]
        for attention in attention_options:
            if attention == "LOOPA24" and not has_onnxruntime("1.24"):
                continue
            with self.subTest(attention=attention):
                print()
                print(f"-- attention={attention!r}")
                os.environ["QWEN25ATTENTION"] = attention
                filename = self.get_dump_file(
                    f"test_qwen25_vli_visual.{device}.{dtype}.{attention}.{exporter}.onnx"
                )
                fileep = self.get_dump_file(
                    f"test_qwen25_vli_visual.{device}.{dtype}.{attention}.{exporter}.graph"
                )

                begin = time.perf_counter()
                export_inputs = inputs
                with torch_export_patches(
                    patch_torch=False,
                    patch_sympy=False,
                    patch_transformers=True,
                    verbose=1,
                    stop_if_static=2,
                ):
                    to_onnx(
                        model_to_export,
                        kwargs=export_inputs,
                        dynamic_shapes=dynamic_shapes,
                        filename=filename,
                        exporter=exporter,
                        verbose=1,
                        save_ep=None if self.unit_test_going() else (fileep, 2**35),
                        target_opset=24 if attention == "LOOPA24" else 22,
                        optimize=True,
                        onnx_plugs=PLUGS,
                    )

                if not self.unit_test_going():
                    with open(
                        self.get_dump_file(
                            f"sbs_qwen25_vli_visual.{device}.{dtype}.{attention}.{exporter}.sh"
                        ),
                        "w",
                    ) as f:
                        f.write(
                            textwrap.dedent(
                                f"""
                                clear&&python -m onnx_diagnostic sbs \\
                                    -i qwen25_vli_visual.inputs.pt \\
                                    -e test_qwen25_vli_visual.{device}.{dtype}.{attention}.{exporter}.graph.ep.pt2 \\
                                    -m test_qwen25_vli_visual.{device}.{dtype}.{attention}.{exporter}.onnx \\
                                    -o test_qwen25_vli_visual.{device}.{dtype}.{attention}.{exporter}.xlsx \\
                                    -v 1 --atol 0.1 --rtol 1000
                                """
                            )
                        )
                print(f"-- MODEL CONVERTED IN {time.perf_counter() - begin}")
                model = onnx.load(filename, load_external_data=False)
                if attention == "PACKED":
                    self.assertIn('"PackedMultiHeadAttention"', str(model))
                elif attention == "BIGMASK":
                    self.assertNotIn('"PackedMultiHeadAttention"', str(model))
                    self.assertNotIn("MultiHeadAttention", str(model))
                    self.assertNotIn("Loop", {n.op_type for n in model.graph.node})
                elif attention == "LOOPMHA":
                    self.assertNotIn('"PackedMultiHeadAttention"', str(model))
                    self.assertIn('"MultiHeadAttention"', str(model))
                    self.assertIn("Loop", {n.op_type for n in model.graph.node})
                elif attention == "LOOPA24":
                    self.assertNotIn('"PackedMultiHeadAttention"', str(model))
                    self.assertNotIn('"MultiHeadAttention"', str(model))
                    self.assertIn("Loop", {n.op_type for n in model.graph.node})
                else:
                    raise AssertionError(f"attention={attention!r} not expected")

                pt2_files = [f"{fileep}.backup.pt2", f"{fileep}.ep.pt2", f"{fileep}.pt2"]
                pt2_files = [f for f in pt2_files if os.path.exists(f)]
                assert (
                    self.unit_test_going() or pt2_files
                ), f"Unable to find an existing file among {pt2_files!r}"

                # pt2_file = (
                #    (pt2_files[0] if pt2_files else None)
                #    if not self.unit_test_going()
                #    else None
                # )

                # self.assertExists(pt2_file)
                # ep = torch.export.load(pt2_file)
                # diff = self.max_diff(ep.module()(**export_inputs), model.visual(**export_inputs))
                # print("----------- diff", diff)
                begin = time.perf_counter()
                self.assert_onnx_disc(
                    (f"test_qwen25_vli_visual.{device}.{dtype}.{attention}.{exporter}"),
                    filename,
                    model_to_export,
                    export_inputs,
                    verbose=1,
                    providers=(
                        ["CUDAExecutionProvider", "CPUExecutionProvider"]
                        if device == "cuda"
                        else ["CPUExecutionProvider"]
                    ),
                    use_ort=True,
                    atol=0.05,
                    rtol=10,
                    # ep=pt2_file,
                    expected=expected,
                )
                print(f"-- MODEL VERIFIED IN {time.perf_counter() - begin}")
        os.environ["QWEN25ATTENTION"] = qwen25_attention
        if self.unit_test_going():
            self.clean_dump()


if __name__ == "__main__":
    unittest.main(verbosity=2)
