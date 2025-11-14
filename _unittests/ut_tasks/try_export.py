import os
import unittest
import torch
from onnx_diagnostic.ext_test_case import ExtTestCase, never_test, ignore_warnings
from onnx_diagnostic.torch_export_patches import torch_export_patches
from onnx_diagnostic.torch_models.hghub.model_inputs import get_untrained_model_with_inputs
from onnx_diagnostic.export.api import to_onnx

# from onnx_diagnostic.export.shape_helper import make_fake_with_dynamic_dimensions


class TestTryExportHuggingFaceHubModel(ExtTestCase):
    @never_test()
    @ignore_warnings(UserWarning)
    def test_imagetext2text_qwen_2_5_vl_instruct_visual(self):
        """
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
        """
        device = os.environ.get("TESTDEVICE", "cpu")
        dtype = os.environ.get("TESTDTYPE", "float32")
        torch_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[dtype]
        exporter = os.environ.get("EXPORTER", "custom")

        from transformers import AutoModel, AutoProcessor

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

        model = model.to(device).to(getattr(torch, dtype))

        print(f"-- config._attn_implementation={model.config._attn_implementation}")
        print(f"-- model.dtype={model.dtype}")
        print(f"-- model.device={model.device}")
        processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
        print(f"-- processor={type(processor)}")

        inputs = dict(
            hidden_states=torch.rand((1292, 1176), dtype=torch_dtype).to(device),
            grid_thw=torch.tensor([[1, 34, 38]], dtype=torch.int64).to(device),
        )
        print("-- save inputs")
        torch.save(inputs, self.get_dump_file("qwen_2_5_vl_instruct_visual.inputs.pt"))

        print(f"-- inputs: {self.string_type(inputs, with_shape=True)}")
        # this is too long
        expected = model.visual(**inputs)
        print(f"-- expected: {self.string_type(expected, with_shape=True)}")

        filename = self.get_dump_file(
            f"test_imagetext2text_qwen_2_5_vl_instruct_visual.{device}.{dtype}.{exporter}.onnx"
        )
        fileep = self.get_dump_file(
            f"test_imagetext2text_qwen_2_5_vl_instruct_visual.{device}.{dtype}.{exporter}.graph"
        )
        dynamic_shapes = dict(
            hidden_states={0: "hidden_width", 1: "hidden_height"},
            grid_thw={},  # {0: "n_images"}, # TODO: fix
        )

        # fake_inputs = make_fake_with_dynamic_dimensions(inputs, dynamic_shapes)[0]
        export_inputs = inputs
        print()
        with torch_export_patches(
            patch_torch=False,
            patch_sympy=False,
            patch_transformers=True,
            verbose=1,
            stop_if_static=2,
        ):
            to_onnx(
                model.visual,
                kwargs=export_inputs,
                dynamic_shapes=dynamic_shapes,
                filename=filename,
                exporter=exporter,
                verbose=1,
                save_ep=(fileep, 2**35),
                target_opset=22,
                optimize=True,
            )

        self.assert_onnx_disc(
            f"test_imagetext2text_qwen_2_5_vl_instruct_visual.{device}.{dtype}.{exporter}",
            filename,
            model.visual,
            export_inputs,
            verbose=1,
            providers=(
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if device == "cuda"
                else ["CPUExecutionProvider"]
            ),
            use_ort=True,
            atol=0.02,
            rtol=10,
            ort_optimized_graph=False,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
