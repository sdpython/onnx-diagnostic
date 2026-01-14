import unittest
from onnx_diagnostic.ext_test_case import (
    ExtTestCase,
    ignore_errors,
    requires_torch,
    requires_transformers,
    hide_stdout,
)
from onnx_diagnostic.helpers.model_builder_helper import (
    download_model_builder_to_cache,
    import_model_builder,
    create_model_builder,
    save_model_builder,
    find_names_pattern,
)
from onnx_diagnostic.torch_models.hghub import get_untrained_model_with_inputs
from onnx_diagnostic.helpers.rt_helper import make_feeds


class TestModelBuilderHelper(ExtTestCase):
    # This is to limit impact on CI.
    @requires_transformers("4.52")
    @requires_torch("2.7.99")
    @ignore_errors(OSError)  # connectivity issues
    def test_download_model_builder(self):
        path = download_model_builder_to_cache()
        self.assertExists(path)
        builder = import_model_builder()
        self.assertHasAttr(builder, "create_model")

    # This is to limit impact on CI.
    @requires_transformers("4.52")
    @requires_torch("2.7.99")
    @hide_stdout()
    @ignore_errors(OSError)  # connectivity issues
    def test_model_builder_id(self):
        # clear&&python ~/.cache/onnx-diagnostic/builder.py
        # --model arnir0/Tiny-LLM -p fp16 -c dump_cache -e cpu -o dump_model
        folder = self.get_dump_folder("test_model_builder_id")
        data = get_untrained_model_with_inputs("arnir0/Tiny-LLM")
        onnx_model = create_model_builder(
            data["configuration"],
            data["model"],
            precision="fp32",
            execution_provider="cpu",
            cache_dir=folder,
            verbose=1,
        )
        self.assertGreater(onnx_model.model.graph.num_nodes(), 5)
        model_name = save_model_builder(onnx_model, folder, verbose=1)
        self.assertExists(model_name)

        import onnxruntime

        sess = onnxruntime.InferenceSession(model_name, providers=["CPUExecutionProvider"])
        del data["inputs"]["position_ids"]
        feeds = make_feeds(
            [i.name for i in sess.get_inputs()],
            data["inputs"],
            use_numpy=True,
            check_flatten=False,
        )
        expected = data["model"](**data["inputs"])

        try:
            got = sess.run(None, feeds)
        except onnxruntime.capi.onnxruntime_pybind11_state.InvalidArgument as e:
            if "batch_size must be 1 when sequence_length > 1" in str(e):
                raise unittest.SkipTest("batch_size must be 1 when sequence_length > 1")
        self.assertEqualAny(expected, got)

    def test_find_names_pattern(self):
        pats = ["past_key_values_key_0", "past_key_values_key_1"]
        self.assertEqual("past_key_values_key_%d", find_names_pattern(pats))
        self.assertEqual("past_key_values_key_%d", find_names_pattern(pats[:1]))

    @requires_transformers("4.52")
    def test_model_buildersupported_classes(self):
        import torch

        def has_final_norm(module, orig_model):
            if orig_model.__class__.__name__.startswith("Peft"):
                model = orig_model.base_model.model
            else:
                model = orig_model

            hf_norm = (
                hasattr(model, "model")
                and hasattr(model.model, "norm")
                and module == model.model.norm
            )
            hf_final_layernorm = (
                hasattr(model, "model")
                and hasattr(model.model, "final_layernorm")
                and module == model.model.final_layernorm
            )
            hf_transformer_final_layernorm = (
                hasattr(model, "transformer")
                and hasattr(model.transformer, "encoder")
                and hasattr(model.transformer.encoder, "final_layernorm")
                and module == model.transformer.encoder.final_layernorm
            )
            hf_language_model_norm = (
                hasattr(model, "model")
                and hasattr(model.model, "language_model")
                and hasattr(model.model.language_model, "norm")
                and module == model.model.language_model.norm
            )

            gguf_final_norm = hasattr(model, "final_norm") and module == model.final_norm
            hf_names = [
                hf_norm,
                hf_final_layernorm,
                hf_transformer_final_layernorm,
                hf_language_model_norm,
            ]
            gguf_names = [gguf_final_norm]
            return any(hf_names + gguf_names)

        data = get_untrained_model_with_inputs("microsoft/Phi-3.5-mini-instruct")
        model = data["model"]

        exclude_lm_head = False  # extra_options, use hidden_stats instead of logits (outputs)
        exclude_embeds = False  # extra_options, use input_embeds instead of input_ids (inputs)

        # num_hidden_layers
        num_layers = (
            model.config.num_hidden_layers
            if hasattr(model.config, "num_hidden_layers")
            else model.config.num_layers
        )

        cls = []
        layer_id = -1
        prefix_layer = None
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.ModuleList):
                prefix_layer = name
                continue
            if prefix_layer and not name.startswith(prefix_layer):
                layer_id = -1
            if isinstance(module, torch.nn.Embedding) or (
                hasattr(model, "embedding") and module == model.embedding
            ):
                if not exclude_embeds:
                    cls.append(("make_embedding", layer_id, name, module))
                    continue
            if (
                module.__class__.__name__.endswith("DecoderLayer")
                or module.__class__.__name__.endswith("GLMBlock")
            ) and layer_id < num_layers:
                layer_id += 1
                cls.append(("make_layer", layer_id, name, module))
                continue
            if layer_id == num_layers and has_final_norm(module, model):
                cls.append(("make_layernorm", layer_id, name, module))
                continue
            if isinstance(module, torch.nn.Linear) or (
                hasattr(model, "lm_head") and module == model.lm_head
            ):
                if not exclude_lm_head:
                    cls.append(("make_lm_head", layer_id, name, module))
                    continue

            cls.append(("skipped", layer_id, name, module))

        colnames = ["converter", "layer_id", "name", "class"]
        cls = [dict(zip(colnames, (*obs[:3], obs[3].__class__.__name__))) for obs in cls]
        length = {k: max(len(str(obs[k])) for obs in cls) + 1 for k in cls[0]}
        msg = []
        for obs in cls:
            cc = [f"{v}{' ' * (length[k] - len(str(v)))}" for k, v in obs.items()]
            msg.append(" ".join(cc))
        self.assertEqual(len(msg), 30)
        self.assertEqual(cls[-1]["layer_id"], -1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
