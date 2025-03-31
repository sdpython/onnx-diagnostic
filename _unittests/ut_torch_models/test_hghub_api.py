import unittest
import pandas
from onnx_diagnostic.ext_test_case import (
    ExtTestCase,
    hide_stdout,
    long_test,
    never_test,
    requires_torch,
    requires_transformers,
)
from onnx_diagnostic.torch_models.hghub.hub_api import (
    enumerate_model_list,
    get_model_info,
    get_pretrained_config,
    task_from_id,
    task_from_arch,
    task_from_tags,
)
from onnx_diagnostic.torch_models.hghub.hub_data import (
    load_architecture_task,
    load_models_testing,
)


class TestHuggingFaceHubApi(ExtTestCase):

    @requires_transformers("4.50")  # we limit to some versions of the CI
    @requires_torch("2.7")
    @hide_stdout()
    def test_enumerate_model_list(self):
        models = list(
            enumerate_model_list(
                2,
                verbose=1,
                dump="test_enumerate_model_list.csv",
                filter="image-classification",
                library="transformers",
            )
        )
        self.assertEqual(len(models), 2)
        df = pandas.read_csv("test_enumerate_model_list.csv")
        self.assertEqual(df.shape, (2, 12))
        tasks = [task_from_id(c, "missing") for c in df.id]
        self.assertEqual(len(tasks), 2)

    @requires_transformers("4.50")
    @requires_torch("2.7")
    def test_task_from_id(self):
        for name, etask in [
            ("arnir0/Tiny-LLM", "text-generation"),
            ("microsoft/phi-2", "text-generation"),
        ]:
            with self.subTest(name=name, task=etask):
                task = task_from_id(name, True)
                self.assertEqual(etask, task)

    @requires_transformers("4.50")
    @requires_torch("2.7")
    @never_test()
    def test_task_from_id_long(self):
        for name, etask in [
            ("microsoft/Phi-3.5-mini-instruct", "text-generation"),
            ("microsoft/Phi-3.5-vision-instruct", "text-generation"),
        ]:
            with self.subTest(name=name, task=etask):
                task = task_from_id(name, True)
                self.assertEqual(etask, task)

    @requires_transformers("4.50")
    @requires_torch("2.7")
    @hide_stdout()
    def test_get_pretrained_config(self):
        conf = get_pretrained_config("microsoft/phi-2")
        self.assertNotEmpty(conf)
        print(conf)

    @requires_transformers("4.50")
    @requires_torch("2.7")
    @hide_stdout()
    def test_get_model_info(self):
        info = get_model_info("microsoft/phi-2")
        self.assertEqual(info.pipeline_tag, "text-generation")

        info = get_model_info("microsoft/Phi-3.5-vision-instruct")
        self.assertEqual(info.pipeline_tag, "image-text-to-text")

        info = get_model_info("microsoft/Phi-4-multimodal-instruct")
        self.assertEqual(info.pipeline_tag, "automatic-speech-recognition")

    def test_task_from_arch(self):
        task = task_from_arch("LlamaForCausalLM")
        self.assertEqual("text-generation", task)

    @never_test()
    def test_hf_all_models(self):
        list(enumerate_model_list(-1, verbose=1, dump="test_hf_all_models.csv"))

    def test_load_architecture_task(self):
        data = load_architecture_task()
        print(set(data.values()))

    def test_task_from_tags(self):
        _tags = [
            ("text-generation|nlp|code|en|text-generation-inference", "text-generation"),
            (
                "text-generation|nlp|code|vision|image-text-to-text|conversational",
                "image-text-to-text",
            ),
            (
                "text-generation|nlp|code|audio|automatic-speech-recognition|speech-summarization|speech-translation|visual-question-answering",
                "automatic-speech-recognition",
            ),
        ]
        for tags, etask in _tags:
            with self.subTest(tags=tags, task=etask):
                task = task_from_tags(tags)
                self.assertEqual(etask, task)

    def test_model_testings(self):
        models = load_models_testing()
        self.assertNotEmpty(models)

    @long_test()
    def test_model_testings_and_architectures(self):
        models = load_models_testing()
        for mid in models:
            task = task_from_id(mid)
            self.assertNotEmpty(task)

    def test__ccached_config_64(self):
        from onnx_diagnostic.torch_models.hghub.hub_data_cached_configs import (
            _cached_hf_internal_testing_tiny_random_beitforimageclassification,
        )

        conf = _cached_hf_internal_testing_tiny_random_beitforimageclassification()
        self.assertEqual(conf.auxiliary_channels, 256)


if __name__ == "__main__":
    unittest.main(verbosity=2)
