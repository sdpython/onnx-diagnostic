import unittest
import torch
from onnx_diagnostic.ext_test_case import ExtTestCase, requires_transformers
from onnx_diagnostic.helpers import flatten_object
from onnx_diagnostic.helpers.cache_helper import make_dynamic_cache
from onnx_diagnostic.helpers.fake_tensor_helper import make_fake, FakeTensorContext


class TestMakeTensorHelper(ExtTestCase):

    @requires_transformers("4.55")
    def test_fake_inputs(self):
        inputs, _ = make_fake(
            dict(
                input_ids=torch.randint(30360, size=(2, 3), dtype=torch.int64),
                attention_mask=torch.randint(1, size=(2, 33), dtype=torch.int64),
                position_ids=torch.randint(32, size=(2, 3), dtype=torch.int64),
                past_key_values=make_dynamic_cache(
                    [
                        (
                            torch.rand((2, 32, 30, 96), dtype=torch.float16),
                            torch.rand((2, 32, 30, 96), dtype=torch.float16),
                        ),
                        (
                            torch.rand((2, 32, 30, 96), dtype=torch.float16),
                            torch.rand((2, 32, 30, 96), dtype=torch.float16),
                        ),
                    ]
                ),
            )
        )
        flat = flatten_object(inputs, drop_keys=True)
        for t in flat:
            self.assertIsInstance(t, torch.Tensor)
            assert all(
                isinstance(s, torch.SymInt) for s in t.shape
            ), f"Wrong type {[type(s) for s in t.shape]} in {t.shape}"

    def test_fake_reshape_generic(self):
        t = torch.zeros((2, 3, 4, 5), dtype=torch.float32)
        reshaped = FakeTensorContext().fake_reshape(t, {0: "batch", 2: "seq_length"})
        self.assertIsInstance(reshaped.shape[0], torch.SymInt)
        self.assertIsInstance(reshaped.shape[2], torch.SymInt)
        self.assertEqual(reshaped.shape[1], 3)
        self.assertEqual(reshaped.shape[3], 5)

    def test_fake_reshape_dim_1(self):
        t = torch.zeros((1, 3, 4, 5), dtype=torch.float32)
        reshaped = FakeTensorContext().fake_reshape(t, {0: "batch", 2: "seq_length"})
        self.assertIsInstance(reshaped.shape[0], torch.SymInt)
        self.assertIsInstance(reshaped.shape[2], torch.SymInt)
        self.assertEqual(reshaped.shape[1], 3)
        self.assertEqual(reshaped.shape[3], 5)

    def test_fake_reshape_dim_0(self):
        t = torch.zeros((0, 3, 4, 5), dtype=torch.float32)
        reshaped = FakeTensorContext().fake_reshape(t, {0: "batch", 2: "seq_length"})
        self.assertIsInstance(reshaped.shape[0], torch.SymInt)
        self.assertIsInstance(reshaped.shape[2], torch.SymInt)
        self.assertEqual(reshaped.shape[1], 3)
        self.assertEqual(reshaped.shape[3], 5)

    def test_fake_reshape_different(self):
        t = torch.zeros((2, 3, 2, 5), dtype=torch.float32)
        reshaped = FakeTensorContext().fake_reshape(t, {0: "batch", 2: "seq_length"})
        self.assertIsInstance(reshaped.shape[0], torch.SymInt)
        self.assertIsInstance(reshaped.shape[2], torch.SymInt)
        self.assertEqual(reshaped.shape[1], 3)
        self.assertEqual(reshaped.shape[3], 5)
        self.assertNotEqual(reshaped.shape[0], reshaped.shape[2])


if __name__ == "__main__":
    unittest.main(verbosity=2)
