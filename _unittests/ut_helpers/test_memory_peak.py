import os
import time
import unittest
import numpy as np
import torch
from onnx_diagnostic.ext_test_case import (
    ExtTestCase,
    skipif_ci_apple,
    ignore_warnings,
    requires_cuda,
)
from onnx_diagnostic.helpers.memory_peak import get_memory_rss, start_spying_on, Monitor


class TestMemoryPeak(ExtTestCase):
    def test_basic(self):
        m = Monitor()
        self.assertEqual(
            repr(m),
            "Monitor(begin=0, end=0, peak=0, average=0, n=0, d_end=0, d_peak=0, d_avg=0)",
        )
        m.update(1)
        self.assertEqual(
            repr(m),
            "Monitor(begin=1, end=1, peak=1, average=1, n=1, d_end=0, d_peak=0, d_avg=0.0)",
        )

    @skipif_ci_apple("stuck")
    def test_memory(self):
        mem = get_memory_rss(os.getpid())
        self.assertIsInstance(mem, int)

    @skipif_ci_apple("stuck")
    @ignore_warnings(DeprecationWarning)
    def test_spy_cpu(self):
        p = start_spying_on(cuda=False)
        n_elements = 0
        for _i in range(10):
            time.sleep(0.005)
            value = np.empty(2**23, dtype=np.int64)
            time.sleep(0.005)
            value += 1
            time.sleep(0.005)
            n_elements = max(value.shape[0], n_elements)
        time.sleep(0.02)
        measures = p.stop()
        self.assertGreater(n_elements, 0)
        self.assertIsInstance(measures, dict)
        self.assertLessEqual(measures["cpu"].end, measures["cpu"].max_peak)
        self.assertLessEqual(measures["cpu"].begin, measures["cpu"].max_peak)
        self.assertGreater(measures["cpu"].begin, 0)
        # Zero should not happen...
        self.assertGreaterOrEqual(measures["cpu"].delta_peak, 0)
        self.assertGreaterOrEqual(measures["cpu"].delta_peak, measures["cpu"].delta_end)
        self.assertGreaterOrEqual(measures["cpu"].delta_peak, measures["cpu"].delta_avg)
        self.assertGreaterOrEqual(measures["cpu"].delta_end, 0)
        self.assertGreaterOrEqual(measures["cpu"].delta_avg, 0)
        # Too unstable.
        # self.assertGreater(measures["cpu"].delta_peak, n_elements * 8 * 0.5)
        self.assertIsInstance(measures["cpu"].to_dict(), dict)

    @skipif_ci_apple("stuck")
    @requires_cuda()
    def test_spy_cuda(self):
        p = start_spying_on(cuda=True)
        n_elements = 0
        for _i in range(10):
            time.sleep(0.005)
            value = torch.empty(2**23, dtype=torch.int64, device="cuda")
            value += 1
            n_elements = max(value.shape[0], n_elements)
        time.sleep(0.02)
        measures = p.stop()
        self.assertIsInstance(measures, dict)
        self.assertIn("gpus", measures)
        gpu = measures["gpus"][0]
        self.assertLessEqual(gpu.end, gpu.max_peak)
        self.assertLessEqual(gpu.begin, gpu.max_peak)
        self.assertGreater(gpu.delta_peak, 0)
        self.assertGreaterOrEqual(gpu.delta_peak, gpu.delta_end)
        self.assertGreaterOrEqual(gpu.delta_peak, gpu.delta_avg)
        self.assertGreater(gpu.delta_end, 0)
        self.assertGreater(gpu.delta_avg, 0)
        self.assertGreater(gpu.delta_peak, n_elements * 8 * 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
