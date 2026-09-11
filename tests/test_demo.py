import unittest
from unittest.mock import patch

import torch

from ttie.demo import make_toy, recovery_metrics, run_demo


class DemoTests(unittest.TestCase):
    def test_metrics_known_value(self):
        clean = torch.zeros(1, 3, 64, 96)
        metrics = recovery_metrics(clean + 0.1, clean)
        self.assertAlmostEqual(metrics["mse"], 0.01, places=6)
        self.assertAlmostEqual(metrics["psnr_db"], 20, places=5)

    def test_full_demo_is_deterministic(self):
        first, first_tensors = run_demo()
        second, second_tensors = run_demo()
        self.assertEqual(first, second)
        for name in first_tensors:
            torch.testing.assert_close(first_tensors[name], second_tensors[name], atol=0, rtol=0)
        self.assertTrue(all(case["all_finite"] for case in first["cases"].values()))

    def test_evaluation_reference_does_not_affect_adaptation(self):
        clean, degraded = make_toy()
        report, tensors = run_demo(steps=3)
        with patch("ttie.demo.make_toy", return_value=(1 - clean, degraded)):
            changed, changed_tensors = run_demo(steps=3)
        for name in ("global", "spatial"):
            torch.testing.assert_close(tensors[name], changed_tensors[name], atol=0, rtol=0)
            self.assertEqual(report["cases"][name]["loss_trajectory"], changed["cases"][name]["loss_trajectory"])
            self.assertNotEqual(report["cases"][name]["mse"], changed["cases"][name]["mse"])


if __name__ == "__main__":
    unittest.main()
