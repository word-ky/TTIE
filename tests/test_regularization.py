import json
import unittest
from pathlib import Path

import torch

from ttie.adapt import adapt, local_statistics_loss
from ttie.demo import run_demo
from ttie.isp import physical_parameters
from ttie.regularization import correction_penalties, normalized_correction


class RegularizationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_identity_normalization_anchor_and_bounds(self):
        raw = torch.zeros(1, 6, 4, 4, requires_grad=True)
        physical = physical_parameters(raw)
        self.assertEqual(torch.count_nonzero(normalized_correction(physical)).item(), 0)
        anchor, smooth = correction_penalties(physical)
        self.assertEqual(anchor.item(), 0)
        self.assertEqual(smooth.item(), 0)
        for magnitude in (0.3, 20., -20.):
            raw = torch.full((1, 6, 4, 4), magnitude, requires_grad=True)
            normalized = normalized_correction(physical_parameters(raw))
            self.assertTrue(torch.isfinite(normalized).all())
            self.assertTrue((normalized.abs() <= 1 + 1e-6).all())
            anchor, smooth = correction_penalties(physical_parameters(raw))
            self.assertGreater(anchor.item(), 0)
            (anchor + smooth).backward()
            self.assertTrue(torch.isfinite(raw.grad).all())

    def test_tv_constant_varying_and_analytic_value(self):
        raw = torch.full((1, 6, 2, 2), 0.3)
        self.assertEqual(correction_penalties(physical_parameters(raw))[1].item(), 0)
        physical = physical_parameters(torch.zeros_like(raw))
        physical[:, 0, :, 1] = 2  # EV normalized diff=1, averaged over six coordinates.
        anchor, smooth = correction_penalties(physical)
        self.assertAlmostEqual(smooth.item(), 1 / 6, places=6)
        self.assertAlmostEqual(anchor.item(), 1 / 12, places=6)

    def test_global_unaffected_by_tv_and_components_reconstruct_total(self):
        image = torch.full((1, 3, 16, 16), 0.2)
        first = adapt(image, local_statistics_loss, steps=20)
        regularized = adapt(image, local_statistics_loss, steps=20, lambda_s=10)
        torch.testing.assert_close(first.image, regularized.image, rtol=0, atol=0)
        self.assertEqual(first.diagnostics["loss_trajectory"], regularized.diagnostics["loss_trajectory"])
        self.assertEqual(regularized.diagnostics["component_trajectories"]["smooth"], [0.] * 21)
        result = adapt(image, local_statistics_loss, mode="spatial", steps=5, lambda_a=1, lambda_s=.1)
        c = result.diagnostics["component_trajectories"]
        for prior, anchor, smooth, total in zip(c["prior"], c["anchor"], c["smooth"], result.diagnostics["loss_trajectory"]):
            self.assertAlmostEqual(prior + anchor + .1 * smooth, total, places=6)

    def test_zero_weight_t001_fixed_numeric_regression(self):
        baseline = json.loads(Path("research_log/artifacts/T001_cpu/metrics.json").read_text())
        actual, _ = run_demo()
        for mode in ("identity", "global", "spatial"):
            self.assertAlmostEqual(actual["cases"][mode]["mse"], baseline["cases"][mode]["mse"], delta=1e-9)


if __name__ == "__main__":
    unittest.main()
