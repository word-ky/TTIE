import unittest
from unittest.mock import patch

import torch

from ttie.safety_sweep import SETTINGS, safety_case
from ttie.suite import clean_family, degraded_image
from ttie.safety_summary import diagnostic_points, UTILITY_CONDITIONS


class SafetySweepTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_all_fixed_settings_and_component_trajectories(self):
        rows, _, diagnostics = safety_case(family="midtone", condition="left_right", seed=7, steps=3)
        self.assertEqual(len(SETTINGS), 11)
        self.assertEqual(len(rows), 13)
        self.assertTrue(all(r["all_finite"] for r in rows))
        for name, diag in diagnostics.items():
            length = 1 if name == "identity" else 4
            self.assertEqual(len(diag["loss_trajectory"]), length)
            for trajectory in diag["component_trajectories"].values():
                self.assertEqual(len(trajectory), length)
        for row in rows[2:]:
            self.assertEqual(row["parameter_count"], 96)
            self.assertEqual(row["anchor_before"], 0)
            self.assertEqual(row["smooth_before"], 0)

    def test_clean_reference_replacement_never_changes_any_trajectory(self):
        clean = clean_family("dark_structures", 7)
        degraded = degraded_image(clean, "left_right")
        first, first_pack, first_diag = safety_case(family="dark_structures", condition="left_right", seed=7, steps=3)
        with patch("ttie.safety_sweep.clean_family", return_value=1 - clean), patch("ttie.safety_sweep.degraded_image", return_value=degraded):
            changed, changed_pack, changed_diag = safety_case(family="dark_structures", condition="left_right", seed=7, steps=3)
        self.assertEqual(first_diag, changed_diag)
        for a, b in zip(first, changed):
            self.assertNotEqual(a["mse"], b["mse"])
            torch.testing.assert_close(first_pack[a["model"]], changed_pack[b["model"]], rtol=0, atol=0)

    def test_predeclared_decision_rule_known_values(self):
        rows = []
        for name, a, s in [("global", 0., 0.)] + list(SETTINGS):
            for family in ("midtone", "dark_structures", "high_key"):
                rows.append(dict(seed=7, family=family, condition="clean", model=name, mse=.1, lambda_a=a, lambda_s=s,
                                 identity_drift_mse=.01 if name in ("a1_s0", "a10_s0") else .1))
            for condition in UTILITY_CONDITIONS:
                mse = .3 if name == "global" else .14 if name == "a1_s0" else .2 if name == "a10_s0" else .1
                rows.append(dict(seed=7, family="midtone", condition=condition, model=name, mse=mse, lambda_a=a, lambda_s=s))
        summary = diagnostic_points(rows)
        self.assertEqual(summary["qualifying_settings"], ["a1_s0"])
        self.assertAlmostEqual(summary["baseline_heterogeneous_improvement"], .2)
        strong = next(p for p in summary["settings"] if p["model"] == "a10_s0")
        self.assertAlmostEqual(strong["drift_reduction_factor"], 10)
        self.assertAlmostEqual(strong["utility_retention"], .5)


if __name__ == "__main__":
    unittest.main()
