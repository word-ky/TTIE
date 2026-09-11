import json
import tempfile
import unittest
from pathlib import Path

import torch

from ttie.demo import make_toy
from ttie.suite import (FAMILIES, CONDITIONS, adapt_variants, clean_family, degraded_image,
                        evaluate_case, gain_map, save_rows, save_panel)


class SuiteTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_clean_families_and_determinism(self):
        base, _ = make_toy(7)
        for family in FAMILIES:
            image = clean_family(family, 7)
            torch.testing.assert_close(image, clean_family(family, 7), rtol=0, atol=0)
            torch.testing.assert_close(degraded_image(image, "clean"), image, rtol=0, atol=0)
            self.assertTrue(((image >= 0) & (image <= 1)).all())
        self.assertLess(clean_family("dark_structures", 7).mean(), base.mean())
        self.assertGreater(clean_family("high_key", 7).mean(), base.mean())

    def test_degradation_masks_and_t001_equivalence(self):
        clean, degraded = make_toy(7)
        torch.testing.assert_close(degraded_image(clean, "left_right"), degraded, rtol=0, atol=0)
        for condition in CONDITIONS:
            gain = gain_map(condition)
            self.assertEqual(gain.shape, (1, 1, 64, 96))
            self.assertTrue(torch.isfinite(gain).all())
        self.assertAlmostEqual(gain_map("quadrants")[0, 0, 0, 0].item(), 0.45, places=6)
        self.assertAlmostEqual(gain_map("quadrants")[0, 0, 0, -1].item(), 1.55, places=6)
        self.assertTrue((gain_map("homogeneous_dark") == 0.45).all())
        self.assertTrue((gain_map("homogeneous_bright") == 1.55).all())
        self.assertTrue((gain_map("smooth_gradient")[..., 1:] > gain_map("smooth_gradient")[..., :-1]).all())
        for condition, transitions in (("stripes_4", 7), ("stripes_12", 23)):
            values = gain_map(condition)[0, 0, 0]
            self.assertEqual((values[1:] != values[:-1]).sum().item(), transitions)

    def test_evaluation_only_reference_and_exports(self):
        clean = clean_family("midtone", 7)
        degraded = degraded_image(clean, "left_right")
        adapted = adapt_variants(degraded, steps=2, lr=0.03)
        rows, pack = evaluate_case(clean, degraded, adapted, family="midtone", condition="left_right", seed=7)
        changed, changed_pack = evaluate_case(1 - clean, degraded, adapted, family="midtone", condition="left_right", seed=7)
        self.assertEqual(len(rows), 7)
        self.assertEqual([r["parameter_count"] for r in rows], [0, 6, 96, 6, 24, 96, 384])
        for first, second in zip(rows, changed):
            self.assertEqual(first["loss_after"], second["loss_after"])
            torch.testing.assert_close(pack[first["model"]], changed_pack[first["model"]], rtol=0, atol=0)
            self.assertNotEqual(first["mse"], second["mse"])
            self.assertTrue(first["all_finite"])
        for name in ("identity", "global", "uniform96", "spatial1"):
            row = next(r for r in rows if r["model"] == name)
            self.assertLess(max(v for k, v in row.items() if k.startswith("field_variance_")), 1e-12)
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary)
            save_rows(rows, destination)
            self.assertEqual(json.loads((destination / "metrics.json").read_text()), rows)
            save_panel([("midtone/left_right", pack)], destination / "panel.png")
            self.assertTrue((destination / "panel.png").is_file())

    def test_identity_drift_clean_condition(self):
        clean = clean_family("high_key", 7)
        adapted = adapt_variants(clean, steps=2, lr=0.03)
        rows, _ = evaluate_case(clean, clean, adapted, family="high_key", condition="clean", seed=7)
        self.assertEqual(rows[0]["mse"], 0)
        self.assertIsNone(rows[0]["psnr_db"])
        for row in rows:
            self.assertEqual(row["mse"], row["identity_drift_mse"])
        self.assertGreater(rows[1]["identity_drift_mse"], 0)


if __name__ == "__main__":
    unittest.main()
