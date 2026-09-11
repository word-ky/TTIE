import unittest

import torch
from torch import nn

from ttie.adapt import adapt, local_statistics_loss


class AdaptTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.image = torch.full((1, 3, 16, 16), 0.2)

    def test_only_fast_parameters_updated(self):
        # A callable can include a fixed module without accumulating its gradients.
        module = nn.Conv2d(3, 3, 1)
        before = {key: value.clone() for key, value in module.state_dict().items()}
        source = self.image.clone().requires_grad_()
        result = adapt(source, lambda image: module(image).square().mean(), steps=3)
        self.assertTrue((result.raw_parameters != 0).any())
        self.assertEqual(result.diagnostics["parameter_count"], 6)
        self.assertIsNone(source.grad)
        torch.testing.assert_close(source.detach(), self.image)
        for key, value in module.state_dict().items():
            torch.testing.assert_close(value, before[key])
        self.assertTrue(all(parameter.grad is None for parameter in module.parameters()))

    def test_episode_reset_reproducibility(self):
        for mode in ("global", "spatial"):
            first = adapt(self.image, local_statistics_loss, mode=mode, steps=5)
            adapt(1 - self.image, local_statistics_loss, mode=mode, steps=5)
            repeat = adapt(self.image, local_statistics_loss, mode=mode, steps=5)
            torch.testing.assert_close(first.image, repeat.image, rtol=0, atol=0)
            torch.testing.assert_close(first.raw_parameters, repeat.raw_parameters, rtol=0, atol=0)
            self.assertEqual(first.diagnostics, repeat.diagnostics)

    def test_cpu_diagnostics_and_loss_decrease(self):
        result = adapt(self.image, local_statistics_loss, mode="spatial", steps=10)
        self.assertEqual(result.image.device.type, "cpu")
        self.assertEqual(result.parameter_grid.shape, (1, 6, 4, 4))
        self.assertEqual(result.parameter_field.shape, (1, 6, 16, 16))
        self.assertEqual(len(result.diagnostics["loss_trajectory"]), 11)
        self.assertEqual(len(result.diagnostics["gradient_norms"]), 10)
        self.assertTrue(result.diagnostics["all_finite"])
        self.assertLess(result.diagnostics["loss_trajectory"][-1], result.diagnostics["loss_trajectory"][0])

    def test_zero_steps_identity(self):
        result = adapt(self.image, local_statistics_loss, steps=0)
        torch.testing.assert_close(result.image, self.image)

    def test_fixed_prior_value(self):
        self.assertAlmostEqual(local_statistics_loss(self.image).item(), 0.09, places=6)


if __name__ == "__main__":
    unittest.main()
