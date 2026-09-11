import unittest

import torch

from ttie.adapt import adapt, local_statistics_loss
from ttie.isp import ISP, physical_parameters


class UniformTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        self.image = torch.rand(1, 3, 16, 24) * 0.4

    def test_count_aggregation_and_uniform_field(self):
        model = ISP("uniform_control")
        self.assertEqual(model.raw.numel(), 96)
        with torch.no_grad():
            model.raw.normal_(std=0.2)
        expected = physical_parameters(model.raw.mean((-2, -1), keepdim=True))
        field = model.parameter_field((16, 24))
        torch.testing.assert_close(field, expected.expand_as(field), rtol=0, atol=2e-7)
        model(self.image).sum().backward()
        self.assertTrue(torch.isfinite(model.raw.grad).all())
        self.assertTrue((model.raw.grad.abs() > 0).all())
        torch.testing.assert_close(model.raw.grad, model.raw.grad[..., :1, :1].expand_as(model.raw.grad))

    def test_symmetric_control_stays_near_global_and_resets(self):
        global_result = adapt(self.image, local_statistics_loss, steps=20)
        uniform = adapt(self.image, local_statistics_loss, mode="uniform_control", steps=20)
        repeat = adapt(self.image, local_statistics_loss, mode="uniform_control", steps=20)
        self.assertEqual(uniform.diagnostics["parameter_count"], 96)
        self.assertEqual(uniform.parameter_grid.shape, (1, 6, 1, 1))
        torch.testing.assert_close(uniform.raw_parameters, uniform.raw_parameters[..., :1, :1].expand_as(uniform.raw_parameters))
        # Adam epsilon is not invariant to the 1/16 raw-gradient scaling.
        torch.testing.assert_close(global_result.image, uniform.image, atol=1e-5, rtol=0)
        torch.testing.assert_close(uniform.image, repeat.image, atol=0, rtol=0)

    def test_spatial_one_matches_global_exactly(self):
        global_result = adapt(self.image, local_statistics_loss, steps=20)
        spatial = adapt(self.image, local_statistics_loss, mode="spatial", grid_size=(1, 1), steps=20)
        torch.testing.assert_close(global_result.image, spatial.image, rtol=0, atol=0)
        self.assertEqual(global_result.diagnostics, spatial.diagnostics)


if __name__ == "__main__":
    unittest.main()
