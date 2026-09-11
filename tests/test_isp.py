import unittest

import torch

from ttie.isp import ISP, physical_parameters


class ISPTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.image = torch.rand(1, 3, 16, 24)

    def test_identity_including_black_and_white(self):
        self.image[..., 0, 0] = 0
        self.image[..., -1, -1] = 1
        for mode in ("global", "spatial"):
            torch.testing.assert_close(ISP(mode)(self.image), self.image, atol=2e-7, rtol=0)

    def test_each_operator_has_finite_nonzero_gradient(self):
        model = ISP()
        image = torch.full((1, 3, 8, 8), 0.3)
        model(image).sum().backward()
        self.assertTrue(torch.isfinite(model.raw.grad).all())
        self.assertTrue((model.raw.grad.abs() > 0).all())
        # Black/white pixels must also permit finite backward evaluation.
        model.zero_grad()
        model(torch.cat((torch.zeros_like(image), torch.ones_like(image)), dim=-1)).sum().backward()
        self.assertTrue(torch.isfinite(model.raw.grad).all())

    def test_single_operator_semantics(self):
        image = torch.full((1, 3, 2, 2), 0.3)
        for index in range(6):
            model = ISP()
            with torch.no_grad():
                model.raw[0, index] = 0.2
            value = physical_parameters(model.raw)[0, index, 0, 0].item()
            expected = image.clone()
            if index == 0:
                expected *= 2 ** value
            elif index == 1:
                expected = (expected + 1e-6) ** value - 1e-6 ** value
            elif index < 5:
                expected[:, index - 2] *= value
            else:
                expected = 0.5 + value * (expected - 0.5)
            torch.testing.assert_close(model(image), expected)

    def test_spatial_interpolation_and_bounds(self):
        model = ISP("spatial", (2, 2))
        with torch.no_grad():
            model.raw[..., 0] = -0.5
            model.raw[..., 1] = 0.5
        grid = physical_parameters(model.raw)
        field = model.parameter_field((3, 3))
        self.assertEqual(field.shape, (1, 6, 3, 3))
        torch.testing.assert_close(field[..., 0, 0], grid[..., 0, 0])
        torch.testing.assert_close(field[..., 1, 1], grid.mean(dim=(-2, -1)))
        self.assertTrue((field[:, :1].abs() <= 2).all())
        self.assertTrue(((field[:, 1:] >= 0.5) & (field[:, 1:] <= 2)).all())

    def test_constant_spatial_grid_matches_global(self):
        global_model, spatial_model = ISP(), ISP("spatial")
        with torch.no_grad():
            global_model.raw.uniform_(-0.2, 0.2)
            spatial_model.raw.copy_(global_model.raw.expand_as(spatial_model.raw))
        torch.testing.assert_close(global_model(self.image), spatial_model(self.image))


if __name__ == "__main__":
    unittest.main()
