import unittest

import torch

from ttie.natural import clip_pixels, degrade, views
from ttie.suite import gain_map


class NaturalTests(unittest.TestCase):
    def test_gain_conventions_match_previous_suite(self):
        source = torch.full((1, 3, 64, 96), .4)
        for name in ('clean', 'homogeneous_dark', 'homogeneous_bright', 'left_right', 'quadrants', 'smooth_gradient'):
            torch.testing.assert_close(degrade(source, name), (source*gain_map(name)).clamp(0, 1))

    def test_five_views_and_differentiable_geometry(self):
        image = torch.arange(3*33*47, dtype=torch.float32).reshape(1, 3, 33, 47).div(5000).requires_grad_()
        crops = views(image)
        self.assertEqual([v.shape[-2:] for v in crops], [(33,47), (16,23), (16,24), (17,23), (17,24)])
        batch = clip_pixels(image)
        self.assertEqual(batch.shape, (5,3,224,224))
        grad, = torch.autograd.grad(batch.square().mean(), image)
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(grad.norm().item(), 0)


if __name__ == '__main__':
    unittest.main()
