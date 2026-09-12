import unittest
import torch
from ttie.semantic_ttt import Region2
from ttie.soft_basis.renderer import CANDIDATES, HARD, FixedCorners, weights
from ttie.soft_basis.screen import summarize, ratio


class SoftBasisTests(unittest.TestCase):
    def test_exact_family_and_pixel_center_equation(self):
        self.assertEqual(len(set(CANDIDATES)), 27)
        self.assertEqual(CANDIDATES[HARD], (.5, .5, 0.))
        like = torch.zeros(1, dtype=torch.float64)
        w = weights((7, 9), (.4, .6, .05), like)
        hx = torch.sigmoid(torch.tensor(((2+.5)/9-.4)/.05))
        hy = torch.sigmoid(torch.tensor(((3+.5)/7-.6)/.05))
        expected = torch.tensor([(1-hx)*(1-hy), hx*(1-hy), (1-hx)*hy, hx*hy])
        self.assertTrue(torch.allclose(w[:, 3, 2], expected.double(), atol=1e-7))
        self.assertTrue(torch.allclose(w.sum(0), torch.ones(7, 9, dtype=like.dtype)))

    def test_hard_integer_boundary_on_odd_image(self):
        w = weights((7, 9), (.4, .6, 0.), torch.zeros(1))
        self.assertEqual(w[0].sum().item(), int(.4*9)*int(.6*7))
        self.assertEqual(w[3, 4, 3].item(), 1.)
        self.assertEqual(w[:, 3, 2].tolist(), [1., 0., 0., 0.])

    def test_identity_exact_for_every_candidate(self):
        torch.manual_seed(16)
        image = torch.rand(1, 3, 17, 19)
        image[:, :, 0] = 0
        image[:, :, -1] = 1
        corners = torch.zeros(1, 2, 2, 2)
        corners[:, 1] = 1
        for candidate in CANDIDATES:
            self.assertTrue(torch.equal(FixedCorners(corners, candidate)(image), image))

    def test_hard_nests_original_region_and_has_no_trainable_actions(self):
        torch.manual_seed(16)
        image = torch.rand(1, 3, 17, 19)
        original = Region2(torch.tensor([True, False, True, True]))
        with torch.no_grad():
            original.raw.copy_(torch.randn_like(original.raw)*.12)
            original.raw[:, :, 0, 1] = 0
        grid = original.physical_grid()[:, :2].detach()
        expected = original(image).detach()
        transferred = FixedCorners(grid, CANDIDATES[HARD])(image)
        self.assertLessEqual((expected-transferred).abs().max().item(), 1e-6)
        reference = torch.rand_like(image)
        self.assertLessEqual(abs(float((expected-reference).square().mean())-
                                 float((transferred-reference).square().mean())), 1e-10)
        for candidate in CANDIDATES:
            model = FixedCorners(grid, candidate)
            before = model.corners.clone()
            model(image)
            self.assertEqual(list(model.parameters()), [])
            self.assertTrue(torch.equal(model.corners, grid))
            self.assertTrue(torch.equal(model.corners, before))

    def test_literal_interpretations_and_global_fixed_selection(self):
        def rows(values):
            return [dict(condition=c, region2_mse=1., t015_oracle_mse=.95,
                         candidate_mse=value) for c in ('left_right', 'quadrants', 'offset_left_right_40')
                    for value in values]
        strong = summarize(rows([[.8, 1.]+[1.]*25, [1., .8]+[1.]*25]))
        self.assertEqual(strong['verdict'], 'strong_adaptive_basis_evidence')
        self.assertEqual(strong['best_fixed_index'], 0)
        fixed = summarize(rows([[.8]+[1.]*26]))
        self.assertEqual(fixed['verdict'], 'fixed_continuous_basis_evidence')
        negative = summarize(rows([[1.]*27]))
        self.assertEqual(negative['verdict'], 'negative_or_inconclusive')
        varied = [dict(condition=c, region2_mse=1., t015_oracle_mse=.9,
                      candidate_mse=v) for c, v in zip(('left_right', 'quadrants', 'offset_left_right_40'),
                      ([.5, .9]+[1.]*25, [.9, .5]+[1.]*25, [.5, .9]+[1.]*25))]
        r = summarize(varied)
        self.assertEqual(r['best_fixed_index'], 0)
        self.assertEqual(r['groups']['quadrants']['best_fixed_soft_mse'], .9)

    def test_zero_denominators_are_null_and_counted(self):
        self.assertIsNone(ratio(0., 0.))
        rows = [dict(condition=c, region2_mse=0., t015_oracle_mse=0., candidate_mse=[0.]*27)
                for c in ('left_right', 'quadrants', 'offset_left_right_40')]
        report = summarize(rows)
        self.assertTrue(all(v is None for v in report['groups']['spatial_pool']['ratios'].values()))
        self.assertEqual(report['groups']['spatial_pool']['relative_gain']['zero_denominators'], 3)


if __name__ == '__main__':
    unittest.main()
