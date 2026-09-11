import copy
import unittest

import torch
from torch import nn

from ttie.clip_signal import FrozenCLIP, calibrate, decisions, gated_objective
from ttie.clip_audit import auc, image_scores, score_rows, summarize


class TinyEncoder(nn.Module):
    """Test gradient ownership only; never used as experimental CLIP evidence."""
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.eye(3))

    def encode_image(self, pixels):
        return pixels.mean(dim=(-2,-1)) @ self.weight


class ClipSignalTests(unittest.TestCase):
    def scorer(self):
        return FrozenCLIP(TinyEncoder(), torch.eye(3), size=16)

    def test_frozen_encoder_has_pixel_gradient(self):
        scorer = self.scorer()
        image = torch.rand(1,3,32,40, requires_grad=True)
        gradient, = torch.autograd.grad(scorer(image).sum(), image)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.norm().item(), 0)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in scorer.parameters()))
        self.assertFalse(scorer.model.training)

    def test_calibration_ignores_heldout_values_and_dirty_calibration(self):
        manifest = {'images':[dict(image_id=1,split='calibration'),dict(image_id=2,split='evaluation')]}
        rows = [dict(image_id=1,condition='clean',d_dark=float(i),d_bright=float(2*i)) for i in range(5)]
        baseline = calibrate(rows, manifest)
        self.assertAlmostEqual(baseline['tau'][0], 3.8)
        self.assertAlmostEqual(baseline['scale'][0], 2**.5)
        rows += [dict(image_id=2,condition='clean',d_dark=1e9,d_bright=-1e9),
                 dict(image_id=1,condition='homogeneous_dark',d_dark=1e9,d_bright=1e9)]
        self.assertEqual(calibrate(rows, manifest), baseline)

    def test_detached_winner_threshold_and_zero_loss(self):
        scores = torch.tensor([[.3,.2],[.1,.4],[.1,.1]], requires_grad=True)
        cal = dict(tau=[.3,.3], scale=[.1,.1])
        winner, active = decisions(scores, cal)
        self.assertEqual(winner.tolist(), [0,1,0])
        self.assertEqual(active.tolist(), [False,True,False])
        image = torch.rand(1,3,32,40)
        loss, _, active = gated_objective(self.scorer(), image, dict(tau=[10.,10.],scale=[.1,.1]))
        output = image.clone().requires_grad_()
        objective = loss(output)
        gradient, = torch.autograd.grad(objective, output)
        self.assertEqual(objective.item(), 0)
        self.assertEqual(gradient.count_nonzero().item(), 0)
        self.assertFalse(active.any())

    def test_scores_do_not_consume_evaluation_reference_or_labels(self):
        scorer = self.scorer()
        image = torch.rand(1,3,32,40)
        scores = image_scores(scorer, image)
        first = score_rows(scores, image, dict(image_id=1,split='evaluation'), 'clean')
        second = score_rows(image_scores(scorer,image), image, dict(image_id=999,split='calibration'), 'quadrants')
        self.assertEqual([(r['d_dark'],r['d_bright']) for r in first], [(r['d_dark'],r['d_bright']) for r in second])
        torch.testing.assert_close(scores, image_scores(scorer, image), rtol=0,atol=0)

    def test_auc_ties_and_literal_conjunction(self):
        self.assertEqual(auc([1,2],[0,1]), .875)
        rows = []
        for i in range(4):
            for view in ('full','top_left','top_right','bottom_left','bottom_right'):
                for cond,d,b,active,kind in [('clean',0,0,False,'dark'),('homogeneous_dark',1,0,True,'dark'),('homogeneous_bright',0,1,True,'bright')]:
                    rows.append(dict(image_id=i,view=view,condition=cond,split='evaluation',d_dark=d,d_bright=b,active=active,type=kind))
        self.assertTrue(summarize(rows)['stage_a_pass'])
        changed = copy.deepcopy(rows)
        for row in changed:
            if row['condition']=='clean': row['active'] = True
        result = summarize(changed)
        self.assertFalse(result['stage_a_pass'])
        self.assertEqual(result['failed_criteria'], ['clean_fpr'])


if __name__ == '__main__':
    unittest.main()
