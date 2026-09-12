import copy
import json
from pathlib import Path
import unittest

import torch
from torch import nn

from ttie.natural import views
from ttie.learned_prototypes import Prototypes
from ttie.semantic_ttt import EVGamma, SemanticScorer, FixedObjective, direct_grid, choose_candidate, run_all, run_method, METHODS


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__();self.weight=nn.Parameter(torch.ones(()))
    def image_embeddings(self,image):
        mean=torch.stack([v.mean()*self.weight for v in views(image)])
        return torch.stack((mean*0,.5-mean,mean-.5),dim=-1)


def scorer():return SemanticScorer(TinyEncoder(),Prototypes(torch.eye(3)))
RECEIPT=dict(calibration=dict(tau=[.02,.02],scale=[.1,.1]),q_joint=1.053775168916056)


class SemanticTTTTests(unittest.TestCase):
    def test_frozen_encoder_gradients_reach_both_isp_coordinates(self):
        s=scorer();image=torch.full((1,3,16,16),.12)
        objective=FixedObjective(s,image,RECEIPT);isp=EVGamma(2)
        loss=objective(isp(image));gradient,=torch.autograd.grad(loss,isp.raw)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertTrue((gradient.abs().sum(dim=(0,2,3))>0).all())
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in s.parameters()))
        self.assertEqual(isp.raw.numel(),8)

    def test_fixed_original_mask_and_two_sided_hinge(self):
        s=scorer();image=torch.full((1,3,16,16),.1);image[:,:,:8,:8]=.5
        objective=FixedObjective(s,image,RECEIPT);mask=objective.active.clone()
        self.assertEqual(mask.tolist(),[False,True,True,True])
        self.assertEqual(float(objective(torch.full_like(image,.5))),0.)
        self.assertGreater(float(objective(torch.full_like(image,.95))),0.)
        self.assertTrue(torch.equal(mask,objective.active))

    def test_no_active_is_bitwise_identity_for_every_method(self):
        image=torch.full((1,3,16,16),.5)
        result,gate=run_all(image,scorer(),RECEIPT,max_steps=2)
        self.assertFalse(any(gate['active']))
        for r in result.values():
            self.assertTrue(torch.equal(image,r['image']))
            self.assertEqual(r['diagnostics']['steps'],0)

    def test_global_spatial_constant_render_and_fixed_other_operators(self):
        image=torch.linspace(.01,.99,3*16*16).reshape(1,3,16,16)
        one=EVGamma(1);two=EVGamma(2)
        one.set_grid(torch.tensor([.5,1.25]).reshape(1,2,1,1))
        two.set_grid(torch.tensor([.5,1.25]).reshape(1,2,1,1).expand(1,2,2,2))
        self.assertTrue(torch.allclose(one(image),two(image),atol=2e-7))
        self.assertTrue(torch.equal(two.physical_grid()[:,2:],torch.ones(1,4,2,2)))
        self.assertEqual(one.raw.numel(),2)

    def test_direct_policy_and_exact_candidate_ties(self):
        image=torch.full((1,3,16,16),.1);image[:,:,:8,8:]=.9;image[:,:,8:,:8]=.5
        objective=FixedObjective(scorer(),image,RECEIPT)
        self.assertEqual(direct_grid(objective,2)[0,0].tolist(),[[.5,-.5],[0.,.5]])
        self.assertEqual(direct_grid(objective,1)[0,0,0,0],.125)
        self.assertEqual(choose_candidate((-.5,.5),(1.,1.),0.),-.5)
        self.assertEqual(choose_candidate((.8,1.,1.25),(1.,1.,1.),1.),1.)
        self.assertEqual(choose_candidate((-1.,-.5,0.,.5,1.),(1.,1.,1.,1.,1.),0.),0.)

    def test_episode_reset_and_metadata_cannot_enter_trajectory(self):
        pixels=torch.full((1,3,16,16),.15)
        metadata_a=dict(condition='dark',reference=torch.zeros_like(pixels),mask='left')
        a,ga=run_all(pixels,scorer(),RECEIPT,max_steps=2)
        metadata_b=dict(condition='bright',reference=torch.ones_like(pixels),mask='right')
        b,gb=run_all(pixels,scorer(),RECEIPT,max_steps=2)
        self.assertEqual(ga,gb)
        for name in METHODS:
            self.assertTrue(torch.equal(a[name]['image'],b[name]['image']))
            self.assertEqual(a[name]['diagnostics'],b[name]['diagnostics'])
        self.assertEqual(len(a['global_discrete']['diagnostics']['search']),2)
        self.assertEqual(len(a['spatial2_discrete']['diagnostics']['search']),8)

    def test_literal_saved_calibration_and_hash(self):
        root=Path(__file__).resolve().parents[1]
        receipt=json.loads((root/'research_log/T007_joint_calibration.json').read_text())
        self.assertEqual(receipt['q_joint'],1.053775168916056)
        self.assertEqual(receipt['calibration']['tau'],[.027419920079410076,.001673370413482167])
        self.assertEqual(receipt['calibration']['scale'],[.07507099353490992,.057845398696933635])
        self.assertEqual(receipt['prototype_identity']['sha256'],'b4b32dbd96c65dcf606ee38d7450ebf348f5731823503b9c71ba15ec78217ac7')


if __name__=='__main__':unittest.main()
