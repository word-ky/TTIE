import unittest
import torch
from ttie.semantic_ttt import FixedObjective
from ttie.energy_ttt import make_model,evaluate_energy,trajectory
from ttie.energy_model import EnergyHead,train_energy
from ttie.sobolev_source import source_bank,source_derivatives,derivative_record
from ttie.sobolev_train import train_pair,raw_gradient,cosine,training_statistics
from test_semantic_ttt import scorer,RECEIPT
from test_projected_ttt import pixels


class SobolevCoreTests(unittest.TestCase):
    def test_cached_jacobian_matches_direct_chain_rule_and_reference(self):
        image=pixels();clean=image*.8+.1;obj=FixedObjective(scorer(),image,RECEIPT)
        model=make_model(obj,image,'region2')
        with torch.no_grad():model.raw.copy_(torch.linspace(-.08,.1,8).reshape_as(model.raw))
        record=derivative_record(image,clean,obj,model.raw)
        torch.manual_seed(17);head=EnergyHead().eval().requires_grad_(False)
        with torch.no_grad():
            head.x_scale.copy_(torch.linspace(.5,2.,28));head.y_scale.fill_(2.3)
        value,out,_,_,f=evaluate_energy(model,image,obj,head)
        direct=torch.autograd.grad(value,model.raw,retain_graph=True)[0].flatten()
        reference=torch.autograd.grad(((out-clean).square().mean()+1e-6).log(),model.raw)[0].flatten()
        x=record['features'][None].requires_grad_()
        cached=raw_gradient(head,head.standardized(x),x,record['jacobian'][None])[0]
        torch.testing.assert_close(cached,direct,atol=2e-7,rtol=2e-6)
        torch.testing.assert_close(record['reference_gradient'],reference,atol=0,rtol=0)
        self.assertTrue(torch.equal(record['jacobian'][:12],torch.zeros(12,8)))
        self.assertEqual(tuple(record['jacobian'].shape),(28,8))

    def test_bank_coordinates_unchanged_exact_inactive_and_value_only_rows(self):
        image=pixels();s=scorer();bank=source_bank(image,s,RECEIPT,semantic_steps=2)
        self.assertEqual(len(bank['states']),24)
        self.assertTrue(torch.equal(bank['images'][...,8:,:9],image[...,8:,:9].expand(24,-1,-1,-1,-1)))
        records=source_derivatives(image,torch.full_like(image,.5),s,RECEIPT,bank)
        self.assertEqual(records['jacobian'].shape,(24,28,8));self.assertTrue(records['direction_mask'].all())
        image=torch.full_like(image,.5);bank=source_bank(image,s,RECEIPT)
        record=source_derivatives(image,image,s,RECEIPT,bank)
        self.assertEqual(record['active'].tolist(),[False]);self.assertEqual(record['direction_mask'].tolist(),[False])
        self.assertEqual(float(record['jacobian'].norm()),0.)

    def test_same_source_value_control_is_bitwise_t013_and_direction_loss(self):
        torch.manual_seed(13);x=torch.randn(16,28);mse=torch.linspace(.002,.07,16)
        records=dict(jacobian=torch.randn(16,28,8),reference_gradient=torch.randn(16,8),direction_mask=torch.ones(16,dtype=torch.bool))
        expected,history=train_energy(x,mse);heads,histories=train_pair(x,mse,records)
        self.assertEqual(history,histories['value_only_control'])
        for k,v in expected.state_dict().items():self.assertTrue(torch.equal(v,heads['value_only_control'].state_dict()[k]))
        self.assertEqual(heads['sobolev_primary'].normalization(),expected.normalization())
        self.assertEqual(len(histories['sobolev_primary']),100)
        stats=training_statistics(heads['sobolev_primary'],x,mse,records)
        self.assertEqual(stats['direction_rows'],16);self.assertTrue(torch.isfinite(torch.tensor(stats['cosines'])).all())
        v=cosine(torch.tensor([[1.,0.],[-1.,0.],[0.,1.],[0.,0.]]),torch.tensor([[1.,0.]]).expand(4,-1))
        torch.testing.assert_close((1-v)/2,torch.tensor([0.,1.,.5,.5]))
        records['direction_mask'].zero_();inactive,_=train_pair(x,mse,records)
        for k,v in expected.state_dict().items():self.assertTrue(torch.equal(v,inactive['sobolev_primary'].state_dict()[k]))

    def test_inference_rejects_all_source_supervision(self):
        image=pixels();head=EnergyHead().eval().requires_grad_(False)
        for key in ('clean','labels','condition','mask','gain','annotations','image_id','jacobian','reference_gradient','source_labels'):
            with self.assertRaises(TypeError):trajectory(image,scorer(),RECEIPT,head,**{key:None})

    def test_cached_features_use_same_grad_forward_path_as_derivatives(self):
        class ModeSensitiveScorer(torch.nn.Module):
            def __init__(self):super().__init__();self.base=scorer()
            def forward(self,image):
                return self.base(image)+(1e-3 if torch.is_grad_enabled() and image.requires_grad else 0.)
        image=pixels();s=ModeSensitiveScorer();bank=source_bank(image,s,RECEIPT,semantic_steps=1)
        records=source_derivatives(image,torch.full_like(image,.5),s,RECEIPT,bank)
        self.assertTrue(torch.equal(records['features'],bank['features']))


if __name__=='__main__':unittest.main()
