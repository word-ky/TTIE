import tempfile
from pathlib import Path
import unittest
import torch
from ttie.energy_model import EnergyHead,train_energy,save_energy,load_energy,features
from ttie.energy_bank import state_bank,BANK
from ttie.energy_ttt import trajectory,make_model,evaluate_energy
from ttie.semantic_ttt import FixedObjective
from test_semantic_ttt import scorer,RECEIPT
from test_projected_ttt import pixels


class EnergyCoreTests(unittest.TestCase):
    def test_features_have_live_evidence_and_state_gradients(self):
        image=pixels();obj=FixedObjective(scorer(),image,RECEIPT);model=make_model(obj,image,'region2')
        output=model(image);s=obj.scorer(output);f=features(obj,s,model.physical_grid()[:,:2])
        self.assertEqual(f.shape,(28,));self.assertEqual(f[:8].tolist(),[1,1,0,1,1,-1,0,1])
        evidence,=torch.autograd.grad(f[12:20].square().sum(),model.raw,retain_graph=True)
        state,=torch.autograd.grad(f[20:].sum(),model.raw)
        self.assertGreater(float(evidence.norm()),0);self.assertGreater(float(state.norm()),0)
        global_model=make_model(obj,image,'global');global_f=features(obj,s,global_model.physical_grid()[:,:2])
        self.assertEqual(global_f[20:].tolist(),[0]*4+[1]*4)

    def test_bank_order_sobol_mapping_and_inactive_identity(self):
        image=pixels();bank=state_bank(image,scorer(),RECEIPT,semantic_steps=2)
        self.assertEqual(bank['names'],BANK['order']);self.assertEqual(bank['features'].shape,(24,28))
        # Sobol zero chooses the physical lower corner, not raw identity.
        self.assertTrue(torch.allclose(bank['grids'][8,0,0],torch.tensor([[0.,-.5],[0.,0.]])))
        self.assertTrue(torch.allclose(bank['grids'][8,0,1],torch.tensor([[.8,.8],[1.,.8]])))
        self.assertTrue(torch.equal(bank['states'][5],bank['states'][7]))
        inactive=state_bank(torch.full_like(image,.5),scorer(),RECEIPT)
        self.assertEqual(inactive['names'],['identity']);self.assertTrue(torch.equal(inactive['images'][0],torch.full_like(image,.5)))

    def test_exact_updates_selection_reset_projection_and_metadata_rejection(self):
        torch.manual_seed(7);head=EnergyHead().eval().requires_grad_(False);image=pixels()
        for basis in ('region2','global','bilinear2'):
            a,t,d=trajectory(image,scorer(),RECEIPT,head,basis=basis,max_steps=4)
            b,u,e=trajectory(image,scorer(),RECEIPT,head,basis=basis,max_steps=4)
            self.assertEqual(t['diagnostics']['steps'],4);self.assertEqual(len(t['images']),5);self.assertEqual(d,e)
            self.assertTrue(torch.equal(a['image'],b['image']));self.assertTrue(torch.equal(t['states'][0],torch.zeros_like(t['states'][0])))
            self.assertEqual(d['selected_step'],min(range(5),key=lambda i:(d['scores'][i],i)))
            box=t['diagnostics']['action_box'];lo=torch.tensor(box['lower']);hi=torch.tensor(box['upper'])
            self.assertTrue((t['grids']>=lo-1e-7).all() and (t['grids']<=hi+1e-7).all())
            if basis=='region2':self.assertTrue(torch.equal(t['images'][...,8:,:9],image[...,8:,:9].expand(5,-1,-1,-1,-1)))
        for key in ('clean','labels','condition','mask','gain','annotations','image_id'):
            with self.assertRaises(TypeError):trajectory(image,scorer(),RECEIPT,head,**{key:None})
        for p in head.parameters():p.data.zero_()
        _,t,d=trajectory(image,scorer(),RECEIPT,head,max_steps=2);self.assertEqual(d['selected_step'],0);self.assertEqual(t['diagnostics']['steps'],2)
        clean=torch.full_like(image,.5);out,t,d=trajectory(clean,scorer(),RECEIPT,head)
        self.assertTrue(torch.equal(out['image'],clean));self.assertEqual(t['diagnostics']['steps'],0)

    def test_fixed_training_recipe_normalization_save_and_device_gradient(self):
        torch.manual_seed(3);x=torch.randn(20,28);x[:,0]=1;y=torch.linspace(.001,.1,20)
        head,history=train_energy(x,y);self.assertEqual(len(history),100)
        self.assertTrue(torch.equal(head.x_mean,x.double().mean(0).float()));self.assertEqual(float(head.x_scale[0]),1.)
        self.assertIsInstance(head.net[1],torch.nn.SiLU)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'energy.pt';save_energy(head,path);restored=load_energy(path)
            self.assertTrue(torch.equal(head(x),restored(x)))
            if torch.cuda.is_available():
                cpu=head(x);gpu=restored.cuda()(x.cuda()).cpu()
                self.assertTrue(torch.allclose(cpu,gpu,atol=2e-5,rtol=1e-5));self.assertEqual(int(cpu.argmin()),int(gpu.argmin()))
                z=x.cuda().requires_grad_();g,=torch.autograd.grad(restored(z).sum(),z)
                self.assertTrue(torch.isfinite(g).all());self.assertGreater(float(g.norm()),0)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in head.parameters()))
