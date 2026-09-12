import copy
import unittest
import torch
from ttie.semantic_ttt import FixedObjective
from ttie.projected_ttt import run_candidate
from ttie.stop_trajectory import capture,checkpoint,select_checkpoint,select_image,feature_vectors,SCHEMA
from test_semantic_ttt import scorer,RECEIPT
from test_projected_ttt import pixels


class StopTrajectoryTests(unittest.TestCase):
    def test_passive_capture_exact_t011_outputs_gradients_and_states(self):
        image=pixels();s=scorer()
        original=run_candidate(image,FixedObjective(s,image,RECEIPT),'region2_ttt_projected',max_steps=3)
        trajectory=capture(image,s,RECEIPT,max_steps=3)
        self.assertEqual(trajectory['diagnostics'],original['diagnostics'])
        self.assertTrue(torch.equal(trajectory['images'][-1],original['image']))
        self.assertTrue(torch.equal(trajectory['states'],torch.stack(original['states'])))
        one=run_candidate(image,FixedObjective(s,image,RECEIPT),'region2_ttt_projected_1step')
        self.assertTrue(torch.equal(one['image'],trajectory['images'][1]))
        self.assertTrue(torch.equal(checkpoint(trajectory,40)['image'],original['image']))

    def test_feature_order_normalization_signs_terminal_and_projection_count(self):
        t=capture(pixels(),scorer(),RECEIPT,max_steps=12);f=t['features'];d=t['diagnostics']
        self.assertEqual(f.shape,(13,33));self.assertEqual(len(SCHEMA['names']),33)
        self.assertEqual(f[0,:4].tolist(),[1.,1.,0.,1.]);self.assertEqual(f[0,4:8].tolist(),[1.,-1.,0.,1.])
        z=(t['scores'].double()-torch.tensor(RECEIPT['calibration']['tau'],dtype=torch.float64))/torch.tensor(RECEIPT['calibration']['scale'],dtype=torch.float64)
        self.assertTrue(torch.equal(f[:,12:16],z[:,:,0].float()));self.assertTrue(torch.equal(f[:,16:20],z[:,:,1].float()))
        self.assertEqual(f[0,28:].tolist(),[0.,float(f[0,29]),0.,float(f[0,31]),0.])
        self.assertEqual(float(f[-1,31]),0.);self.assertAlmostEqual(float(f[-1,28]),12/40)
        hits=sum(p['pre_raw']!=p['post_raw'] for p in d['projections'])
        self.assertAlmostEqual(float(f[-1,32]),hits/12,places=6)
        self.assertTrue(torch.allclose(f[1:,30],f[1:,29]-f[:-1,29],atol=1e-6))

    def test_earliest_tie_and_inactive_bypass(self):
        t=capture(pixels(),scorer(),RECEIPT,max_steps=3)
        decision=select_checkpoint(t,lambda f:torch.zeros(len(f)))
        self.assertEqual(decision['selected_step'],0)
        clean=torch.full_like(pixels(),.5)
        def forbidden(_):raise AssertionError('inactive must bypass learned scoring')
        result,t,d=select_image(clean,scorer(),RECEIPT,forbidden)
        self.assertTrue(torch.equal(result['image'],clean));self.assertEqual(d['bypass'],'no_active')
        self.assertEqual(t['features'].shape,(1,33))

    def test_metadata_cannot_enter_capture_features_selection_or_output(self):
        image=pixels();head=lambda f:f[:,29]
        a,ta,da=select_image(image,scorer(),RECEIPT,head,max_steps=3)
        # Unrelated reference/metadata replacements never reach the selector API.
        for value in (torch.zeros_like(image),torch.ones_like(image)):
            _=value.square().mean()
            b,tb,db=select_image(image,scorer(),RECEIPT,head,max_steps=3)
            self.assertTrue(torch.equal(a['image'],b['image']));self.assertEqual(da,db)
            self.assertTrue(torch.equal(ta['features'],tb['features']));self.assertTrue(torch.equal(ta['states'],tb['states']))
        for key in ('clean','source_labels','condition','mask','gain','annotations','image_id','split'):
            with self.assertRaises(TypeError):select_image(image,scorer(),RECEIPT,head,**{key:None})
