import copy
from dataclasses import FrozenInstanceError
import inspect
import unittest
import torch
from ttie.semantic_ttt import FixedObjective,EVGamma,Region2,run_method
from ttie.residual_ttt import Rhos,RHO_GRID,ResidualObjective,run_candidate,run_all,METHODS
from ttie.offline_geometry import Piecewise2
from test_semantic_ttt import scorer,RECEIPT


class ResidualTests(unittest.TestCase):
    def test_zero_rho_loss_gradient_and_trajectory_match(self):
        image=torch.full((1,3,17,19),.15);image[:,:,:,10:]=.85
        old=FixedObjective(scorer(),image,RECEIPT);new=ResidualObjective(scorer(),image,RECEIPT,Rhos(0.,0.))
        for renderer in ('bilinear2','region2'):
            a=run_method(image,old,'spatial2_ttt',max_steps=4,record_states=True,renderer=renderer)
            b=run_method(image,new,'spatial2_ttt',max_steps=4,record_states=True,renderer=renderer)
            self.assertTrue(torch.equal(a['image'],b['image']))
            self.assertEqual(a['diagnostics'],b['diagnostics'])
            self.assertTrue(all(torch.equal(x,y) for x,y in zip(a['states'],b['states'])))

    def test_equation_frozen_winner_initial_energy_opposite_penalty(self):
        image=torch.full((1,3,16,16),.1);image[:,:,:8,8:]=.9
        objective=ResidualObjective(scorer(),image,RECEIPT,Rhos(.5,.75))
        winner=objective.winner.clone();e0=objective.e0.clone();targets=objective.targets.clone()
        z=torch.tensor([[1.,2.],[3.,4.],[5.,6.],[7.,8.]],dtype=torch.float64)
        scores=z*torch.tensor(RECEIPT['calibration']['scale'])+torch.tensor(RECEIPT['calibration']['tau'])
        # Construct in exactly the calibration dtype used by the objective.
        scores=z*z.new_tensor(RECEIPT['calibration']['scale'])+z.new_tensor(RECEIPT['calibration']['tau'])
        expected=[]
        for i,w in enumerate(winner.tolist()):
            if objective.active[i]:expected.append(max(float(z[i,w]**2-targets[i]),0)+float(z[i,1-w]**2))
        self.assertAlmostEqual(float(objective.from_scores(scores)),sum(expected)/len(expected),places=12)
        objective(torch.full_like(image,.99))
        self.assertTrue(torch.equal(winner,objective.winner));self.assertTrue(torch.equal(e0,objective.e0))
        self.assertTrue(torch.equal(targets,objective.targets));self.assertFalse(objective.e0.requires_grad)
        with self.assertRaises(FrozenInstanceError):objective.rhos.dark=.25

    def test_coordinate_only_region_matches_diagnostic_and_constant_bilinear(self):
        image=torch.rand(1,3,17,19);region=Region2();old=Piecewise2();bilinear=EVGamma(2)
        grid=torch.tensor([.5,1.1]).reshape(1,2,1,1).expand(1,2,2,2).clone()
        for m in (region,old,bilinear):m.set_grid(grid)
        self.assertTrue(torch.equal(region(image),old(image)))
        self.assertTrue(torch.allclose(region(image),bilinear(image),atol=2e-7))
        grid[0,0]=torch.tensor([[.5,-.5],[-1.,1.]])
        region.set_grid(grid);old.set_grid(grid)
        self.assertTrue(torch.equal(region(image),old(image)))
        self.assertEqual(list(inspect.signature(region.parameter_field).parameters),['size'])
        with self.assertRaises(TypeError):region(image,mask=torch.ones_like(image))

    def test_all_methods_reset_and_reject_reference_metadata(self):
        pixels=torch.full((1,3,12,12),.2);rho=Rhos(.5,.75)
        a,ga=run_all(pixels,scorer(),RECEIPT,rho,max_steps=3)
        b,gb=run_all(pixels,scorer(),RECEIPT,rho,max_steps=3)
        self.assertEqual(ga,gb);self.assertEqual(tuple(a),METHODS)
        for k in a:
            self.assertTrue(torch.equal(a[k]['image'],b[k]['image']));self.assertEqual(a[k]['diagnostics'],b[k]['diagnostics'])
        for key in ('clean','reference','condition','mask','gain','grouping'):
            with self.assertRaises(TypeError):run_all(pixels,scorer(),RECEIPT,rho,**{key:'unused'})
        self.assertEqual(rho,Rhos(.5,.75));self.assertEqual(RHO_GRID,(.25,.5,.75,.9))

    def test_no_active_exact_identity_all_methods(self):
        image=torch.full((1,3,12,12),.5);results,gate=run_all(image,scorer(),RECEIPT,Rhos(.9,.9),max_steps=2)
        self.assertFalse(any(gate['active']))
        for result in results.values():
            self.assertTrue(torch.equal(image,result['image']));self.assertEqual(result['diagnostics']['steps'],0)
