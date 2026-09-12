import unittest
import torch
from ttie.residual_metrics import PRIMARY,STRESS,PAIRS,candidate_name,stage_a,stage_b
from ttie.residual_ttt import METHODS,run_all,Rhos
from ttie.natural import degrade
from ttie.restoration_metrics import evaluate_outputs
from test_semantic_ttt import scorer,RECEIPT


def row(condition,method,value):
    return dict(condition=condition,method=method,mse=value,psnr_db=10.,clean_drift_mse=value if condition=='clean' else None,
                dark_region_mse=value,bright_region_mse=value,loss_before=1.,loss_after=.2,steps=3,recovery_ratio=.5)


class ResidualMetricsTests(unittest.TestCase):
    def test_literal_grid_feasibility_and_conservative_tie(self):
        self.assertEqual(len(PAIRS),16);rows=[]
        for condition in PRIMARY:
            rows.extend([row(condition,'identity',0. if condition=='clean' else .1),row(condition,'region2_direct',.05)])
            for d,b in PAIRS:rows.append(row(condition,candidate_name(d,b),.001 if condition=='clean' else .04))
        report=stage_a(rows);self.assertEqual(report['feasible_count'],16)
        self.assertEqual((report['selected']['rho_dark'],report['selected']['rho_bright']),(.9,.9))
        # A slightly smaller MSE within tolerance loses to the conservative pair.
        for r in rows:
            if r['method']==candidate_name(.25,.25) and r['condition'] in ('left_right','quadrants'):r['mse']-=5e-7
        self.assertEqual(stage_a(rows)['selected']['rho_dark'],.9)
        for r in rows:
            if r['method'].startswith('rho') and r['condition']=='clean':r['mse']=.006
        report=stage_a(rows);self.assertFalse(report['passes']);self.assertIsNone(report['selected'])

    def test_qualification_conjunction_and_stress_exclusion(self):
        rows=[]
        for condition in (*PRIMARY,STRESS):
            for method in METHODS:
                value=.001 if condition=='clean' else .04 if method=='region2_ttt_rho' else .1
                rows.append(row(condition,method,value))
        self.assertTrue(stage_b(rows)['qualified'])
        for r in rows:
            if r['condition']==STRESS and r['method']=='region2_ttt_rho':r['mse']=100.
        self.assertTrue(stage_b(rows)['qualified'])
        for r in rows:
            if r['condition']=='quadrants' and r['method']=='region2_discrete_rho':r['mse']=.00001
            if r['condition']=='left_right' and r['method']=='region2_discrete_rho':r['mse']=.00001
        self.assertFalse(stage_b(rows)['criteria']['beyond_discrete'])

    def test_offset_boundary_and_offline_regional_mse(self):
        clean=torch.full((1,3,10,13),.4);pixels=degrade(clean,STRESS)
        self.assertTrue(torch.allclose(pixels[:,:,:,:5],torch.full((1,3,10,5),.18)))
        self.assertTrue(torch.allclose(pixels[:,:,:,5:],torch.full((1,3,10,8),.62)))
        results,_=run_all(pixels,scorer(),RECEIPT,Rhos(.5,.5),max_steps=1)
        a=evaluate_outputs(results,clean,image_id=1,condition=STRESS)
        b=evaluate_outputs(results,torch.zeros_like(clean),image_id=1,condition='quadrants')
        self.assertNotEqual(a[0]['mse'],b[0]['mse'])
        self.assertAlmostEqual(a[0]['dark_region_mse'],.22**2,places=6)
        self.assertAlmostEqual(a[0]['bright_region_mse'],.22**2,places=6)
