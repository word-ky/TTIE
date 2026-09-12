import tempfile
from pathlib import Path
import unittest
import torch
from ttie.energy_model import EnergyHead
from ttie.energy_io import save_episode
from ttie.sobolev_io import run_label_free,evaluate_episode,PRIMARY_METHOD,CONTROL,ORACLE
from ttie.sobolev_metrics import stage_a,stage_b,trajectory_diagnostics,summarize_trajectories
from ttie.stop_receipt import sha
from test_semantic_ttt import scorer,RECEIPT
from test_projected_ttt import pixels


class SobolevEvidenceTests(unittest.TestCase):
    def test_reference_replacement_preserves_all_persisted_label_free_evidence(self):
        torch.manual_seed(7);heads={k:EnergyHead().eval().requires_grad_(False) for k in ('value_only_control','sobolev_primary')}
        image=pixels();results,old,ts,decisions=run_label_free(image,scorer(),RECEIPT,heads,max_steps=3)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'episode';save_episode(root,results,old,ts,decisions)
            files={p:sha(p) for p in root.rglob('*') if p.is_file()}
            a,align_a,_=evaluate_episode(root,results,ts,torch.full_like(image,.4),image,image_id=1,condition='left_right',diagnose=True)
            b,align_b,_=evaluate_episode(root,results,ts,torch.full_like(image,.7),image,image_id=999,condition='quadrants',diagnose=True)
            self.assertNotEqual(a[0]['mse'],b[0]['mse']);self.assertEqual(len(align_a),2)
            self.assertNotEqual(align_a[0]['reference_gradient'],align_b[0]['reference_gradient'])
            for p,digest in files.items():self.assertEqual(sha(p),digest)
            again,_,us,e=run_label_free(image,scorer(),RECEIPT,heads,max_steps=3)
            self.assertEqual(decisions,e)
            for m in ts:
                for k in ('images','scores','features','states','grids'):self.assertTrue(torch.equal(ts[m][k],us[m][k]))
            saved=torch.load(root/'outputs.pt',weights_only=True)
            for m in results:self.assertTrue(torch.equal(saved[m]['image'],results[m]['image']))
            # Each selected image owns exactly one image's storage, not all 4 checkpoints.
            for r in saved.values():self.assertEqual(r['image'].untyped_storage().nbytes(),r['image'].numel()*r['image'].element_size())
            detail=trajectory_diagnostics(ts,decisions)
            summary=summarize_trajectories([dict(condition='left_right',trajectory_diagnostics=detail)])
            self.assertEqual(summary[PRIMARY_METHOD]['all']['updates'],3)

    def test_eight_and_twelve_literal_clauses_include_control(self):
        methods=['identity','region2_direct','region2_discrete_projected','region2_ttt_projected','fixed_step_source',CONTROL,
                 'global_ttt_energy_sobolev','bilinear2_ttt_energy_sobolev',PRIMARY_METHOD,ORACLE]
        rows=[]
        for condition in ('clean','homogeneous_dark','homogeneous_bright','left_right','quadrants'):
            for m in methods:
                mse=.001 if m==PRIMARY_METHOD else .003
                rows.append(dict(condition=condition,method=m,mse=mse,psnr_db=30.,clean_drift_mse=mse,dark_region_mse=mse,
                    bright_region_mse=mse,loss_before=1.,loss_after=.1,steps=40,recovery_ratio=.5))
        align=[dict(method=m,condition='left_right',cosine=c) for m,c in ((CONTROL,.2),(PRIMARY_METHOD,.8))]
        a=stage_a(rows,align);b=stage_b(rows)
        self.assertEqual(len(a['criteria']),8);self.assertEqual(len(b['criteria']),12)
        self.assertTrue(a['passes']);self.assertTrue(b['qualified'])
        self.assertAlmostEqual(a['derivative_changes']['median'],.6)
        for r in rows:
            if r['method']==CONTROL and r['condition'] in ('left_right','quadrants'):r['mse']=.001
        a=stage_a(rows,align);self.assertEqual(a['failed'],['beyond_value_only'])
        b=stage_b(rows);self.assertEqual(b['failed'],['beyond_value_only'])


if __name__=='__main__':unittest.main()
