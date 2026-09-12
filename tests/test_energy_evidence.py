import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from ttie.energy_model import EnergyHead
from ttie.energy_io import run_label_free,save_episode,evaluate_episode,save_bank,bank_targets
from ttie.energy_bank import state_bank
from ttie.energy_metrics import stage_a,stage_b_energy,alignment,PRIMARY_METHOD,ORACLE,PRIMARY,STRESS
from ttie.restoration_metrics import evaluate_outputs
from ttie.stop_receipt import sha
from test_projected_ttt import pixels
from test_semantic_ttt import scorer,RECEIPT
from test_residual_metrics import row


class EnergyEvidenceTests(unittest.TestCase):
    def test_saved_gradients_features_selection_survive_reference_replacement(self):
        image=pixels();head=EnergyHead().eval().requires_grad_(False)
        results,old,ts,ds=run_label_free(image,scorer(),RECEIPT,head,max_steps=2)
        frozen=copy.deepcopy(ts);original=copy.deepcopy(ds)
        with tempfile.TemporaryDirectory() as temp:
            d=Path(temp)/'episode';receipts=save_episode(d,results,old,ts,ds)
            def measure(r,clean,**kw):
                for group,files in receipts.items():
                    folder=d if group=='episode' else d/group
                    for name,record in files.items():self.assertEqual(sha(folder/name),record['sha256'])
                return evaluate_outputs(r,clean,**kw)
            with patch('ttie.energy_io.evaluate_outputs',side_effect=measure):
                a,ga,_=evaluate_episode(d,results,ts,torch.full_like(image,.45),image,image_id=1,condition='left_right',diagnose=True)
                b,gb,_=evaluate_episode(d,results,ts,torch.zeros_like(image),image,image_id=999,condition='quadrants',diagnose=True)
            self.assertNotEqual(a[0]['mse'],b[0]['mse']);self.assertNotEqual(ga['reference_gradient'],gb['reference_gradient'])
            self.assertEqual(ga['energy_gradient'],gb['energy_gradient']);self.assertEqual(ds,original)
            for m in ts:
                self.assertEqual(ts[m]['diagnostics'],frozen[m]['diagnostics'])
                for key in ('images','states','features','scores'):self.assertTrue(torch.equal(ts[m][key],frozen[m][key]))
            # Replacing all evaluation metadata never changes a new label-free run.
            r2,_,t2,d2=run_label_free(image,scorer(),RECEIPT,head,max_steps=2)
            self.assertEqual(ds,d2);self.assertTrue(torch.equal(results[PRIMARY_METHOD]['image'],r2[PRIMARY_METHOD]['image']))

    def test_source_bank_saved_before_targets_and_cosine_known_cases(self):
        image=pixels();bank=state_bank(image,scorer(),RECEIPT,semantic_steps=1)
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp)/'bank';receipt=save_bank(d,bank);targets=bank_targets(d,bank,torch.full_like(image,.45))
            self.assertEqual(len(targets),24)
            for name,r in receipt.items():self.assertEqual(sha(d/name),r['sha256'])
        _,_,ts,_=run_label_free(image,scorer(),RECEIPT,EnergyHead(),max_steps=1);t=ts[PRIMARY_METHOD]
        a=alignment(image,torch.full_like(image,.45),t)
        t['diagnostics']['gradient_vectors'][0]=torch.tensor(a['reference_gradient']).reshape(1,2,2,2).tolist()
        self.assertAlmostEqual(alignment(image,torch.full_like(image,.45),t)['cosine'],1.,places=7)
        t['diagnostics']['gradient_vectors'][0]=torch.zeros(1,2,2,2).tolist()
        self.assertEqual(alignment(image,torch.full_like(image,.45),t)['cosine'],0.)

    def test_seven_source_eleven_fresh_gates_and_oracle_not_alternate_pass(self):
        methods=('identity','region2_direct','region2_discrete_projected','region2_ttt_projected','fixed_step_source',
                 'global_ttt_energy','bilinear2_ttt_energy',PRIMARY_METHOD,ORACLE)
        rows=[row(c,m,.001 if c=='clean' else .04 if m in (PRIMARY_METHOD,ORACLE) else .1) for c in (*PRIMARY,STRESS) for m in methods]
        aligned=[dict(cosine=.8)]*4+[dict(cosine=0.)]
        a=stage_a(rows,aligned);b=stage_b_energy(rows)
        self.assertTrue(a['passes']);self.assertEqual(len(a['criteria']),7);self.assertTrue(b['qualified']);self.assertEqual(len(b['criteria']),11)
        self.assertFalse(stage_a(rows,[dict(cosine=.1)])['passes'])
        for r in rows:
            if r['condition']==STRESS and r['method']==PRIMARY_METHOD:r['mse']=100
        self.assertTrue(stage_b_energy(rows)['qualified'])
        for r in rows:
            if r['condition'] in ('left_right','quadrants') and r['method']==PRIMARY_METHOD:r['mse']=.099
        failed=stage_a(rows,aligned);self.assertFalse(failed['passes']);self.assertTrue(failed['oracle']['oracle_beats_discrete_5pct'])
