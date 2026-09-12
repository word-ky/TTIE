import inspect
import json
from pathlib import Path
import tempfile
import unittest
import torch
from ttie.energy_model import EnergyHead
from ttie.sobolev_io import run_label_free as t014
from ttie.routing.core import BASES,PRIMARY,ORACLE,route
from ttie.routing.io import run_label_free,persist_then_evaluate,evaluate_episode
from ttie.stop_receipt import sha
from test_semantic_ttt import scorer,RECEIPT
from test_projected_ttt import pixels


class RoutingCoreTests(unittest.TestCase):
    def test_only_three_scores_and_literal_tie_order(self):
        self.assertEqual(list(inspect.signature(route).parameters),['global_score','bilinear_score','region_score'])
        for scores,index,margin in [((1.,2.,3.),0,1.),((3.,1.,2.),1,1.),((3.,2.,1.),2,1.),
            ((1.,1.,1.),0,0.),((2.,1.,1.),1,0.),((1.,2.,1.),0,0.)]:
            d=route(*scores);self.assertEqual(d['selected_basis'],BASES[index]);self.assertEqual(d['winner_runner_up_margin'],margin)
        with self.assertRaises(TypeError):route(1.,2.,3.,clean=torch.zeros(1))

    def test_original_basis_regression_and_reference_access_order(self):
        torch.manual_seed(7);head=EnergyHead().eval().requires_grad_(False);image=pixels()
        before,old_before,ts_before,ds_before=t014(image,scorer(),RECEIPT,
            dict(sobolev_primary=head,value_only_control=EnergyHead().eval().requires_grad_(False)),max_steps=3)
        results,old,ts,ds=run_label_free(image,scorer(),RECEIPT,head,max_steps=3)
        for m in BASES:
            self.assertEqual(ds[m],ds_before[m])
            for k in ('images','scores','features','states','grids'):self.assertTrue(torch.equal(ts[m][k],ts_before[m][k]))
            for k in ('image','raw','grid'):self.assertTrue(torch.equal(results[m][k],before[m][k]))
        for k in old_before:self.assertEqual(type(old[k]),type(old_before[k]))
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)/'episode'
            def reference():
                receipt=json.loads((directory/'label_free_receipt.json').read_text())
                for group,files in receipt.items():
                    parent=directory if group=='episode' else directory/group
                    for f,r in files.items():self.assertEqual(sha(parent/f),r['sha256'])
                self.assertFalse((directory/'metrics.json').exists())
                self.assertFalse((directory/'oracle_diagnostic.json').exists())
                return torch.full_like(image,.4)
            files,a,diagnostic,_=persist_then_evaluate(directory,results,old,ts,ds,reference,image_id=1,condition='left_right')
            snapshots={p:sha(p) for p in directory.rglob('*') if p.is_file() and p.name not in ('metrics.json','oracle_diagnostic.json')}
            b,_,_=evaluate_episode(directory,results,torch.full_like(image,.7),image_id=999,condition='offset_left_right_40')
            self.assertNotEqual(a[0]['mse'],b[0]['mse'])
            for p,h in snapshots.items():self.assertEqual(sha(p),h)
            self.assertTrue(diagnostic['reference_only']);self.assertNotIn(ORACLE,results)
            self.assertEqual(diagnostic['oracle_mse'],min(diagnostic['basis_mse'].values()))
            saved=torch.load(directory/'outputs.pt',weights_only=True)
            self.assertTrue(torch.equal(saved[PRIMARY]['image'],saved[ds['routing']['selected_basis']]['image']))
            repeated,_,u,e=run_label_free(image,scorer(),RECEIPT,head,max_steps=3)
            self.assertEqual(ds,e)
            for m in BASES:
                for k in ('images','scores','features','states','grids'):self.assertTrue(torch.equal(ts[m][k],u[m][k]))


if __name__=='__main__':unittest.main()
