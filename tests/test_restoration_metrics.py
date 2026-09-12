import copy
import json
from pathlib import Path
import unittest
import torch
from ttie.natural import CONDITIONS
from ttie.semantic_ttt import METHODS,run_all
from ttie.restoration_metrics import evaluate_outputs,summarize_restoration
from test_semantic_ttt import scorer,RECEIPT


class RestorationMetricTests(unittest.TestCase):
    def test_replaced_reference_changes_metrics_only_after_methods_complete(self):
        image=torch.full((1,3,16,16),.15)
        results,gate=run_all(image,scorer(),RECEIPT,max_steps=1)
        outputs={k:r['image'].clone() for k,r in results.items()};diags=copy.deepcopy({k:r['diagnostics'] for k,r in results.items()})
        a=evaluate_outputs(results,torch.full_like(image,.4),image_id=1,condition='left_right')
        b=evaluate_outputs(results,torch.full_like(image,.8),image_id=1,condition='quadrants')
        self.assertNotEqual(a[0]['mse'],b[0]['mse'])
        for k,r in results.items():
            self.assertTrue(torch.equal(outputs[k],r['image']))
            self.assertEqual(diags[k],r['diagnostics'])

    def test_literal_mean_gates_paired_rows_and_gradient_exclusion(self):
        rows=[]
        for i in (1,2):
            for c in CONDITIONS:
                for m in METHODS:
                    value=.0001 if c=='clean' else .06 if m=='spatial2_ttt' else .061 if m=='spatial2_discrete' else .08 if m=='spatial2_direct' else .1
                    rows.append(dict(image_id=i,condition=c,method=m,mse=value,psnr_db=20.,recovery_ratio=.2,clean_drift_mse=value if c=='clean' else None,dark_region_mse=value,bright_region_mse=value,loss_before=1.,loss_after=.5,active_count=2,steps=1))
        report,paired=summarize_restoration(rows)
        self.assertTrue(report['qualifies_later_detector'])
        self.assertEqual(len(paired),2*6*21)
        for r in rows:
            if r['condition']=='smooth_gradient':r['mse']=99.
        self.assertTrue(summarize_restoration(rows)[0]['qualifies_later_detector'])
        for r in rows:
            if r['condition'] in ('left_right','quadrants') and r['method']=='spatial2_discrete':r['mse']=.03
        self.assertEqual(summarize_restoration(rows)[0]['failed_criteria'],['competitive_discrete'])

    def test_official_fresh_manifest_prefix_and_exclusions(self):
        root=Path(__file__).resolve().parents[1]/'research_log'
        prior=set()
        for t in ('T004','T005','T006','T007'):prior.update(r['image_id'] for r in json.loads((root/(t+'_manifest.json')).read_text())['images'])
        manifest=json.loads((root/'T008_manifest.json').read_text());ids=[r['image_id'] for r in manifest['images']]
        self.assertEqual(manifest['image_count'],5000)
        self.assertEqual(len(ids),40);self.assertEqual(ids,sorted(set(ids)))
        self.assertFalse(prior.intersection(ids))
        self.assertEqual(ids,[r['image_id'] for r in manifest['inspected_prefix'] if r['eligible']])


if __name__=='__main__':unittest.main()
