import copy
import math
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from ttie.boundary_confidence import THRESHOLDS,confidence,choose,calibrate_fold
from ttie.boundary_confidence_run import finish,sha,evaluate_decisions,q_summary


def fixture():
    rows=[];refs={}
    for image in range(40):
        for condition in range(3):
            episode=f'{image}/{condition}';scores=[2.]*9;scores[0]=0;scores[4]=1
            rows.append(dict(episode=episode,fold=image%5,predictions=scores))
            refs[episode]=dict(reference_mse=[1.,3.,3.,3.,2.,3.,3.,3.,3.])
    fold=dict(fold=0,train=[i for i,r in enumerate(rows) if r['fold']!=0],heldout=[i for i,r in enumerate(rows) if r['fold']==0],
        train_image_ids=[i for i in range(40) if i%5],heldout_image_ids=list(range(0,40,5)))
    return rows,refs,fold


class ConfidenceTests(unittest.TestCase):
    def test_fixed_margin_and_noncanonical_ties(self):
        scores=[0.,2.,2.,2.,1.,2.,2.,2.,2.]
        j,q=confidence(scores);mean=sum(scores)/9
        expected=1/math.sqrt(sum((s-mean)**2 for s in scores)/9)
        self.assertEqual(j,0);self.assertAlmostEqual(q,expected,places=14)
        self.assertEqual(confidence([5.]*9),(0,0.))
        scores=[0.]*9;scores[4]=1e-15
        self.assertEqual(confidence(scores),(0,.001))
        self.assertEqual(choose(0,-1,0),4)

    def test_exact_grid_strict_rule_and_safe_calibration_tie(self):
        self.assertEqual(THRESHOLDS,(0.,.25,.5,.75,1.,1.5,2.,math.inf))
        self.assertEqual(choose(0,.5,.5),4);self.assertEqual(choose(0,.50000001,.5),0)
        rows,refs,fold=fixture()
        for ref in refs.values():ref['reference_mse']=[1.]*9
        calibration,decisions=calibrate_fold(rows,fold,refs)
        self.assertEqual(calibration['threshold'],'inf')
        self.assertTrue(all(r['selected_index']==4 for r in decisions))

    def test_direct_heldout_reference_guard_and_mutation(self):
        rows,refs,fold=fixture(); allowed={rows[i]['episode'] for i in fold['train']}
        class Guard(dict):
            def __getitem__(self,key):
                if key not in allowed:raise AssertionError('heldout reference accessed')
                return super().__getitem__(key)
        result=calibrate_fold(rows,fold,Guard(refs));changed=copy.deepcopy(refs)
        for i in fold['heldout']:changed[rows[i]['episode']]['reference_mse']=[999.,0.,0.,0.,0.,0.,0.,0.,0.]
        self.assertEqual(result,calibrate_fold(rows,fold,changed))
        self.assertEqual(result[0]['calibration_count'],96);self.assertEqual(result[0]['heldout_count'],24)
        self.assertEqual(set(result[0]['train_image_ids'])&set(result[0]['heldout_image_ids']),set())

    def test_calibration_consumes_prior_oof_rows(self):
        rows,refs,fold=fixture();calibration,decisions=calibrate_fold(rows,fold,refs)
        expected=[rows[i] for i in fold['train']]
        self.assertEqual([r['episode'] for r in calibration['calibration_episodes']],[r['episode'] for r in expected])
        self.assertEqual({r['source_fold'] for r in calibration['calibration_episodes']},{1,2,3,4})
        changed=copy.deepcopy(rows)
        for i in fold['train']:changed[i]['predictions']=[2.]*4+[0.]+[2.]*4
        new,_=calibrate_fold(changed,fold,refs)
        self.assertNotEqual(calibration['threshold'],new['threshold'])
        self.assertTrue(all(r['predictions']==rows[i]['predictions'] for r,i in zip(decisions,fold['heldout'])))

    def test_all_thresholds_and_decisions_frozen_before_evaluation(self):
        rows,refs,_=fixture();calibrations=[];decisions=[]
        for k in range(5):
            fold=dict(fold=k,train=[i for i,r in enumerate(rows) if r['fold']!=k],heldout=[i for i,r in enumerate(rows) if r['fold']==k],
                train_image_ids=[i for i in range(40) if i%5!=k],heldout_image_ids=list(range(k,40,5)))
            calibration,heldout=calibrate_fold(rows,fold,refs);calibrations.append(calibration);decisions+=heldout
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)
            def evaluation(saved,*args):
                frozen=json.loads((output/'decisions_frozen.json').read_text())
                self.assertEqual(len(frozen['thresholds']),5)
                self.assertEqual(frozen['decisions_sha256'],sha(output/'decisions.json'))
                self.assertEqual(frozen['calibrations_sha256'],sha(output/'calibrations.json'))
                self.assertEqual(len(saved),120);self.assertEqual(len({r['episode'] for r in saved}),120)
                return dict(passed=0),saved
            with patch('ttie.boundary_confidence_run.evaluate_decisions',side_effect=evaluation):
                result=finish(output,calibrations,decisions,refs,[],[])
            self.assertEqual(result,dict(passed=0))
            frozen=json.loads((output/'decisions_frozen.json').read_text());receipt=json.loads((output/'evaluation_receipt.json').read_text())
            self.assertLessEqual(frozen['finalized_utc'],receipt['evaluation_started_utc'])

    def test_gain_coverage_diagnostics_and_quantiles(self):
        decisions=[];reference=[];old=[];pointwise=[]
        for i,(condition,mse) in enumerate(zip(('left_right','quadrants','offset_left_right_40'),(1.,3.,2.))):
            values=[4.]*9;values[0]=mse;values[4]=2.
            decisions.append(dict(episode=str(i),fold=0,predictions=[0.,2.,2.,2.,1.,2.,2.,2.,2.],selected_index=0,q=float(i)))
            reference.append(dict(episode=str(i),image_id=0,condition=condition,reference_mse=values,region2_mse=2.,selected_mse=3.))
            old.append(dict(episode=str(i),selected_mse=2.5));pointwise.append(dict(episode=str(i),selected_mse=3.))
        report,rows=evaluate_decisions(decisions,reference,old,pointwise)
        d=report['groups']['spatial_pool']['confidence_diagnostics']
        self.assertEqual(d['adapted_gains'],dict(beneficial=1,harmful=1,zero=1))
        self.assertEqual((d['adaptive'],d['canonical']),(3,0))
        self.assertEqual(q_summary([0.,1.,2.])['quantiles'],{'0':0.,'25':.5,'50':1.,'75':1.5,'100':2.})
        self.assertEqual(report['canonical_candidate_vs_region2_max_abs'],0)


if __name__=='__main__':unittest.main()
