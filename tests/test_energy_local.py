import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from ttie.energy_local import CROSS,choose,select,evaluate,report_groups,sha


class EnergyLocalTests(unittest.TestCase):
    def test_only_five_scores_and_exact_ties(self):
        class Scores:
            def __init__(self,v):self.v=dict(zip(CROSS,v));self.reads=[]
            def __getitem__(self,k):self.reads.append(k);return self.v[k]
        class Row:
            def __init__(self,s):self.s=s
            def __getitem__(self,k):
                if k!='energies':raise AssertionError('metadata read')
                return self.s
        for values,expected in [([1]*5,(.5,.5)),([2,1,1,1,1],(.4,.4)),([2,3,1,3,1],(.6,.6)),([1,1-1e-12,1,1,1],(.4,.5))]:
            scores=Scores(values);d=choose([Row(scores)])[0]
            self.assertEqual((d['bx'],d['by']),expected);self.assertEqual(scores.reads,list(CROSS))

    def test_non_cross_and_metadata_cannot_change_choices(self):
        row=dict(energies=[1.]*9,condition='quadrants',image_id='a',reference_mse=[0.]*9,target_bx=.4)
        before=choose([row]);changed=copy.deepcopy(row)
        for j in (0,2,6,8):changed['energies'][j]=-100.
        changed.update(condition='offset',image_id='b',reference_mse=[100.]*9,target_bx=.6)
        self.assertEqual(choose([changed]),before)

    def test_process_freeze_before_reference_open(self):
        scores={'selection':{'episodes':[dict(energies=[1.]*9) for _ in range(120)]}}
        with tempfile.TemporaryDirectory() as d:
            output=Path(d)/'run'
            with patch('ttie.energy_local.verify_source',return_value={}),patch('ttie.energy_local.load_scoring',return_value=(scores,{})),patch('ttie.energy_local.load_references',side_effect=AssertionError('reference during selection')):
                select(output,'test-source')
            before=(output/'decisions.json').read_bytes();f=json.loads((output/'decisions_frozen.json').read_text())
            self.assertFalse(f['reference_access']);self.assertFalse(f['family_or_image_id_access']);self.assertEqual(f['decisions_sha256'],sha(output/'decisions.json'))
            class StopReference(Exception):pass
            def references():
                self.assertEqual((output/'decisions.json').read_bytes(),before)
                self.assertTrue((output/'decisions_frozen.json').exists());raise StopReference()
            with patch('ttie.energy_local.verify_source',return_value={}),patch('ttie.energy_local.load_references',side_effect=references):
                with self.assertRaises(StopReference):evaluate(output)
            self.assertEqual((output/'decisions.json').read_bytes(),before)

    def test_reporting_agreement_and_harm_example_fields(self):
        rows=[dict(row_index=i,bx=.4,by=.5,target_bx=.5,target_by=.5,H0=1.,Hselected=.9,H_star=.89,x_match=False,y_match=True,joint_match=False,movement='x_only',condition=c) for i,c in enumerate(('left_right','quadrants','offset_left_right_40'))]
        r=report_groups(rows);self.assertEqual(r['passed'],5)
        rows[1]['Hselected']=1.01;self.assertTrue(report_groups(rows)['clauses']['quadrants_safety'])
        rows[1]['Hselected']+=1e-12;r=report_groups(rows);self.assertEqual(r['interpretation'],'frozen_energy_local_signal_insufficient')
        self.assertEqual(r['groups']['spatial_pool']['target_agreement']['y_match']['count'],3)
        example=r['groups']['quadrants']['harmful_examples'][0]
        self.assertEqual(set(example),{'row_index','bx','by','target_bx','target_by','H0','Hselected','H_star'})


if __name__=='__main__':unittest.main()
