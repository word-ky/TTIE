import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
from ttie.local_geometry import CANDIDATES,CROSS_INDICES,local_choice,choices,evaluate,finish,sha,load_inputs,precheck


class LocalGeometryTests(unittest.TestCase):
    def test_axis_rule_finite_differences_and_ties(self):
        r=local_choice([2.,1.,3.,4.,0.])
        self.assertEqual((r['bx'],r['by'],r['hard_index']),(.4,.6,6))
        self.assertEqual((r['gx'],r['gy']),(10.,-20.))
        r=local_choice([1.]*5)
        self.assertEqual((r['bx'],r['by']),(.5,.5))
        r=local_choice([2.,1.,1.,1.,1.])
        self.assertEqual((r['bx'],r['by']),(.4,.4))
        r=local_choice([1.,1.,2.,2.,1.])
        self.assertEqual((r['bx'],r['by']),(.5,.5))

    def test_rule_reads_exactly_five_soft_values_and_no_metadata(self):
        self.assertEqual(CROSS_INDICES,(13,4,22,10,16))
        self.assertEqual(len(CANDIDATES),27)
        seen=[]
        class Values:
            def __getitem__(self,index):
                if index not in CROSS_INDICES:raise AssertionError('non-cross MSE read')
                seen.append(index);return float(index)
        class Row:
            def __getitem__(self,key):
                if key!='candidate_mse':raise AssertionError('metadata read before choices frozen')
                return Values()
        result=choices([Row()]);self.assertEqual(seen,list(CROSS_INDICES));self.assertEqual(len(result),1)
        self.assertNotIn('image_id',result[0]);self.assertNotIn('condition',result[0])
        with self.assertRaises(TypeError):local_choice([1.]*5,image_id=2)
        with self.assertRaises(TypeError):local_choice([1.]*5,condition='quadrants')

    def test_non_cross_values_and_metadata_cannot_change_choice(self):
        a={'image_id':1,'condition':'left_right','candidate_mse':[1.]*27}
        b={'image_id':999,'condition':'offset','candidate_mse':[999.-i for i in range(27)]}
        for i in CROSS_INDICES:b['candidate_mse'][i]=a['candidate_mse'][i]
        self.assertEqual(choices([a]),choices([b]))

    def test_stricter_three_percent_oracle_clause_and_zero_denominators(self):
        table=[];decisions=[]
        for i,c in enumerate(('left_right','quadrants','offset_left_right_40')):
            values=[1.3]*27;values[0]=1.04;values[3]=1.;values[12]=1.2
            table.append(dict(image_id=i,condition=c,source_directory=str(i),candidate_mse=values,region2_mse=1.2))
            decisions.append(dict(row_index=i,bx=.4,by=.4,hard_index=0,gx=0.,gy=0.,cross_values=[1.]*5))
        report,_=evaluate(decisions,table)
        self.assertEqual(report['passed'],4)
        self.assertFalse(report['clauses']['spatial_within_hard_oracle_3pct'])
        for row in table:row['candidate_mse']=[1.]*27;row['region2_mse']=1.
        report,rows=evaluate(decisions,table);g=report['groups']['spatial_pool']
        self.assertIsNone(g['oracle_gain_captured'])
        self.assertEqual(g['gain_distributions']['oracle_gain_captured']['null_count'],3)
        self.assertEqual((g['zero_oracle_gain_episodes'],g['tied_oracle_episodes']),(3,3))
        self.assertTrue(all(r['oracle_minimum_ties']==9 for r in rows))

    def test_all_decisions_frozen_before_family_reporting(self):
        table=[dict(image_id=i,condition=c,source_directory=f'{i}/{c}',candidate_mse=[1.]*27,region2_mse=1.)
               for i in range(40) for c in ('left_right','quadrants','offset_left_right_40')]
        decisions=choices(table)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            def checked(saved,rows):
                frozen=json.loads((root/'decisions_frozen.json').read_text())
                self.assertEqual(frozen['choices'],120)
                self.assertEqual(frozen['decisions_sha256'],sha(root/'decisions.json'))
                self.assertFalse(frozen['family_labels_attached'])
                self.assertTrue(all('condition' not in r and 'image_id' not in r for r in saved))
                return evaluate(saved,rows)
            with patch('ttie.local_geometry.evaluate',side_effect=checked):finish(root,decisions,table)
            receipt=json.loads((root/'evaluation_receipt.json').read_text())
            self.assertEqual(receipt['decisions_sha256'],sha(root/'decisions.json'))

    def test_immutable_input_binding_and_candidate_contract(self):
        with patch('ttie.local_geometry.subprocess.check_output',return_value=b'[]'):
            with self.assertRaises(AssertionError):load_inputs()
        data=dict(candidate_metrics=[{}]*120,config=dict(candidates=list(reversed([list(c) for c in CANDIDATES]))))
        with self.assertRaises(AssertionError):precheck(data)


if __name__=='__main__':unittest.main()
