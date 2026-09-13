import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from ttie.hard_local import CROSS_INDICES,choose,measure,qualify,finish,report_groups,sha


class HardLocalTests(unittest.TestCase):
    def test_five_hard_values_only_and_literal_ties(self):
        class CrossOnly:
            def __init__(self,values):self.values=dict(zip(CROSS_INDICES,values));self.reads=[]
            def __getitem__(self,index):
                if index not in self.values:raise AssertionError('read outside hard cross')
                self.reads.append(index);return self.values[index]
        for cross,expected in [([1,1,1,1,1],(.5,.5)),([2,1,1,1,1],(.4,.4)),([2,3,1,3,1],(.6,.6)),([1,1-1e-12,1,1,1],(.4,.5))]:
            values=CrossOnly(cross);d=choose([{'candidate_mse':values}])[0]
            self.assertEqual((d['bx'],d['by']),expected);self.assertEqual(values.reads,list(CROSS_INDICES))

    def test_harmful_interaction_and_oracle_ties(self):
        values=[7.]*27
        for i,v in zip(CROSS_INDICES,[2.,1.,3.,1.,3.]):values[i]=v
        values[0]=4.;table=[dict(candidate_mse=values)]
        decisions=choose(table);original=copy.deepcopy(decisions);q=measure(decisions,table)[0]
        self.assertEqual(decisions,original);self.assertEqual((q['H0'],q['H1'],q['H_star']),(2.,4.,1.))
        self.assertTrue(q['pure_interaction_failure']);self.assertTrue(q['x_nonworse'] and q['y_nonworse'])
        self.assertEqual(q['oracle_recovery'],-2.);self.assertEqual(q['oracle_tie_count'],2)
        self.assertFalse(q['in_oracle_tie_set']);self.assertEqual(q['movement'],'both')

    def test_literal_clause_boundaries(self):
        g={k:dict(mse=dict(H0=1.,H1=.95,H_star=.94)) for k in ('spatial_pool','left_right','quadrants','offset_left_right_40')}
        self.assertEqual(qualify(g)['passed'],5)
        for key in ('left_right','quadrants'):
            g[key]['mse']['H1']=1.01;self.assertTrue(qualify(g)['clauses'][key+'_safety'])
            g[key]['mse']['H1']+=1e-12;self.assertFalse(qualify(g)['clauses'][key+'_safety']);g[key]['mse']['H1']=.95
        g['offset_left_right_40']['mse']['H1']+=1e-12
        self.assertEqual(qualify(g)['interpretation'],'hard_local_factorization_insufficient')

    def test_choice_and_quantities_freeze_before_metadata(self):
        table=[dict(candidate_mse=[1.]*27,condition=c) for c in ('left_right','quadrants','offset_left_right_40')]
        decisions=choose(table)
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)
            def observed(q,t):
                for name in ('decisions','quantities'):
                    f=json.loads((output/(name+'_frozen.json')).read_text(encoding='utf-8'))
                    self.assertEqual(f[name+'_sha256'],sha(output/(name+'.json')))
                    self.assertFalse(f['family_labels_used'])
                self.assertTrue(all('condition' not in r and 'image_id' not in r for r in q))
                return report_groups(q,t)
            with patch('ttie.hard_local.report_groups',side_effect=observed):r=finish(output,decisions,table)
            self.assertEqual(json.loads((output/'decisions.json').read_text(encoding='utf-8')),decisions)
            for g in r['groups'].values():
                self.assertIsNone(g['oracle_recovery']);self.assertEqual(g['zero_headroom_count'],g['count'])
                self.assertEqual(g['oracle_tie_set_match_fraction'],1.);self.assertEqual(g['tied_oracle_episodes'],g['count'])
                self.assertEqual(g['movements']['no_move'],g['count']);self.assertEqual(g['harmful_examples'],[])


if __name__=='__main__':unittest.main()
