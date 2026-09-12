import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from ttie.matched_soft import baseline,decompose,clauses_and_interpretation,finish,report_groups,sha


class MatchedSoftTests(unittest.TestCase):
    def test_exact_boundary_and_all_three_interpretations(self):
        def groups(s1,s0=1.):return {k:dict(mse=dict(H0=1.,S0=s0,S1=s1,S_star=.94)) for k in ('spatial_pool','left_right','quadrants','offset_left_right_40')}
        g=groups(.95);r=clauses_and_interpretation(g)
        self.assertEqual(r['passed'],5);self.assertEqual(r['interpretation'],'matched_soft_adaptive_geometry_viable')
        g=groups(.95,.95);self.assertEqual(clauses_and_interpretation(g)['interpretation'],'fixed_soft_dominant')
        g['quadrants']['mse']['S1']=1.01;self.assertTrue(clauses_and_interpretation(g)['clauses']['quadrants_safety'])
        g['quadrants']['mse']['S1']=1.01+1e-12;self.assertEqual(clauses_and_interpretation(g)['interpretation'],'matched_soft_insufficient')
        g=groups(.99*.95,.95);self.assertTrue(clauses_and_interpretation(g)['pooled_adaptive_1pct'])
        g['spatial_pool']['mse']['S1']+=1e-12;self.assertFalse(clauses_and_interpretation(g)['pooled_adaptive_1pct'])

    def test_decomposition_negative_and_zero_headroom(self):
        rows=[dict(row_index=0,H0=4.,S0=2.,S1=1.5,S_star=1.),dict(row_index=1,H0=4.,S0=2.,S1=3.,S_star=2.)]
        original=copy.deepcopy(rows);q=decompose(rows);self.assertEqual(rows,original)
        self.assertEqual(q[0]['fixed_smoothing_gain'],.5);self.assertEqual(q[0]['adaptive_soft_gain'],.25)
        self.assertEqual(q[0]['soft_oracle_recovery'],.5);self.assertIsNone(q[1]['soft_oracle_recovery'])
        self.assertEqual(q[1]['adaptive_soft_gain'],-.5)
        self.assertTrue(all('condition' not in r for r in q))

    def test_freeze_before_family_report_and_no_choice_mutation(self):
        rows=[dict(row_index=i,H0=4.,S0=2.,S1=2.,S_star=2.) for i in range(3)];quantities=decompose(rows)
        table=[dict(condition=c,image_id=str(i)) for i,c in enumerate(('left_right','quadrants','offset_left_right_40'))]
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory);raw=b'[{"frozen":true}]\n';(output/'frozen_A_decisions.json').write_bytes(raw)
            def observed(q,t):
                f=json.loads((output/'quantities_frozen.json').read_text(encoding='utf-8'))
                self.assertEqual(f['quantities_sha256'],sha(output/'quantities.json'));self.assertFalse(f['family_labels_attached'])
                self.assertEqual(f['decisions_sha256'],sha(output/'frozen_A_decisions.json'))
                return report_groups(q,t)
            with patch('ttie.matched_soft.report_groups',side_effect=observed):r=finish(output,quantities,table)
            self.assertEqual((output/'frozen_A_decisions.json').read_bytes(),raw)
            for g in r['groups'].values():
                self.assertIsNone(g['soft_oracle_recovery']);self.assertEqual(g['zero_soft_oracle_headroom_count'],g['count'])
                self.assertEqual(g['soft_oracle_recovery_distribution']['null_count'],g['count'])
                self.assertEqual(g['S1_vs_S0']['equal'],g['count'])


if __name__=='__main__':unittest.main()
