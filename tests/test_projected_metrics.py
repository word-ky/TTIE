import unittest
from ttie.projected_ttt import METHODS
from ttie.projected_metrics import summarize,PRIMARY,STRESS,markdown
from test_residual_metrics import row


class ProjectedMetricsTests(unittest.TestCase):
    def test_literal_criteria_one_step_not_selector_stress_excluded(self):
        rows=[]
        for condition in (*PRIMARY,STRESS):
            for method in METHODS:
                value=.001 if condition=='clean' else .04 if method=='region2_ttt_projected' else .1
                rows.append(row(condition,method,value))
        report=summarize(rows);self.assertTrue(report['qualified']);self.assertEqual(len(report['criteria']),10)
        for r in rows:
            if r['method']=='region2_ttt_projected_1step':r['mse']=0.
            if r['condition']==STRESS and r['method']=='region2_ttt_projected':r['mse']=100.
        report=summarize(rows);self.assertTrue(report['qualified'])
        self.assertTrue(report['one_step_diagnostic']['heterogeneous']['one_step_better'])
        for r in rows:
            if r['method']=='region2_discrete_projected' and r['condition'] in ('left_right','quadrants'):r['mse']=.04
        report=summarize(rows);self.assertEqual(report['failed'],['beyond_discrete'])
        self.assertIn('diagnostic only',markdown(report))

    def test_each_threshold_is_enforced(self):
        rows=[row(c,m,.001 if c=='clean' else .04 if m=='region2_ttt_projected' else .1) for c in PRIMARY for m in METHODS]
        for condition,value,clause in (('clean',.006,'clean_p95'),('homogeneous_dark',.061,'homogeneous_dark'),
                                      ('homogeneous_bright',.061,'homogeneous_bright'),('quadrants',.091,'renderer_support')):
            altered=[dict(r,mse=value) if r['condition']==condition and r['method']=='region2_ttt_projected' else r for r in rows]
            self.assertFalse(summarize(altered)['criteria'][clause])
