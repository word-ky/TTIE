import unittest
from ttie.stop_metrics import FIXED_STEPS,fixed_name,choose_fixed_step,stage_a,stage_b_stop,PRIMARY_METHOD
from ttie.residual_metrics import PRIMARY,STRESS
from test_residual_metrics import row


class StopMetricsTests(unittest.TestCase):
    def test_fixed_step_safety_then_hetero_then_earliest(self):
        self.assertEqual(FIXED_STEPS,(0,1,2,4,8,16,40));rows=[]
        for c in PRIMARY:
            for step in FIXED_STEPS:rows.append(row(c,fixed_name(step),.001 if c=='clean' else .04))
        self.assertEqual(choose_fixed_step(rows)['selected_step'],0)
        for r in rows:
            if r['method']==fixed_name(4):r['mse']=.006 if r['condition']=='clean' else .01
            if r['method']==fixed_name(8):r['mse']=.003 if r['condition']=='clean' else .03
        self.assertEqual(choose_fixed_step(rows)['selected_step'],8)

    def test_stage_a_five_clauses_and_stage_b_eleven_with_stress_exclusion(self):
        methods=('identity','region2_direct','region2_discrete_projected','global_ttt_projected','bilinear2_ttt_projected',
                 'region2_ttt_projected','fixed_step_source',PRIMARY_METHOD,'oracle_best_checkpoint')
        rows=[]
        for c in (*PRIMARY,STRESS):
            for m in methods:
                r=row(c,m,.001 if c=='clean' else .04 if m==PRIMARY_METHOD else .1)
                r['selected_step']=4;rows.append(r)
        self.assertTrue(stage_a(rows)['passes']);self.assertEqual(len(stage_a(rows)['criteria']),5)
        self.assertTrue(stage_b_stop(rows)['qualified']);self.assertEqual(len(stage_b_stop(rows)['criteria']),11)
        for r in rows:
            if r['condition']==STRESS and r['method']==PRIMARY_METHOD:r['mse']=100.
        self.assertTrue(stage_b_stop(rows)['qualified'])
        for r in rows:
            if r['condition'] in ('left_right','quadrants') and r['method']=='fixed_step_source':r['mse']=.041
        self.assertFalse(stage_a(rows)['criteria']['beyond_fixed_step'])
        self.assertFalse(stage_b_stop(rows)['criteria']['beyond_fixed_step'])
