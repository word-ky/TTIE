import ast
from pathlib import Path
import unittest
from ttie.nonspatial_target import axis,target,summarize,CONDITIONS


class TargetTests(unittest.TestCase):
    def test_positive_axis_matches_original_and_zero_keeps_center(self):
        tree=ast.parse(Path('research_log/T020B_source_inputs/utility_deadband_T019A.py').read_text())
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='axis')
        namespace={'DELTA':.01};exec(compile(ast.Module(body=[function],type_ignores=[]),'original','exec'),namespace)
        for center in (1.,100.,.0001):
            for lower in (0.,.98*center,.99*center,center,2*center):
                for upper in (0.,.98*center,.99*center,center,2*center):
                    self.assertEqual(axis(center,lower,upper),namespace['axis'](center,lower,upper))
        self.assertEqual(axis(100.,99.,99.)['value'],.4)
        self.assertEqual(axis(100.,99.0000000001,100.)['value'],.5)
        for lower,upper in ((0.,0.),(1.,0.),(0.,1.),(1.,2.)):
            self.assertEqual(axis(0.,lower,upper)['value'],.5)

    def test_joint_interaction_is_reported_without_joint_reoptimization(self):
        values=[1.2,.98,1.,.98,1.,1.,1.,1.,1.]
        r=target(values);self.assertEqual((r['bx'],r['by'],r['H_delta']),(.4,.4,1.2))
        rows=[dict(row_index=i,condition=c,**target([1.]*9)) for i,c in enumerate(CONDITIONS)]
        rows[0]=dict(row_index=0,condition='clean',**r)
        s=summarize(rows);self.assertFalse(s['clauses']['zero_harmful']);self.assertFalse(s['target_viable'])

    def test_all_five_clauses_and_exact_oracle_ties(self):
        rows=[dict(row_index=i,condition=c,**target([0.]*9 if i==0 else [1.]*9)) for i,c in enumerate(CONDITIONS)]
        s=summarize(rows);self.assertEqual(s['acceptance_vector'],[True]*5)
        self.assertEqual(s['groups']['clean']['center_in_oracle_tie_fraction'],1)
        self.assertIsNone(s['groups']['clean']['ratios']['H_delta_over_H0'])
        self.assertEqual(s['groups']['nonspatial_pool']['labels']['x']['center'],3)
        rows[1]['H_delta']=1.01
        s=summarize(rows);self.assertTrue(s['clauses']['homogeneous_dark_safety']);self.assertFalse(s['clauses']['zero_harmful'])
        rows[1]['H_delta']=1.0100000000001
        self.assertFalse(summarize(rows)['clauses']['homogeneous_dark_safety'])


if __name__=='__main__':unittest.main()
