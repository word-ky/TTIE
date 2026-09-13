import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from ttie.utility_deadband import axis,choose,finish,summarize,CROSS_INDICES
from ttie.hard_local import measure


class DeadbandTests(unittest.TestCase):
    def test_exact_threshold_and_noncenter_tie_rules(self):
        self.assertEqual(axis(100.,99.,101.)['value'],.4)
        self.assertTrue(axis(100.,99.,101.)['threshold_equal'])
        self.assertEqual(axis(100.,math.nextafter(99.,math.inf),101.)['value'],.5)
        self.assertEqual(axis(100.,math.nextafter(99.,-math.inf),101.)['value'],.4)
        self.assertEqual(axis(100.,98.,98.)['value'],.4)
        self.assertTrue(axis(100.,98.,98.)['moving_tie'])
        self.assertEqual(axis(100.,101.,98.)['value'],.6)
        self.assertEqual(axis(100.,100.,100.)['value'],.5)

    def test_only_five_hard_values_and_microscopic_move_suppression(self):
        class CrossOnly:
            def __init__(self):self.reads=[]
            def __getitem__(self,j):
                self.reads.append(j)
                return dict(zip(CROSS_INDICES,[100.,99.5,101.,100.,98.]))[j]
        values=CrossOnly();d=choose([dict(candidate_mse=values)])[0]
        self.assertEqual(values.reads,list(CROSS_INDICES))
        self.assertEqual((d['bx'],d['by'],d['original_bx'],d['original_by']),(.5,.6,.4,.6))

    def test_zero_harmful_is_required_even_when_five_performance_gates_pass(self):
        def row(condition,lower=100.):
            v=[100.]*27;v[3]=lower
            return dict(condition=condition,candidate_mse=v)
        table=[row('left_right',90.),row('offset_left_right_40',80.)]+[row('quadrants') for _ in range(4)]
        table[-1]['candidate_mse'][3]=98.;table[-1]['candidate_mse'][9]=98.;table[-1]['candidate_mse'][0]=101.
        d=choose(table);report,_=summarize(measure(d,table),table,d)
        self.assertEqual(report['passed'],5)
        self.assertFalse(report['zero_harmful']);self.assertFalse(report['accepted'])
        self.assertEqual(len(report['interaction_failures']),1)
        self.assertEqual(report['interaction_failures'][0]['row_index'],5)

    def test_choices_freeze_before_family_report_and_zero_headroom_is_null(self):
        table=[dict(candidate_mse=[100.]*27,condition=c) for c in ('left_right','quadrants','offset_left_right_40')]
        d=choose(table)
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)
            def observed(q,t,choices):
                frozen=json.loads((output/'decisions_frozen.json').read_text())
                self.assertFalse(frozen['family_labels_used']);self.assertEqual(frozen['count'],3)
                self.assertTrue(all('condition' not in x and 'image_id' not in x for x in choices))
                return summarize(q,t,choices)
            with patch('ttie.utility_deadband.summarize',side_effect=observed):r=finish(output,d,table,'test-only')
            self.assertTrue(r['zero_harmful']);self.assertFalse(r['accepted'])
            self.assertIsNone(r['groups']['spatial_pool']['oracle_recovery'])
            self.assertEqual(r['groups']['spatial_pool']['targets']['joint']['0.5,0.5'],3)


if __name__=='__main__':unittest.main()
