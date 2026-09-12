import unittest
import copy
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
from ttie.geometry_attribution import quantities,dominance,interpretation,measure,report_groups,finish,sha


class AttributionTests(unittest.TestCase):
    def test_exact_values_and_three_exclusive_harm_categories(self):
        values=[5.]*27;values[12]=1.;values[0]=2.;values[13]=4.;values[1]=3.;values[4]=2.
        q=quantities(values,0)
        self.assertEqual([q[k] for k in ('S0','S1','H0','H1','S_star')],[4.,3.,1.,2.,2.])
        self.assertEqual((q['soft_joint_gain'],q['hard_gain'],q['soft_separability_regret']),(1.,-1.,1.))
        self.assertEqual(q['harmful_category'],'transfer_flip')
        values[1]=5.;self.assertEqual(quantities(values,0)['harmful_category'],'soft_interaction_failure')
        values[1]=4.;self.assertEqual(quantities(values,0)['harmful_category'],'zero/tie')
        values[0]=1.;self.assertIsNone(quantities(values,0)['harmful_category'])
        values[0]=0.;self.assertIsNone(quantities(values,0)['harmful_category'])
        values[0]=2.;values[1]=4.-1e-12;self.assertEqual(quantities(values,0)['harmful_category'],'transfer_flip')

    def test_soft_oracle_ties_and_unchanged_choice(self):
        values=[1.]*27;original=values.copy();q=quantities(values,12)
        self.assertEqual(values,original);self.assertEqual(q['soft_oracle_ties'],9)
        self.assertFalse(q['equals_first_soft_oracle']);self.assertTrue(q['in_soft_oracle_tie_set'])
        self.assertEqual(q['soft_separability_regret'],0.)
        self.assertNotIn('selected_index',q)
        with self.assertRaises(TypeError):quantities(values,12,condition='quadrants')

    def test_exact_two_thirds_both_groups_and_no_harm(self):
        a=dict(category_counts=dict(transfer_flip=2,soft_interaction_failure=1,**{'zero/tie':0}),harmful_hard_moves=3)
        b=dict(category_counts=dict(transfer_flip=1,soft_interaction_failure=2,**{'zero/tie':0}),harmful_hard_moves=3)
        self.assertEqual(interpretation(a,a),'transfer_dominant');self.assertEqual(interpretation(b,b),'interaction_dominant')
        self.assertEqual(interpretation(a,b),'mixed/inconclusive')
        empty=dict(category_counts=dict(transfer_flip=0,soft_interaction_failure=0,**{'zero/tie':0}),harmful_hard_moves=0)
        self.assertEqual(interpretation(empty,empty),'mixed/inconclusive')
        self.assertEqual(dominance(empty['category_counts'],0),'no_harmful_moves')

    def test_metadata_free_measurement_and_reporting_order(self):
        table=[{'candidate_mse':[1.]*27} for _ in range(3)]
        decisions=[dict(row_index=i,hard_index=12) for i in range(3)]
        original=copy.deepcopy(decisions);measured=measure(decisions,table)
        self.assertEqual(decisions,original)
        self.assertTrue(all('condition' not in r and 'image_id' not in r for r in measured))
        for i,c in enumerate(('left_right','quadrants','offset_left_right_40')):
            table[i].update(condition=c,image_id=str(i))
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory);raw=json.dumps(decisions).encode()
            (output/'frozen_A_decisions.json').write_bytes(raw)
            real_report=report_groups
            def observe(rows,metadata):
                frozen=json.loads((output/'quantities_frozen.json').read_text())
                self.assertEqual(frozen['quantities_sha256'],sha(output/'quantities.json'))
                self.assertFalse(frozen['family_labels_attached'])
                return real_report(rows,metadata)
            with patch('ttie.geometry_attribution.report_groups',side_effect=observe):
                result=finish(output,measured,table)
            self.assertEqual((output/'frozen_A_decisions.json').read_bytes(),raw)
            self.assertEqual(result['interpretation'],'mixed/inconclusive')
            for group in result['groups'].values():
                self.assertEqual(group['sign_contingency']['zero']['zero'],group['count'])
                self.assertEqual(group['soft_separability_regret']['zero_count'],group['count'])
                self.assertEqual(group['soft_separability_regret']['p95'],0.)
                self.assertEqual(group['in_soft_oracle_tie_set_fraction'],1.)
                self.assertEqual(group['first_soft_oracle_equal_fraction'],0.)
                self.assertEqual(group['soft_oracle_tied_episodes'],group['count'])
                self.assertTrue(all(v is None for v in group['category_fractions'].values()))


if __name__=='__main__':unittest.main()
