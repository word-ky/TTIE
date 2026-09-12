import unittest
from ttie.routing.core import BASES,PRIMARY,ORACLE,route
from ttie.routing.metrics import summarize,routing_diagnostics


def fixture():
    rows=[]
    for c in ('clean','homogeneous_dark','homogeneous_bright','left_right','quadrants','offset_left_right_40'):
        for m in ('identity','region2_direct','region2_discrete_projected','fixed_step_source',*BASES,PRIMARY,ORACLE):
            mse=.001 if c=='clean' else (.05 if m in (PRIMARY,ORACLE) else .1)
            rows.append(dict(image_id=1,condition=c,method=m,mse=mse,psnr_db=30.,clean_drift_mse=mse,
                dark_region_mse=mse,bright_region_mse=mse,loss_before=1.,loss_after=.1,steps=40,recovery_ratio=.5))
    return rows


class RoutingMetricTests(unittest.TestCase):
    def test_literal_ten_clauses_and_offset_is_primary(self):
        rows=fixture();s=summarize(rows);self.assertEqual(len(s['criteria']),10);self.assertTrue(s['qualified'])
        self.assertEqual(s['groups']['spatial_pool'][PRIMARY]['mse']['count'],3)
        for r in rows:
            if r['method']==PRIMARY and r['condition']=='offset_left_right_40':r['mse']=.3
        s=summarize(rows);self.assertIn('beyond_best_fixed',s['failed']);self.assertIn('offset_noninferiority',s['failed'])
        self.assertTrue(s['criteria']['aligned_noninferiority'])

    def test_best_fixed_is_aggregate_not_per_image_oracle(self):
        rows=fixture()
        for r in rows:
            if r['condition'] in ('left_right','quadrants','offset_left_right_40') and r['method'] in BASES:
                r['mse']=.02 if r['method']==BASES[['left_right','quadrants','offset_left_right_40'].index(r['condition'])] else .2
        s=summarize(rows);self.assertAlmostEqual(s['groups']['spatial_pool'][BASES[0]]['mse']['mean'],.14)
        self.assertAlmostEqual(s['values']['best_fixed_ratio'],.05/.14);self.assertEqual(s['best_fixed_basis_spatial'],BASES[0])

    def test_counts_full_margins_and_regret_on_selected_basis(self):
        entries=[]
        for i in range(3):
            scores=[2.,2.,2.];scores[i]=1.;d=route(*scores)
            entries.append(dict(image_id=i,condition='left_right',routing=d,
                oracle=dict(selected_basis=BASES[0],routed_mse=.1,oracle_mse=.05,mse_regret=.05,disagreement=i!=0)))
        s=routing_diagnostics(entries)['spatial_pool']
        self.assertEqual(s['basis_counts'],dict.fromkeys(BASES,1));self.assertEqual(len(s['full_energy_margins']),3)
        self.assertAlmostEqual(s['disagreement_rate'],2/3)
        self.assertEqual(s['regret_by_routed_basis'][BASES[1]]['ratio_of_mean_mse'],2.)


if __name__=='__main__':unittest.main()
