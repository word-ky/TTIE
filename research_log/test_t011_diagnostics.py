"""Hand-counted fixture for the newly requested offline statistics."""
import unittest
from t011_projection_diagnostics import episode_counts,fractions


class DiagnosticTest(unittest.TestCase):
    def test_one_ev_clip_one_no_clip_and_active_only_occupancy(self):
        post=[[[[.1,0.],[0.,0.]],[[0.,0.],[0.,0.]]]]
        pre=[[[[.2,0.],[0.,0.]],[[0.,0.],[0.,0.]]]]
        diag=dict(steps=2,projections=[dict(pre_raw=pre,post_raw=post),dict(pre_raw=post,post_raw=post)],
            final_grid=[[[[.5,0.],[0.,0.]],[[1.,1.],[1.,1.]]]],
            action_box=dict(lower=[[[[0.,0.],[0.,0.]],[[.8,1.],[1.,1.]]]],upper=[[[[.5,0.],[0.,0.]],[[1.25,1.],[1.,1.]]]]))
        r=fractions(episode_counts(diag,dict(active=[True,False,False,False])))
        self.assertEqual(r['projection_hit_rate'],.5);self.assertEqual(r['ev_update_hit_rate'],.5)
        self.assertEqual(r['ev_coordinate_hit_rate_all'],1/8);self.assertEqual(r['ev_coordinate_hit_rate_active'],.5)
        self.assertEqual(r['gamma_update_hit_rate'],0);self.assertEqual(r['final_boundary_occupancy'],.5)
        self.assertEqual(r['final_ev_boundary_occupancy'],1);self.assertEqual(r['final_gamma_boundary_occupancy'],0)
        diag['steps']=0;diag['projections']=[]
        r=fractions(episode_counts(diag,dict(active=[False]*4)))
        self.assertIsNone(r['projection_hit_rate']);self.assertIsNone(r['final_boundary_occupancy'])


if __name__=='__main__':unittest.main()
