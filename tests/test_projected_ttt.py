import inspect
import unittest
import torch
from ttie.semantic_ttt import FixedObjective,EVGamma,Region2,run_method
from ttie.projected_ttt import ActionBox,run_candidate,run_all,METHODS
from test_semantic_ttt import scorer,RECEIPT


def pixels():
    image=torch.full((1,3,16,18),.1)
    image[:,:,:8,9:]=.9;image[:,:,8:,:9]=.5
    return image


class ProjectedTests(unittest.TestCase):
    def test_projection_bounds_signs_and_inactive_exact_identity(self):
        image=pixels();obj=FixedObjective(scorer(),image,RECEIPT);box=ActionBox(obj,2);model=Region2()
        for value in (-10.,10.):
            with torch.no_grad():model.raw.fill_(value)
            box(model);grid=model.physical_grid()[:,:2]
            self.assertTrue((grid>=box.lower-1e-7).all());self.assertTrue((grid<=box.upper+1e-7).all())
            self.assertTrue(torch.equal(model.raw[:,:,1,0],torch.zeros(1,2)))
            self.assertTrue(torch.equal(model(image)[:,:,8:,:9],image[:,:,8:,:9]))
        self.assertEqual(box.lower[0,0].tolist(),[[0.,-.5],[0.,0.]])
        self.assertEqual(box.upper[0,0].tolist(),[[.5,0.],[0.,.5]])
        self.assertTrue(torch.equal(model.physical_grid()[:,2:],torch.ones(1,4,2,2)))

    def test_global_conflict_agreement_and_no_active(self):
        for image,expected in ((pixels(),(-.5,.5)),(torch.full_like(pixels(),.1),(0.,.5)),
                               (torch.full_like(pixels(),.9),(-.5,0.)),(torch.full_like(pixels(),.5),(0.,0.))):
            box=ActionBox(FixedObjective(scorer(),image,RECEIPT),1)
            self.assertEqual((float(box.lower[0,0]),float(box.upper[0,0])),expected)

    def test_inactive_low_value_pixels_are_bitwise_unchanged(self):
        image=pixels();obj=FixedObjective(scorer(),image,RECEIPT)
        image[:,:,8:,:9]=torch.rand_like(image[:,:,8:,:9])*.02
        # Freeze a label-free gate, then test exact inactive rendering at low values.
        for method in ('region2_ttt_projected','region2_discrete_projected'):
            result=run_candidate(image,obj,method,max_steps=2)
            self.assertTrue(torch.equal(result['image'][:,:,8:,:9],image[:,:,8:,:9]))

    def test_one_step_matches_first_full_update_and_audits_projection(self):
        image=pixels();obj=FixedObjective(scorer(),image,RECEIPT)
        a=run_candidate(image,obj,'region2_ttt_projected_1step')
        b=run_candidate(image,obj,'region2_ttt_projected',max_steps=3)
        self.assertEqual(a['diagnostics']['steps'],1);self.assertEqual(len(a['states']),2)
        self.assertTrue(torch.equal(a['raw'].cpu(),b['states'][1]))
        self.assertEqual(a['diagnostics']['projections'][0],b['diagnostics']['projections'][0])
        self.assertEqual(len(b['diagnostics']['gradient_vectors']),b['diagnostics']['steps'])
        # Frozen active gate, zero current objective: exact diagnostic still updates once.
        obj.from_scores=lambda scores:scores.sum()*0
        c=run_candidate(image,obj,'region2_ttt_projected_1step')
        self.assertEqual(c['diagnostics']['steps'],1)

    def test_discrete_candidates_are_legal_and_inactive_output_unchanged(self):
        image=pixels();obj=FixedObjective(scorer(),image,RECEIPT)
        a=run_candidate(image,obj,'region2_discrete_projected')
        expected=((0.,.25,.5),(-.5,-.25,0.),(0.,),(0.,.25,.5))
        for entry in a['diagnostics']['search']:
            y,x=entry['node'];i=2*y+x
            legal=expected[i] if entry['channel']==0 else (1.,) if i==2 else (.8,1.,1.25)
            self.assertEqual(entry['candidates'],legal);self.assertIn(entry['chosen'],legal)
        self.assertTrue(torch.equal(a['image'][:,:,8:,:9],image[:,:,8:,:9]))

    def test_renderer_and_envelope_baseline_parity(self):
        image=pixels();obj=FixedObjective(scorer(),image,RECEIPT)
        for name,core,renderer in (('global_ttt_envelope','global_ttt','bilinear2'),('region2_ttt_envelope','spatial2_ttt','region2')):
            a=run_candidate(image,obj,name,max_steps=3)
            b=run_method(image,obj,core,max_steps=3,record_states=True,renderer=renderer)
            self.assertTrue(torch.equal(a['image'],b['image']));self.assertEqual(a['diagnostics'],b['diagnostics'])
        r=Region2();b=EVGamma(2);grid=torch.tensor([.4,1.1]).reshape(1,2,1,1).expand(1,2,2,2)
        r.set_grid(grid);b.set_grid(grid)
        self.assertTrue(torch.allclose(r(image),b(image),atol=2e-7))

    def test_episode_reset_no_active_and_no_metadata_api(self):
        image=pixels();a,ga=run_all(image,scorer(),RECEIPT,max_steps=2);b,gb=run_all(image,scorer(),RECEIPT,max_steps=2)
        self.assertEqual(tuple(a),METHODS);self.assertEqual(ga,gb)
        for name in a:
            self.assertTrue(torch.equal(a[name]['image'],b[name]['image']));self.assertEqual(a[name]['diagnostics'],b[name]['diagnostics'])
        clean=torch.full_like(image,.5);results,_=run_all(clean,scorer(),RECEIPT)
        for r in results.values():self.assertTrue(torch.equal(r['image'],clean));self.assertEqual(r['diagnostics']['steps'],0)
        for key in ('clean','reference','condition','mask','gain','annotations'):
            with self.assertRaises(TypeError):run_all(image,scorer(),RECEIPT,**{key:None})
