import copy
import inspect
import json
from pathlib import Path
import unittest
import torch
from ttie.semantic_ttt import EVGamma,CoordinateISP,FixedObjective,run_method
from ttie.offline_geometry import reference_gradient,oracle_renderer,Piecewise2,SURFACE_EV,SURFACE_GAMMA,attach_trajectory_mse,rank,correlation,surface_semantics,label_surface
from test_semantic_ttt import scorer,RECEIPT
from ttie.geometry_summary import diagnose,CONDITIONS,COORDINATES


class GeometryTests(unittest.TestCase):
    def test_recording_leaves_t008_trajectory_unchanged_and_reference_is_offline(self):
        image=torch.full((1,3,16,16),.15);objective=FixedObjective(scorer(),image,RECEIPT)
        original=run_method(image,objective,'spatial2_ttt',max_steps=3)
        traced=run_method(image,objective,'spatial2_ttt',max_steps=3,record_states=True)
        self.assertTrue(torch.equal(original['image'],traced['image']))
        self.assertEqual(original['diagnostics'],traced['diagnostics'])
        self.assertEqual(len(traced['states']),4)
        before=copy.deepcopy(traced['diagnostics'])
        a=attach_trajectory_mse(image,torch.full_like(image,.5),traced,size=2,coordinates='ev_gamma',condition='left_right')
        b=attach_trajectory_mse(image,torch.full_like(image,.8),traced,size=2,coordinates='ev_gamma',condition='quadrants')
        self.assertNotEqual(a[-1]['mse'],b[-1]['mse']);self.assertEqual(before,traced['diagnostics'])
        self.assertTrue(torch.equal(original['image'],traced['image']))

    def test_reference_gradient_changes_without_affecting_semantic_api(self):
        image=torch.full((1,3,16,16),.2)
        a=reference_gradient(image,torch.full_like(image,.5))
        b=reference_gradient(image,torch.full_like(image,.1))
        self.assertFalse(torch.equal(a,b));self.assertLess(float((a*b).sum()),0.)
        self.assertNotIn('clean',inspect.signature(run_method).parameters)
        self.assertNotIn('clean',inspect.signature(FixedObjective).parameters)
        for api in (reference_gradient,oracle_renderer):
            with self.assertRaises(AttributeError):run_method(image,api,'spatial2_ttt')

    def test_coordinate_restrictions_exact(self):
        image=torch.full((1,3,16,16),.2);objective=FixedObjective(scorer(),image,RECEIPT)
        for mode,channel,value in (('ev_only',1,1.),('gamma_only',0,0.)):
            result=run_method(image,objective,'spatial2_ttt',max_steps=3,coordinates=mode,record_states=True)
            self.assertTrue(torch.equal(result['grid'][:,channel],torch.full((1,2,2),value)))
            self.assertEqual(result['raw'].numel(),4)

    def test_piecewise_constant_equivalence_and_odd_quadrant_boundary(self):
        a=EVGamma(2);b=Piecewise2();values=torch.tensor([.5,1.25]).reshape(1,2,1,1).expand(1,2,2,2)
        a.set_grid(values);b.set_grid(values);image=torch.full((1,3,17,19),.2)
        self.assertTrue(torch.allclose(a(image),b(image),atol=2e-7))
        values=values.clone();values[0,0]=torch.tensor([[.5,-.5],[-1.,1.]])
        b.set_grid(values);field=b.parameter_field((17,19))
        self.assertTrue(torch.allclose(field[0,0,:8,:9],torch.full((8,9),.5)))
        self.assertTrue(torch.allclose(field[0,0,8:,9:],torch.ones(9,10)))

    def test_literal_surface_and_tied_ranks(self):
        self.assertEqual(SURFACE_EV,(-1.25,-1.,-.75,-.5,-.25,0.,.25,.5,.75,1.,1.25))
        self.assertEqual(SURFACE_GAMMA,(.8,.9,1.,1.1,1.25))
        self.assertEqual(rank([1,1,3]),[.5,.5,2.])
        self.assertAlmostEqual(correlation([1,2,3],[3,2,1]),-1.,places=14)
        image=torch.full((1,3,8,8),.2);objective=FixedObjective(scorer(),image,RECEIPT)
        rows=surface_semantics(image,objective);before=copy.deepcopy(rows)
        labeled,report=label_surface(image,torch.full_like(image,.5),rows,condition='homogeneous_dark')
        self.assertEqual(len(labeled),55);self.assertEqual(rows,before)
        self.assertNotIn('mse',rows[0]);self.assertIn('crossing_category',report)

    def test_oracle_resets_identity_and_is_diagnostic_only(self):
        image=torch.full((1,3,8,8),.2);clean=torch.full_like(image,.5)
        a=oracle_renderer(image,clean,'piecewise2',steps=3);b=oracle_renderer(image,clean,'piecewise2',steps=3)
        self.assertTrue(a['diagnostic_only'] and a['initial_raw_zero'])
        self.assertTrue(torch.equal(a['image'],b['image']))
        self.assertEqual(a['steps'],3);self.assertEqual(len(a['loss_trajectory']),4)
        self.assertLess(a['loss_trajectory'][-1],a['loss_trajectory'][0])

    def test_development_manifest_excludes_all_old_data(self):
        root=Path(__file__).resolve().parents[1]/'research_log';prior=set()
        for task in ('T004','T005','T006','T007','T008'):
            prior.update(r['image_id'] for r in json.loads((root/(task+'_manifest.json')).read_text())['images'])
        manifest=json.loads((root/'T009_manifest.json').read_text());rows=manifest['images'];ids=[r['image_id'] for r in rows]
        self.assertEqual(len(ids),40);self.assertEqual(ids,sorted(set(ids)));self.assertFalse(prior.intersection(ids))
        self.assertTrue(all(r['split']=='development_t009' for r in rows))
        self.assertEqual(ids,[r['image_id'] for r in manifest['inspected_prefix'] if r['eligible']])

    def test_diagnosis_rules_and_objective_priority_are_literal(self):
        records=[];renderers=[]
        for condition in CONDITIONS:
            for size in (1,2):
                for coord in COORDINATES:
                    final=.06 if coord=='ev_only' else .08
                    records.append(dict(condition=condition,size=size,coordinates=coord,active_count=0 if condition=='clean' else 1,
                        trace=[dict(mse=m,semantic_loss=l) for m,l in ((.1,2.),(.05,1.5),(final,1.))],
                        alignment=None if condition=='clean' else dict(cosine=.8)))
        for condition in ('left_right','quadrants'):
            for method in ('direct','oracle'):
                for renderer in ('bilinear2','piecewise2'):
                    value=.1 if renderer=='bilinear2' else .05
                    renderers.append(dict(condition=condition,method=method,renderer=renderer,mse=value,dark_mse=value,bright_mse=value))
        report=diagnose(records,[],renderers)
        self.assertEqual(report['rules'],dict(objective_gradient_failure=False,overcorrection_stopping_failure=True,gamma_failure=True,renderer_failure=True))
        sparse=copy.deepcopy(records)
        for r in sparse:
            if r['condition'] in ('left_right','quadrants') and r['alignment']:r['alignment']['cosine']=None
        self.assertTrue(diagnose(sparse,[],renderers)['rules']['objective_gradient_failure'])
        for r in records:
            if r['alignment']:r['alignment']['cosine']=-.2
        changed=diagnose(records,[],renderers)
        self.assertTrue(changed['rules']['objective_gradient_failure'])
        self.assertFalse(changed['rules']['overcorrection_stopping_failure'])


if __name__=='__main__':unittest.main()
