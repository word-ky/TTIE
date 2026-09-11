import copy
import json
from pathlib import Path
import unittest

import torch

from ttie.natural import views
from ttie.relative_clip import DELTA_EV, probes, relative_scores, relative_calibration, relative_gate
from ttie.relative_audit import relative_rows, attach_decisions, summarize_relative


def mock_scores(image):
    means = torch.stack([v.mean() for v in views(image)])
    return torch.stack((1-means, means), dim=-1)


class RelativeTests(unittest.TestCase):
    def test_exact_probe_sign_magnitude_and_response(self):
        image = torch.linspace(0,1,3*16*20).reshape(1,3,16,20)
        plus, minus = probes(image)
        self.assertEqual(DELTA_EV, .25)
        torch.testing.assert_close(plus, (image*2**.25).clamp(0,1), rtol=0, atol=0)
        torch.testing.assert_close(minus, (image*2**(-.25)).clamp(0,1), rtol=0, atol=0)
        result = relative_scores(mock_scores,image)
        expected = torch.stack((mock_scores(image)[:,0]-mock_scores(plus)[:,0],
                                mock_scores(image)[:,1]-mock_scores(minus)[:,1]), dim=-1)
        torch.testing.assert_close(result['response'], expected, rtol=0, atol=0)
        self.assertTrue((result['response'] > 0).all())
        torch.testing.assert_close(result['targets'][:,0],mock_scores(plus)[:,0])
        torch.testing.assert_close(result['targets'][:,1],mock_scores(minus)[:,1])

    def test_fresh_manifest_and_calibration_excludes_other_rows(self):
        root=Path(__file__).resolve().parents[1]/'research_log'
        current=json.loads((root/'T005_manifest.json').read_text())
        previous=json.loads((root/'T004_manifest.json').read_text())
        self.assertEqual(len(current['images']),30)
        self.assertEqual(sum(r['split']=='calibration' for r in current['images']),10)
        self.assertFalse({r['image_id'] for r in current['images']} & {r['image_id'] for r in previous['images']})
        manifest={'images':[dict(image_id=1,split='calibration'),dict(image_id=2,split='evaluation')]}
        rows=[dict(image_id=1,condition='clean',r_dark=i/100,r_bright=i/200) for i in range(5)]
        cal=relative_calibration(rows,manifest)
        self.assertAlmostEqual(cal['tau'][0],.038)
        self.assertEqual(cal['scale'][1],.01)
        rows += [dict(image_id=2,condition='clean',r_dark=999,r_bright=999),
                 dict(image_id=1,condition='homogeneous_bright',r_dark=999,r_bright=999)]
        self.assertEqual(relative_calibration(rows,manifest),cal)

    def test_metadata_does_not_change_responses_gates_targets(self):
        image=torch.rand(1,3,16,20)
        first=relative_rows(mock_scores,image,dict(image_id=1,split='evaluation'),'clean')
        second=relative_rows(mock_scores,image,dict(image_id=999,split='calibration'),'left_right')
        cal=dict(tau=[.01,.01],scale=[.01,.01])
        attach_decisions(first,cal,cal);attach_decisions(second,cal,cal)
        keys=('d_dark','d_bright','r_dark','r_bright','active','type','target')
        self.assertEqual([[r[k] for k in keys] for r in first],[[r[k] for k in keys] for r in second])
        score=relative_scores(mock_scores,image.requires_grad_())
        winner,active,target=relative_gate(score,cal)
        self.assertFalse(target.requires_grad)
        self.assertFalse(score['response'].requires_grad)

    def test_literal_gate_uses_correct_type_tpr_and_precision(self):
        rows=[]
        for i in range(4):
            for view in ('full','top_left','top_right','bottom_left','bottom_right'):
                for cond,d,b,active,kind in [('clean',0,0,False,'dark'),('homogeneous_dark',1,0,True,'dark'),('homogeneous_bright',0,1,True,'bright')]:
                    rows.append(dict(image_id=i,view=view,condition=cond,split='evaluation',d_dark=d,d_bright=b,r_dark=d,r_bright=b,active=active,type=kind,absolute_active=active,absolute_type=kind))
        self.assertTrue(summarize_relative(rows)['stage_a_pass'])
        changed=copy.deepcopy(rows)
        for r in changed:
            if r['condition']=='homogeneous_bright': r['type']='dark'
        result=summarize_relative(changed)
        self.assertFalse(result['stage_a_pass'])
        self.assertEqual(set(result['failed_criteria']),{'bright_correct_tpr','active_type_precision'})
        for r in changed: r['active']=False
        self.assertFalse(summarize_relative(changed)['criteria']['active_type_precision'])


if __name__=='__main__':
    unittest.main()
