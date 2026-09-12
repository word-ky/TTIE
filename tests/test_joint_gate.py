import copy
import hashlib
import json
from pathlib import Path
import unittest

import torch
from torch.nn import functional as F

from ttie.clip_signal import decisions
from ttie.joint_gate import joint_calibration, both_gates
from ttie.learned_prototypes import Prototypes
from ttie.natural import VIEW_NAMES

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'research_log/remote_runs/20260912-071126-ttie-t006-a6000/artifacts/audit'
HASH='b4b32dbd96c65dcf606ee38d7450ebf348f5731823503b9c71ba15ec78217ac7'


class JointGateTests(unittest.TestCase):
    def test_calibration_image_max_linear_percentile_and_exclusion(self):
        manifest={'images':[dict(image_id=i,split='source_calibration') for i in (1,2)]}
        rows=[dict(image_id=i,condition='clean',view=v,d_dark=float(i+j),d_bright=-10.)
              for i in (1,2) for j,v in enumerate(VIEW_NAMES)]
        cal=dict(tau=[1.,0.],scale=[2.,1.])
        receipt=joint_calibration(rows,manifest,cal)
        self.assertEqual(receipt['image_maxima'],[2.,2.5])
        self.assertAlmostEqual(receipt['q_joint'],2.475)
        dirty=[dict(r,condition='homogeneous_dark',d_dark=1e9) for r in rows]
        heldout=[dict(r,image_id=99,d_dark=1e9) for r in rows]
        self.assertEqual(receipt,joint_calibration(rows+dirty+heldout,manifest,cal))

    def test_same_scores_raw_winner_strict_boundary_and_activation_only(self):
        scores=torch.tensor([[.1,.2],[.5,.3],[1.,1.]])
        before=scores.clone()
        receipt=dict(calibration=dict(tau=[.1,.1],scale=[.2,10.]),q_joint=.5)
        w,b,j,e=both_gates(scores,receipt)
        ow,ob=decisions(scores,receipt['calibration'])
        self.assertTrue(torch.equal(w,ow) and torch.equal(b,ob) and torch.equal(scores,before))
        self.assertEqual(w.tolist(),[1,0,0])
        self.assertEqual(j.tolist(),[False,True,True])
        exact=dict(receipt,q_joint=e[1].item())
        self.assertFalse(both_gates(scores,exact)[2][1].item())

    def test_exact_saved_prototype_and_t006_score_arithmetic(self):
        path=OLD/'prototypes.pt'
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),HASH)
        saved=torch.load(path,map_location='cpu',weights_only=True)
        prototypes=Prototypes(saved['raw']).eval().requires_grad_(False)
        features=torch.load(OLD/'source_features.pt',map_location='cpu',weights_only=True)['features'][:5]
        old_sim=features@F.normalize(saved['raw'],dim=-1).T
        self.assertTrue(torch.equal(prototypes.scores(features),old_sim[:,1:]-old_sim[:,:1]))
        self.assertTrue(torch.equal(prototypes.normalized(),saved['normalized']))

    def test_fresh_manifest_excludes_every_prior_split(self):
        prior=set()
        for task in ('T004','T005','T006'):
            prior.update(r['image_id'] for r in json.loads((ROOT/f'research_log/{task}_manifest.json').read_text())['images'])
        manifest=json.loads((ROOT/'research_log/T007_manifest.json').read_text())
        rows=manifest['images'];ids=[r['image_id'] for r in rows]
        self.assertEqual(len(ids),40)
        self.assertEqual(ids,sorted(set(ids)))
        self.assertFalse(prior.intersection(ids))
        self.assertEqual(set(manifest['excluded_prior_ids']),prior)
        self.assertTrue(all(min(r['width'],r['height'])>=320 and r['split']=='evaluation_t007' for r in rows))


if __name__=='__main__':unittest.main()
