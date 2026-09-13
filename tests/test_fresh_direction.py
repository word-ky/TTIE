import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from PIL import Image
from test_semantic_ttt import scorer, RECEIPT
from ttie.semantic_ttt import FixedObjective
from ttie.energy_model import EnergyHead
from ttie.energy_ttt import trajectory
from ttie.soft_basis.selection import score_episode
from ttie.fresh_direction.select import cross_vectors
from ttie.fresh_direction.prepare import freeze_manifest
from ttie.fresh_direction.evaluate import summarize, evaluate
from ttie.fresh_direction.common import read, sha, CONDITIONS


class FreshDirectionTests(unittest.TestCase):
    def test_five_features_exactly_reuse_accepted_nine_candidate_pipeline(self):
        torch.manual_seed(7); torch.set_num_threads(1)
        image=torch.linspace(.02,.98,3*16*18).reshape(1,3,16,18)
        s=scorer(); head=EnergyHead().eval().requires_grad_(False)
        result,t,_=trajectory(image,s,RECEIPT,head,basis='region2',max_steps=2)
        expected=score_episode(image,result['grid'],t['gate'],RECEIPT['calibration'],s,head)
        actual=cross_vectors(image,result['grid'],t['gate'],RECEIPT['calibration'],s)
        self.assertEqual(actual,[expected['features'][i] for i in (4,1,7,3,5)])
        self.assertEqual([len(v) for v in actual],[28]*5)

    def test_manifest_is_first40_after_exclusions_and_original_eligibility(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);pool=root/'pool';pool.mkdir()
            for i in range(45):Image.new('RGB',(319 if i==2 else 320,320)).save(pool/f'{i:012d}.jpg')
            original_read=read
            prior=dict(excluded_ids=[0,1],development_T016_T018_ids=[1])
            def fake_read(p):return prior if str(p).endswith('T018E_exclusions.json') else original_read(p)
            with patch('ttie.fresh_direction.prepare.preflight',return_value=({'selector_receipt_sha256':'pin'},{})),patch('ttie.fresh_direction.prepare.read',side_effect=fake_read):
                freeze_manifest(pool,root/'cohort','source')
            manifest=read(root/'cohort/manifest.json');frozen=read(root/'cohort/manifest_frozen.json')
            self.assertEqual([e['image_id'] for e in manifest['images']],list(range(3,43)))
            self.assertEqual(frozen['manifest_sha256'],sha(root/'cohort/manifest.json'))
            self.assertFalse(frozen['reference_metric_access'])
            self.assertFalse((root/'cohort/inputs').exists())

    def test_selection_process_rejects_reference_and_mapping_reads(self):
        code="""from pathlib import Path
import tempfile
from ttie.fresh_direction.select import install_reference_barrier
with tempfile.TemporaryDirectory() as d:
 p=Path(d).resolve(); (p/'input.pt').write_bytes(b'opaque')
 blocked=install_reference_barrier(p)
 assert (p/'input.pt').read_bytes()==b'opaque'
 for name in ('manifest.json','mapping.json','source.jpg','targets.json','candidate_metrics.json'):
  try: (p/name).read_bytes()
  except RuntimeError: pass
  else: raise AssertionError(name)
 assert len(blocked)==5
"""
        result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_evaluation_cannot_open_reference_before_global_freeze(self):
        reads=[]
        def denied(path):reads.append(Path(path).name);raise FileNotFoundError(path)
        with patch('ttie.fresh_direction.evaluate.preflight',return_value=({},{})),patch('ttie.fresh_direction.evaluate.read',side_effect=denied):
            with self.assertRaises(FileNotFoundError):evaluate(Path('cohort'),Path('selected'),Path('images'),Path('out'),'source','cpu')
        self.assertEqual(reads,['decisions_frozen.json'])

    def test_literal_five_gates_and_harmful_cases(self):
        rows=[dict(row_index=i,condition=c,H0=1.,H1=h,H_star=.94) for i,(c,h) in enumerate(zip(CONDITIONS,(.96,.96,.94)))]
        result=summarize(rows);self.assertTrue(result['fresh_qualified']);self.assertEqual(result['passed'],5)
        rows[1]['H1']=1.0100000000001
        result=summarize(rows);self.assertFalse(result['clauses']['quadrants_safety'])
        self.assertFalse(result['fresh_qualified']);self.assertEqual(result['groups']['quadrants']['harmful_rows'],[1])


if __name__=='__main__':unittest.main()
