import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image
import torch
from ttie.projected_ttt import run_all,METHODS
from ttie.residual_pilot import persist_then_evaluate
from ttie.restoration_metrics import evaluate_outputs
from scripts.prepare_t011 import prepare
from test_semantic_ttt import scorer,RECEIPT
from test_projected_ttt import pixels


class ProjectedPilotTests(unittest.TestCase):
    def test_all_nine_outputs_decisions_and_hashes_precede_metrics(self):
        image=pixels();results,gate=run_all(image,scorer(),RECEIPT,max_steps=2)
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp)/'episode'
            def measure(results,clean,**kwargs):
                self.assertEqual(tuple(torch.load(directory/'outputs.pt',weights_only=True)),METHODS)
                receipt=json.loads((directory/'label_free_receipt.json').read_text())
                self.assertEqual(set(receipt),{'outputs','states','decisions'})
                for r in receipt.values():self.assertEqual(hashlib.sha256((directory/r['file']).read_bytes()).hexdigest(),r['sha256'])
                decisions=json.loads((directory/'decisions.json').read_text())
                for name in METHODS:
                    d=decisions['methods'][name]
                    if 'ttt' in name:self.assertEqual(len(d['gradient_vectors']),d['steps'])
                    if 'projected' in name and 'ttt' in name:self.assertEqual(len(d['projections']),d['steps'])
                return evaluate_outputs(results,clean,**kwargs)
            with patch('ttie.residual_pilot.evaluate_outputs',side_effect=measure):
                rows,_=persist_then_evaluate(results,gate,torch.ones_like(image),directory=directory,image_id=1,condition='left_right')
            self.assertEqual(len(rows),9)

    def test_replacement_reference_leaves_outputs_gradients_and_states_unchanged(self):
        image=pixels();a,ga=run_all(image,scorer(),RECEIPT,max_steps=3)
        saved=copy.deepcopy(a)
        rows1=evaluate_outputs(a,torch.zeros_like(image),image_id=1,condition='left_right')
        rows2=evaluate_outputs(a,torch.ones_like(image),image_id=2,condition='quadrants')
        b,gb=run_all(image,scorer(),RECEIPT,max_steps=3)
        self.assertNotEqual(rows1[0]['mse'],rows2[0]['mse']);self.assertEqual(ga,gb)
        for name in METHODS:
            self.assertTrue(torch.equal(saved[name]['image'],a[name]['image']))
            self.assertTrue(torch.equal(a[name]['image'],b[name]['image']))
            self.assertEqual(saved[name]['diagnostics'],b[name]['diagnostics'])
            for x,y in zip(a[name]['states'],b[name]['states']):self.assertTrue(torch.equal(x,y))

    def test_fresh_manifest_uses_existing_order_exclusions_and_new_split(self):
        logs=Path(__file__).resolve().parents[1]/'research_log'
        with tempfile.TemporaryDirectory() as temp:
            pool=Path(temp)
            prior=json.loads((logs/'T009_manifest.json').read_text())['images'][0]['image_id']
            Image.new('RGB',(320,321)).save(pool/f'{prior:012d}.jpg')
            Image.new('RGB',(319,330)).save(pool/'000000999999.jpg')
            for i in range(1000000,1000042):Image.new('RGB',(320,321)).save(pool/f'{i:012d}.jpg')
            manifest=prepare(pool,logs)
            self.assertEqual([r['image_id'] for r in manifest['images']],list(range(1000000,1000040)))
            self.assertEqual(len(manifest['excluded_prior_ids']),268)
            self.assertTrue(all(r['split']=='evaluation_t011' for r in manifest['images']))
