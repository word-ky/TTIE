import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image
import torch
from ttie.residual_data import evaluation_manifest
from ttie.residual_pilot import run_source,persist_then_evaluate
from ttie.residual_ttt import run_all,Rhos,METHODS
from ttie.restoration_metrics import evaluate_outputs
from test_semantic_ttt import scorer,RECEIPT


class ResidualPilotTests(unittest.TestCase):
    def test_all_eight_outputs_and_decisions_precede_reference_access(self):
        pixels=torch.full((1,3,12,12),.2);results,gate=run_all(pixels,scorer(),RECEIPT,Rhos(.5,.75),max_steps=2,record_states=True)
        original=copy.deepcopy({k:r['diagnostics'] for k,r in results.items()})
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp)/'episode'
            def measured(results,clean,**kwargs):
                pack=torch.load(directory/'outputs.pt',weights_only=True)
                decisions=json.loads((directory/'decisions.json').read_text())
                self.assertEqual(tuple(pack),METHODS);self.assertEqual(tuple(decisions['methods']),METHODS)
                self.assertTrue((directory/'states.pt').exists())
                return evaluate_outputs(results,clean,**kwargs)
            with patch('ttie.residual_pilot.evaluate_outputs',side_effect=measured):
                rows,receipt=persist_then_evaluate(results,gate,torch.ones_like(pixels),directory=directory,image_id=1,condition='left_right')
            self.assertEqual(len(rows),8);self.assertGreater(receipt['outputs']['bytes'],0)
            other=evaluate_outputs(results,torch.zeros_like(pixels),image_id=2,condition='quadrants')
            self.assertNotEqual(rows[0]['mse'],other[0]['mse']);self.assertEqual(original,{k:r['diagnostics'] for k,r in results.items()})

    def test_source_actions_include_all_fixed_pairs_without_reference_api(self):
        pixels=torch.full((1,3,8,8),.2)
        results,gate=run_source(pixels,scorer(),RECEIPT,max_steps=1)
        self.assertEqual(len(results),19);self.assertEqual(sum(k.startswith('rho_') for k in results),16)
        self.assertEqual(results['rho_d090_b090']['diagnostics']['rhos'],{'dark':.9,'bright':.9})
        with self.assertRaises(TypeError):run_source(pixels,scorer(),RECEIPT,clean=pixels)

    def test_manifest_algorithm_excludes_all_prior_ids_and_small_images(self):
        root=Path(__file__).resolve().parents[1];prior=set()
        for task in ('T004','T005','T006','T007','T008','T009'):
            prior.update(r['image_id'] for r in json.loads((root/f'research_log/{task}_manifest.json').read_text())['images'])
        self.assertEqual(len(prior),268)
        with tempfile.TemporaryDirectory() as temp:
            pool=Path(temp)
            Image.new('RGB',(320,320)).save(pool/f'{min(prior):012d}.jpg')
            Image.new('RGB',(319,330)).save(pool/'000000999999.jpg')
            for i in range(1000000,1000042):Image.new('RGB',(320,321)).save(pool/f'{i:012d}.jpg')
            manifest=evaluation_manifest(pool,prior);ids=[r['image_id'] for r in manifest['images']]
            self.assertEqual(ids,list(range(1000000,1000040)));self.assertFalse(set(ids)&prior)
            self.assertTrue(all(r['split']=='evaluation_t010' for r in manifest['images']))
