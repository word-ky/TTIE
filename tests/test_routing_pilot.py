import json
from pathlib import Path
import tempfile
import unittest
from PIL import Image
import torch
from ttie.energy_model import EnergyHead,save_energy
from ttie.routing.pilot import evaluate
from ttie.routing.core import BASES,PRIMARY
from ttie.stop_receipt import sha
from ttie.sobolev_receipt import code_hashes
from scripts.prepare_t015 import fresh
from test_semantic_ttt import scorer,RECEIPT


class RoutingPilotTests(unittest.TestCase):
    def test_fresh_six_conditions_no_training_and_persisted_route(self):
        torch.manual_seed(7)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);images=root/'images';images.mkdir();path=images/'000000000001.jpg'
            Image.new('RGB',(16,18),(115,115,115)).save(path)
            manifest=dict(images=[dict(image_id=1,filename=path.name,sha256=sha(path))])
            head=EnergyHead().eval().requires_grad_(False);save_energy(head,root/'head.pt');digest=sha(root/'head.pt')
            report,entries=evaluate(manifest,images,root/'audit',scorer(),RECEIPT,head,'cpu',max_steps=2)
            self.assertEqual(len(entries),6);self.assertEqual(len(report['criteria']),10);self.assertTrue(report['offset_is_primary'])
            self.assertFalse((root/'audit/training.json').exists());self.assertEqual(sha(root/'head.pt'),digest)
            for e in entries:
                folder=root/'audit'/e['directory'];routing=json.loads((folder/'routing.json').read_text())
                self.assertEqual(routing,e['routing'])
                self.assertEqual(len(e['energy_checkpoints']),3)
                outputs=torch.load(folder/'outputs.pt',weights_only=True)
                self.assertTrue(torch.equal(outputs[PRIMARY]['image'],outputs[routing['selected_basis']]['image']))

    def test_648_exclusions_and_frozen_old_code_inventory(self):
        logs=Path(__file__).resolve().parents[1]/'research_log'
        r=json.loads((logs/'T014_energy_receipt.json').read_text())
        self.assertEqual(code_hashes(),r['source_code_sha256'])
        with tempfile.TemporaryDirectory() as tmp:
            pool=Path(tmp)
            for i in range(1000000,1000040):Image.new('RGB',(320,321)).save(pool/f'{i:012d}.jpg')
            m=fresh(pool,logs);self.assertEqual(len(m['excluded_prior_ids']),648)
            self.assertEqual(len(m['images']),40);self.assertEqual(m['images'][0]['image_id'],1000000)
            self.assertTrue(all(e['split']=='evaluation_t015' for e in m['images']))
            self.assertEqual(m['frozen_sobolev_sha256'],r['energy_sha256'])


if __name__=='__main__':unittest.main()
