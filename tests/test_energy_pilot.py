import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image
import torch
from ttie.energy_pilot import train_source,calibrate
from ttie.energy_model import train_energy,save_energy
from ttie.energy_receipt import create_receipt,verify_receipt,verify_git_receipt
from ttie.stop_receipt import sha
from scripts.prepare_t013 import development,fresh
from test_semantic_ttt import scorer,RECEIPT


class EnergyPilotTests(unittest.TestCase):
    def test_tiny_source_train_calibration_and_no_fresh_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);images=root/'images';images.mkdir();out=root/'audit';out.mkdir();manifest={'images':[]}
            for i,split in ((1,'train_t013_energy'),(2,'calibration_t013_energy')):
                path=images/f'{i:012d}.jpg';Image.new('RGB',(16,18),(115,115,115)).save(path)
                manifest['images'].append(dict(image_id=i,split=split,filename=path.name,sha256=sha(path)))
            head,bank=train_source(manifest,images,out,scorer(),RECEIPT,'cpu',semantic_steps=2)
            frozen={k:v.clone() for k,v in head.state_dict().items()};digest=sha(out/'energy.pt')
            report,entries=calibrate(manifest,images,out,scorer(),RECEIPT,head,'cpu',max_steps=2)
            self.assertEqual(len(bank),5);self.assertEqual(len(entries),5);self.assertEqual(len(report['criteria']),7)
            self.assertEqual(len(json.loads((out/'training.json').read_text())['history']),100)
            self.assertEqual(sha(out/'energy.pt'),digest);self.assertTrue(all(torch.equal(v,head.state_dict()[k]) for k,v in frozen.items()))
            self.assertFalse((root/'evaluation_manifest.json').exists())

    def test_frozen_energy_bank_normalization_source_and_actual_git_blob(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest=root/'manifest.json';manifest.write_text(json.dumps(dict(images=[])))
            head,_=train_energy(torch.randn(10,28),torch.linspace(.01,.1,10));hp=root/'energy.pt';save_energy(head,hp)
            receipt=create_receipt(hp,manifest,'frozen',dict(passes=True),{},RECEIPT);p=root/'receipt.json';p.write_text(json.dumps(receipt))
            self.assertEqual(verify_receipt(p,hp,manifest,{},RECEIPT)['fixed_step_source'],16)
            for key,value in (('energy_sha256','wrong'),('normalization',{}),('state_bank',{}),('passes',False),('source_code_sha256',{}),('fixed_step_source',8)):
                p.write_text(json.dumps(dict(receipt,**{key:value})))
                with self.assertRaises(AssertionError):verify_receipt(p,hp,manifest,{},RECEIPT)
            p.write_text(json.dumps(receipt));blob=p.read_bytes()
            with patch('ttie.energy_receipt.subprocess.check_output',return_value=blob):
                self.assertTrue(verify_git_receipt(root,'commit',p)['git_blob_verified'])
                p.write_text(json.dumps(dict(receipt,source_sha='modified')))
                with self.assertRaises(AssertionError):verify_git_receipt(root,'commit',p)

    def test_new_source_excludes408_and_failed_source_prevents_fresh_read(self):
        logs=Path(__file__).resolve().parents[1]/'research_log'
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pool=root/'pool';pool.mkdir()
            for i in range(1000000,1000100):Image.new('RGB',(320,321)).save(pool/f'{i:012d}.jpg')
            m=development(pool,logs);self.assertEqual(len(m['excluded_prior_ids']),408)
            self.assertEqual(sum(e['split']=='train_t013_energy' for e in m['images']),80)
            source=root/'source.json';source.write_text(json.dumps(m));p=root/'receipt.json';p.write_text(json.dumps(dict(passes=False)))
            with patch('scripts.prepare_t013.evaluation_manifest',side_effect=AssertionError('must not read pool')) as read:
                with self.assertRaises(AssertionError):fresh(pool,logs,source,p,dict(git_blob_verified=True,receipt_sha256=sha(p)))
                read.assert_not_called()
