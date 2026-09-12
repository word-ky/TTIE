import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image
import torch
from ttie.sobolev_pilot import train_source,calibrate
from ttie.sobolev_train import train_pair
from ttie.energy_model import save_energy
from ttie.sobolev_receipt import create_receipt,verify_receipt,verify_git_receipt
from ttie.stop_receipt import sha
from scripts.prepare_t014 import development,fresh
from test_semantic_ttt import scorer,RECEIPT


class SobolevPilotTests(unittest.TestCase):
    def test_tiny_complete_source_two_heads_calibration_and_no_fresh(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);images=root/'images';images.mkdir();out=root/'audit';out.mkdir();manifest={'images':[]}
            for i,split in ((1,'train_t014_sobolev'),(2,'calibration_t014_sobolev')):
                path=images/f'{i:012d}.jpg';Image.new('RGB',(16,18),(115,115,115)).save(path)
                manifest['images'].append(dict(image_id=i,split=split,filename=path.name,sha256=sha(path)))
            heads,bank=train_source(manifest,images,out,scorer(),RECEIPT,'cpu',semantic_steps=2)
            frozen={n:sha(out/(n+'.pt')) for n in heads}
            report,entries=calibrate(manifest,images,out,scorer(),RECEIPT,heads,'cpu',max_steps=2)
            self.assertEqual(len(bank),5);self.assertEqual(len(entries),5);self.assertEqual(len(report['criteria']),8)
            training=json.loads((out/'training.json').read_text())
            self.assertEqual(len(training['history']['sobolev_primary']),100)
            for n in heads:self.assertEqual(sha(out/(n+'.pt')),frozen[n])
            self.assertFalse((root/'evaluation_manifest.json').exists())
            for entry in bank:
                for record in entry['source_supervision_files'].values():self.assertEqual(sha(out/entry['directory']/record['file']),record['sha256'])

    def test_receipt_both_heads_convention_and_actual_git_blobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest=root/'manifest.json';manifest.write_text(json.dumps(dict(images=[])))
            records=dict(jacobian=torch.randn(8,28,8),reference_gradient=torch.randn(8,8),direction_mask=torch.ones(8,dtype=torch.bool))
            heads,_=train_pair(torch.randn(8,28),torch.linspace(.001,.1,8),records)
            primary=root/'primary.pt';control=root/'control.pt';save_energy(heads['sobolev_primary'],primary);save_energy(heads['value_only_control'],control)
            training=root/'training.json';training.write_text('[]')
            receipt=create_receipt(primary,control,manifest,'frozen',dict(passes=True),{},RECEIPT,training)
            path=root/'receipt.json';path.write_text(json.dumps(receipt))
            self.assertEqual(verify_receipt(path,primary,control,manifest,{},RECEIPT)['task'],'T014')
            blobs=[path.read_bytes(),primary.read_bytes(),control.read_bytes()]
            with patch('ttie.sobolev_receipt.subprocess.check_output',side_effect=blobs):
                self.assertTrue(verify_git_receipt(root,'commit',path)['both_checkpoint_blobs_verified'])
            for key,value in (('passes',False),('value_only_sha256','wrong'),('jacobian_convention',{}),('derivative_loss',{})):
                path.write_text(json.dumps(dict(receipt,**{key:value})))
                with self.assertRaises(AssertionError):verify_receipt(path,primary,control,manifest,{},RECEIPT)

    def test_508_prior_exclusions_and_failed_gate_never_reads_fresh_pool(self):
        logs=Path(__file__).resolve().parents[1]/'research_log'
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pool=root/'pool';pool.mkdir()
            for i in range(1000000,1000100):Image.new('RGB',(320,321)).save(pool/f'{i:012d}.jpg')
            m=development(pool,logs);self.assertEqual(len(m['excluded_prior_ids']),508)
            self.assertEqual(sum(e['split']=='train_t014_sobolev' for e in m['images']),80)
            source=root/'source.json';source.write_text(json.dumps(m));p=root/'receipt.json';p.write_text(json.dumps(dict(passes=False)))
            with patch('scripts.prepare_t014.evaluation_manifest',side_effect=AssertionError('must not read pool')) as read:
                with self.assertRaises(AssertionError):fresh(pool,logs,source,p,dict(git_blob_verified=True,both_checkpoint_blobs_verified=True,receipt_sha256=sha(p)))
                read.assert_not_called()


if __name__=='__main__':unittest.main()
