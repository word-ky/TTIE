import copy
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image
import torch
from ttie.natural import degrade
from ttie.stop_trajectory import capture,checkpoint,select_checkpoint
from ttie.stop_quality import train_head,save_head
from ttie.stop_pilot import controls,source_calibration
from ttie.stop_io import save_label_free,evaluate_reference
from ttie.stop_receipt import create_receipt,verify_receipt,verify_git_receipt,sha
from ttie.stop_metrics import PRIMARY,PRIMARY_METHOD
from ttie.restoration_metrics import evaluate_outputs
from scripts.prepare_t012 import development,fresh
from test_semantic_ttt import scorer,RECEIPT


class StopPilotTests(unittest.TestCase):
    def test_stage_b_persistence_and_reference_replacement(self):
        clean=torch.full((1,3,12,12),.45);image=degrade(clean,'left_right')
        trajectory=capture(image,scorer(),RECEIPT,max_steps=2)
        head,_=train_head(trajectory['features'],torch.linspace(.05,.01,len(trajectory['features'])))
        decision=select_checkpoint(trajectory,head)
        results=controls(image,scorer(),RECEIPT,trajectory,full=True)
        results['fixed_step_source']=checkpoint(trajectory,1);results[PRIMARY_METHOD]=checkpoint(trajectory,decision['selected_step'])
        frozen=copy.deepcopy(trajectory);original_decision=copy.deepcopy(decision)
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp)/'episode';save_label_free(directory,trajectory,results,decision)
            def measure(results,reference,**kwargs):
                receipt=json.loads((directory/'label_free_receipt.json').read_text())
                for name,r in receipt.items():self.assertEqual(sha(directory/name),r['sha256'])
                saved=json.loads((directory/'decisions.json').read_text());self.assertEqual(saved['selection'],original_decision)
                self.assertEqual(len(saved['methods']),8)
                return evaluate_outputs(results,reference,**kwargs)
            with patch('ttie.stop_io.evaluate_outputs',side_effect=measure):
                a,_=evaluate_reference(directory,trajectory,results,clean,image_id=1,condition='left_right')
                b,_=evaluate_reference(directory,trajectory,results,torch.zeros_like(clean),image_id=999,condition='quadrants')
            self.assertNotEqual(a[0]['mse'],b[0]['mse'])
            for key in ('images','features','scores','states','grids'):self.assertTrue(torch.equal(frozen[key],trajectory[key]))
            self.assertEqual(original_decision,select_checkpoint(trajectory,head))

    def test_source_training_and_calibration_end_to_end_without_fresh_data(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);out=root/'audit';out.mkdir();(out/'figures').mkdir();images=root/'images';images.mkdir()
            clean=torch.full((1,3,12,12),.45);entries=[];manifest={'images':[]}
            for image_id,split in ((1,'train_t012_stop'),(2,'calibration_t012_stop')):
                manifest['images'].append(dict(image_id=image_id,split=split));Image.new('RGB',(12,12),(115,115,115)).save(images/f'{image_id:012d}.jpg')
                for c in PRIMARY:
                    image=degrade(clean,c);t=capture(image,scorer(),RECEIPT,max_steps=2)
                    result=controls(image,scorer(),RECEIPT,t,full=False)
                    directory=out/'episodes'/str(len(entries));files=save_label_free(directory,t,result)
                    evaluate_reference(directory,t,result,clean,image_id=image_id,condition=c)
                    entries.append(dict(image_id=image_id,split=split,condition=c,directory=str(directory.relative_to(out)),files=files))
            path=root/'manifest.json';path.write_text(json.dumps(manifest))
            source_calibration(out,path,entries,'fixture-source',{},RECEIPT,images)
            report=json.loads((out/'summary.json').read_text());training=json.loads((out/'training.json').read_text())
            self.assertEqual(len(report['criteria']),5);self.assertEqual(training['images'],1);self.assertEqual(len(training['history']),100)
            receipt=json.loads((out/'T012_stopping_receipt.json').read_text())
            self.assertEqual(receipt['head_sha256'],sha(out/'head.pt'));self.assertEqual(receipt['source_manifest_sha256'],sha(path))
            self.assertFalse((root/'evaluation_manifest.json').exists())

    def test_freeze_checks_head_manifest_code_normalization_and_git_blob(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'research_log').mkdir();manifest=root/'source.json'
            manifest.write_text(json.dumps(dict(images=[dict(image_id=1,split='train_t012_stop'),dict(image_id=2,split='calibration_t012_stop')])))
            head,_=train_head(torch.randn(10,33),torch.linspace(.01,.1,10));head_path=root/'head.pt';save_head(head,head_path)
            receipt=create_receipt(head_path,manifest,'frozen',dict(passes=True),dict(selected_step=4),{},RECEIPT)
            path=root/'research_log/T012_stopping_receipt.json';path.write_text(json.dumps(receipt))
            self.assertEqual(verify_receipt(path,head_path,manifest,{},RECEIPT)['source_sha'],'frozen')
            for key,value in (('head_sha256','wrong'),('source_manifest_sha256','wrong'),('source_code_sha256',{}),('normalization',{}),('passes',False)):
                altered=dict(receipt,**{key:value});path.write_text(json.dumps(altered))
                with self.assertRaises(AssertionError):verify_receipt(path,head_path,manifest,{},RECEIPT)
            path.write_text(json.dumps(receipt));blob=path.read_bytes()
            with patch('ttie.stop_receipt.subprocess.check_output',return_value=blob):
                self.assertTrue(verify_git_receipt(root,'commit',path)['git_blob_verified'])
                path.write_text(json.dumps(dict(receipt,source_sha='changed')))
                with self.assertRaises(AssertionError):verify_git_receipt(root,'commit',path)

    def test_development_80_20_split_and_fresh_barrier_precedes_pool_read(self):
        logs=Path(__file__).resolve().parents[1]/'research_log'
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);pool=root/'pool';pool.mkdir()
            for i in range(1000000,1000102):Image.new('RGB',(320,321)).save(pool/f'{i:012d}.jpg')
            manifest=development(pool,logs);self.assertEqual(len(manifest['excluded_prior_ids']),308)
            self.assertEqual([r['image_id'] for r in manifest['images']],list(range(1000000,1000100)))
            self.assertEqual(sum(r['split']=='train_t012_stop' for r in manifest['images']),80)
            self.assertEqual(sum(r['split']=='calibration_t012_stop' for r in manifest['images']),20)
            source=root/'source.json';source.write_text(json.dumps(manifest));receipt=root/'receipt.json';receipt.write_text(json.dumps(dict(passes=False)))
            with patch('scripts.prepare_t012.evaluation_manifest',side_effect=AssertionError('pool must stay unread')) as read:
                with self.assertRaises(AssertionError):fresh(pool,logs,source,receipt,dict(git_blob_verified=True,receipt_sha256=sha(receipt)))
                read.assert_not_called()
