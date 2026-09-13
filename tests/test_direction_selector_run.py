import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from ttie.direction_probe import DirectionHead
from ttie.direction_selector import CROSS_NAMES, cross_features
from ttie.direction_selector_run import train, replay, SOURCES, training_inputs
from ttie.local_geometry import sha


class FinalSelectorRunTests(unittest.TestCase):
    def test_two_fit_calls_all_rows_frozen_receipt_and_replay(self):
        torch.manual_seed(8); cross={name:torch.randn(120,28).tolist() for name in CROSS_NAMES}
        targets=[dict(bx=(.5,.4,.6)[i%3],by=(.6,.5,.4)[i%3]) for i in range(120)]
        schema=dict(dimension=28,names=[str(i) for i in range(28)])
        inputs=(cross,targets,schema,{},[str(i) for i in range(120)])
        calls=[]
        def fake_fit(z,labels):
            calls.append((z.clone(),labels.tolist()));head=DirectionHead()
            with torch.no_grad():
                head.x_mean.copy_(z.double().mean(0));head.x_scale.copy_(z.double().std(0,unbiased=False).clamp_min(1e-12))
            return head,[dict(epoch=i,train_cross_entropy=1.) for i in range(1,101)]
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'run';code={p:sha(Path(p)) for p in SOURCES}
            with patch('ttie.direction_selector_run.verify_source',return_value=code),patch('ttie.direction_selector_run.training_inputs',return_value=inputs),patch('ttie.direction_selector_run.fit_head',side_effect=fake_fit):
                train(output,'fixture')
            self.assertEqual(len(calls),2)
            for i,axis in enumerate(('x','y')):
                self.assertTrue(torch.equal(calls[i][0],cross_features(**cross)[i]))
                self.assertEqual(calls[i][1],[(.5,.4,.6).index(t['b'+axis]) for t in targets])
            receipt=json.loads((output/'selector_frozen.json').read_text(encoding='utf-8'))
            self.assertEqual(receipt['fit_calls'],dict(x=1,y=1));self.assertEqual(receipt['training_rows'],120)
            for path,expected in receipt['files_sha256'].items(): self.assertEqual(sha(output/path),expected)
            pinned=sha(output/'selector_frozen.json')
            # Training inputs and target blobs must be unavailable to replay.
            with patch('ttie.direction_selector_run.training_inputs',side_effect=AssertionError('training inputs opened')),patch('ttie.direction_selector_run.blob',side_effect=AssertionError('target opened')):
                replay(output,pinned)
            self.assertEqual(sha(output/'selector_frozen.json'),pinned)


if __name__=='__main__': unittest.main()
