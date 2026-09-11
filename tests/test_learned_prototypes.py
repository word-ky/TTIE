import json
from pathlib import Path
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from ttie.clip_signal import FrozenCLIP, decisions
from ttie.learned_prototypes import Prototypes, train_prototypes, source_calibration, learned_scores


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight=nn.Parameter(torch.eye(3))
    def encode_image(self, pixels):
        return pixels.mean(dim=(-2,-1))@self.weight


class LearnedPrototypeTests(unittest.TestCase):
    def test_exact_initialization_and_gradient_ownership(self):
        encoder=FrozenCLIP(TinyEncoder(),torch.eye(3),size=16)
        model=Prototypes(encoder.text_prototypes)
        torch.testing.assert_close(model.vectors,encoder.text_prototypes,rtol=0,atol=0)
        image=torch.rand(1,3,32,40,requires_grad=True)
        features=encoder.image_embeddings(image).detach()
        loss=F.cross_entropy(model(features)/.07,torch.zeros(5,dtype=torch.long))
        loss.backward()
        self.assertEqual(model.vectors.shape,(3,3))
        self.assertGreater(model.vectors.grad.norm().item(),0)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in encoder.parameters()))
        self.assertIsNone(image.grad)
        torch.testing.assert_close(encoder(image),model.scores(encoder.image_embeddings(image)),rtol=0,atol=0)

    def test_source_only_training_freezes_before_inference(self):
        manifest={'images':[dict(image_id=1,split='source_train'),dict(image_id=2,split='source_calibration'),dict(image_id=3,split='evaluation')]}
        records=[dict(image_id=1,condition=c) for c in ('clean','homogeneous_dark','homogeneous_bright')]+[dict(image_id=2,condition='clean'),dict(image_id=3,condition='clean')]
        features=torch.cat((torch.eye(3),torch.ones(2,3)))
        initial=F.normalize(torch.tensor([[1.,.1,.1],[.1,1.,.1],[.1,.1,1.]]),dim=-1)
        a,history=train_prototypes(initial,features,records,manifest,steps=5)
        features[3:]=999
        b,_=train_prototypes(initial,features,records,manifest,steps=5)
        torch.testing.assert_close(a.vectors,b.vectors,rtol=0,atol=0)
        self.assertEqual(history['example_count'],3)
        self.assertEqual(history['class_counts'],[1,1,1])
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in a.parameters()))
        self.assertEqual(len(history['loss_trajectory']),6)

    def test_clean_source_calibration_only(self):
        manifest={'images':[dict(image_id=1,split='source_train'),dict(image_id=2,split='source_calibration'),dict(image_id=3,split='evaluation')]}
        rows=[dict(image_id=2,condition='clean',d_dark=i/100,d_bright=i/200) for i in range(5)]
        baseline=source_calibration(rows,manifest)
        rows.extend(dict(image_id=i,condition=c,d_dark=999,d_bright=999) for i,c in [(1,'clean'),(3,'clean'),(2,'homogeneous_dark')])
        self.assertEqual(source_calibration(rows,manifest),baseline)
        self.assertEqual(baseline['image_ids'],[2])

    def test_fresh_deterministic_manifest(self):
        root=Path(__file__).resolve().parents[1]/'research_log'
        current=json.loads((root/'T006_manifest.json').read_text())
        old=[json.loads((root/f'T{n:03}_manifest.json').read_text()) for n in (4,5)]
        previous={r['image_id'] for m in old for r in m['images']}
        ids=[r['image_id'] for r in current['images']]
        self.assertEqual(ids,sorted(set(ids)))
        self.assertFalse(set(ids)&previous)
        self.assertTrue(all(min(r['width'],r['height'])>=320 for r in current['images']))
        self.assertEqual([r['split'] for r in current['images']],['source_train']*60+['source_calibration']*20+['evaluation']*20)

    def test_pixel_only_inference_and_decisions(self):
        encoder=FrozenCLIP(TinyEncoder(),torch.eye(3),size=16)
        model=Prototypes(encoder.text_prototypes).eval().requires_grad_(False)
        image=torch.rand(1,3,32,40)
        first,zero=learned_scores(encoder,model,image)
        second,_=learned_scores(encoder,model,image)
        torch.testing.assert_close(first,second,rtol=0,atol=0)
        torch.testing.assert_close(first,zero,rtol=0,atol=0)
        cal=dict(tau=[.1,.1])
        self.assertTrue(torch.equal(decisions(first,cal)[1],decisions(second,cal)[1]))


if __name__=='__main__': unittest.main()
