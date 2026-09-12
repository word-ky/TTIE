import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from ttie.joint_audit import inference, persist_then_label, summarize_joint
from ttie.learned_prototypes import Prototypes
from test_prototype_audit import perfect_rows


class PixelEncoder(nn.Module):
    text_prototypes=torch.eye(3)
    def image_embeddings(self,image):
        return image.mean(dim=(-2,-1))


class JointAuditTests(unittest.TestCase):
    def test_metadata_is_attached_after_persistence_cannot_change_pixel_inference(self):
        encoder=PixelEncoder();prototypes=Prototypes(torch.eye(3)).requires_grad_(False)
        receipt=dict(calibration=dict(tau=[.1,.1],scale=[.2,.2]),q_joint=1.)
        pixels=torch.tensor([.1,.9,.2]).reshape(1,3,1,1)
        rows=inference(encoder,prototypes,pixels,receipt)
        a=[dict(image_id=1,condition='left_right',view='bottom_left')]
        b=[dict(image_id=99,condition='quadrants',view='bottom_left',mask='replaced',reference='replaced')]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'scores.json'
            labeled_a=persist_then_label(rows,a,path)
            self.assertEqual(json.loads(path.read_text()),rows)
            labeled_b=persist_then_label(inference(encoder,prototypes,pixels,receipt),b,path)
        self.assertEqual({k:labeled_a[0][k] for k in rows[0]},{k:labeled_b[0][k] for k in rows[0]})
        self.assertNotEqual(labeled_a[0]['region_true_type'],labeled_b[0]['region_true_type'])
        self.assertNotIn('condition',rows[0])

    def test_image_any_denominators_tradeoffs_auc_and_gate(self):
        original=perfect_rows()
        rows=[dict(d_dark=r['d_dark'],d_bright=r['d_bright'],type=r['type'],active_baseline=r['active'],active_joint=r['active']) for r in original]
        metadata=[{k:r[k] for k in ('image_id','condition','view')} for r in original]
        with tempfile.TemporaryDirectory() as directory:
            labeled=persist_then_label(rows,metadata,Path(directory)/'before.json')
        self.assertTrue(summarize_joint(labeled)['qualifies_later_pilot'])
        clean=[r for r in labeled if r['condition']=='clean']
        clean[0]['active_baseline']=True
        clean[1]['active_baseline']=clean[1]['active_joint']=True
        dark=[r for r in labeled if r['condition']=='homogeneous_dark']
        dark[0]['active_joint']=False
        summary=summarize_joint(labeled)
        self.assertEqual(summary['baseline']['homogeneous']['all']['clean_false_activation'],.2)
        self.assertEqual(summary['joint']['homogeneous']['all']['clean_false_activation'],.1)
        self.assertEqual(summary['joint']['homogeneous']['all']['clean_image_any_activation'],.5)
        self.assertEqual(summary['tradeoff_counts']['all']['clean_activations_removed'],1)
        self.assertEqual(summary['tradeoff_counts']['all']['homogeneous_dark_correct_activations_lost'],1)
        self.assertTrue(summary['auc_identical'])
        self.assertEqual(summary['failed_criteria'],['clean_image_any'])


if __name__=='__main__':unittest.main()
