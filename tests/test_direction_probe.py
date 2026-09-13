import copy
import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import torch
from ttie.direction_probe import axis_features,DirectionHead,fit_axis_fold,predict,RECIPE
from ttie.direction_probe_run import freeze_oof,evaluate,confusion
from ttie.local_geometry import write,sha


class DirectionProbeTests(unittest.TestCase):
    def test_exact_feature_contrast_and_no_extra_coordinates(self):
        f=torch.arange(2*9*28,dtype=torch.float32).reshape(2,9,28);z=axis_features(f)
        self.assertEqual(z.shape,(2,2,84))
        for axis,low,high in ((0,1,7),(1,3,5)):
            self.assertTrue(torch.equal(z[:,axis],torch.cat([f[:,4],f[:,low]-f[:,4],f[:,high]-f[:,4]],1)))
        changed=f.clone();changed[:,[0,2,6,8]]+=1000;self.assertTrue(torch.equal(axis_features(changed),z))

    def test_exact_architecture_and_logit_ties(self):
        h=DirectionHead();self.assertEqual([(m.in_features,m.out_features) for m in h.net if isinstance(m,torch.nn.Linear)],[(84,64),(64,64),(64,3)])
        self.assertEqual(sum(isinstance(m,torch.nn.SiLU) for m in h.net),2)
        with torch.no_grad():
            for p in h.parameters():p.zero_()
        self.assertEqual(predict(h,torch.zeros(2,84))[1],[0,0])
        with torch.no_grad():h.net[-1].bias.copy_(torch.tensor([0.,1.,1.]))
        self.assertEqual(predict(h,torch.zeros(1,84))[1],[1])

    def test_real_training_excludes_heldout_targets_and_normalization(self):
        torch.manual_seed(21);z=torch.randn(8,2,84);z[:,:,0]=4.;fold=dict(train=list(range(6)),heldout=[6,7])
        targets=[dict(bx=(.5,.4,.6)[i%3],by=.5) for i in range(8)]
        class TrainOnly:
            def __getitem__(self,i):
                if i not in fold['train']:raise AssertionError('heldout target read')
                return targets[i]
        a,history,logits,classes=fit_axis_fold(z,TrainOnly(),fold,0)
        changed=copy.deepcopy(targets);changed[6]=dict(bx=.6,by=.6);changed[7]=dict(bx=.4,by=.4)
        b,_,_,_=fit_axis_fold(z,changed,fold,0)
        for k,v in a.state_dict().items():self.assertTrue(torch.equal(v,b.state_dict()[k]))
        self.assertEqual(len(history),100);self.assertEqual(history[-1]['epoch'],100)
        train=z[fold['train'],0].double()
        self.assertTrue(torch.equal(a.x_mean,train.mean(0).float()))
        self.assertTrue(torch.equal(a.x_scale,train.std(0,unbiased=False).clamp_min(1e-12).float()))
        self.assertEqual(len(logits),2);self.assertEqual(RECIPE['loss'],'unweighted_cross_entropy')

    def test_all_oof_freeze_before_reference_and_confusion_order(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);write(p/'config.json',dict(source_sha='mock'));write(p/'folds.json',[])
            for i in range(5):
                (p/f'fold{i}').mkdir();write(p/f'fold{i}'/'fold_frozen.json',dict(fold=i))
            decisions=[dict(row_index=i,x_logits=[0,0,0],y_logits=[0,0,0],bx=.5,by=.5) for i in range(120)]
            freeze_oof(p,decisions);f=json.loads((p/'OOF_frozen.json').read_text());self.assertEqual(f['head_count'],10)
            self.assertEqual(f['decisions_sha256'],sha(p/'decisions.json'));self.assertFalse(f['heldout_reference_evaluated'])
            class AfterFreeze(Exception):pass
            def references():
                self.assertTrue((p/'OOF_frozen.json').exists());self.assertEqual(f['decisions_sha256'],sha(p/'decisions.json'));raise AfterFreeze()
            with patch('ttie.direction_probe_run.verify_source',return_value={}),patch('ttie.direction_probe_run.load_references',side_effect=references):
                with self.assertRaises(AfterFreeze):evaluate(p)
        r=confusion([dict(target_bx=.5,target_by=.6,bx=.4,by=.6)])
        self.assertEqual(r['x'],[[0,1,0],[0,0,0],[0,0,0]]);self.assertEqual(r['y'],[[0,0,0],[0,0,0],[0,0,1]])


if __name__=='__main__':unittest.main()
