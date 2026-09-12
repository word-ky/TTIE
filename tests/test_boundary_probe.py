import copy
import json
from pathlib import Path
import tempfile
import unittest
import torch
from ttie.boundary_probe import grouped_folds, probe_features, train_probe, predict
from ttie.energy_model import train_energy
from ttie.boundary_probe_run import train_fold
from ttie.boundary_probe_metrics import evaluate,interpretation,spearman


class ProbeTests(unittest.TestCase):
    def test_sorted_modulo_folds_keep_all_conditions_and_candidates_together(self):
        metadata=[dict(image_id=i, condition=c) for i in reversed(range(40)) for c in ('left_right','quadrants','offset')]
        seen=[]
        for fold in grouped_folds(metadata):
            self.assertEqual(fold['heldout_image_ids'],list(range(fold['fold'],40,5)))
            self.assertEqual(len(fold['heldout']),24); self.assertEqual(len(fold['train']),96)
            self.assertFalse(set(fold['train_image_ids']) & set(fold['heldout_image_ids']))
            self.assertEqual(len(fold['train_image_ids']),32); self.assertEqual(len(fold['heldout_image_ids']),8)
            seen.extend(fold['heldout'])
        self.assertEqual(sorted(seen),list(range(120)))

    def test_exact_features_dimensions_and_geometry(self):
        candidates=[(x,y,0.) for x in (.4,.5,.6) for y in (.4,.5,.6)]
        torch.manual_seed(2); features=torch.rand(3,9,28)
        x=probe_features(features.tolist(),candidates,28); y=probe_features(features.tolist(),candidates,30)
        self.assertTrue(torch.equal(x,features)); self.assertEqual(y.shape,(3,9,30))
        self.assertTrue(torch.equal(y[...,:28],features))
        self.assertTrue(torch.equal(y[0,:,28:],torch.tensor([[-1,-1],[-1,0],[-1,1],[0,-1],[0,0],[0,1],[1,-1],[1,0],[1,1]],dtype=torch.float32)))

    def test_train_only_normalization_and_exact_value_head_reuse(self):
        torch.manual_seed(4); x=torch.randn(20,28); x[:,0]=3.; mse=torch.linspace(.001,.1,20)
        head,history=train_probe(x,mse); donor,old=train_energy(x,mse)
        self.assertEqual(history,old); self.assertEqual(len(history),100)
        for k,v in head.state_dict().items(): self.assertTrue(torch.equal(v,donor.state_dict()[k]))
        self.assertTrue(torch.equal(head.x_mean,x.double().mean(0).float()))
        self.assertEqual(head.x_scale[0].item(),1.)
        expected=(mse.double()+1e-6).log()
        self.assertEqual(head.y_mean.item(),expected.mean().float().item())
        before=copy.deepcopy(head.state_dict())
        predict(head,torch.full((2,9,28),10000.))
        for k,v in head.state_dict().items(): self.assertTrue(torch.equal(v,before[k]))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in head.parameters()))

    def test_heldout_scoring_has_no_reference_argument_and_first_tie(self):
        torch.manual_seed(7); x=torch.randn(20,30); mse=torch.linspace(.01,.2,20)
        head,_=train_probe(x,mse); heldout=torch.randn(2,9,30)
        before=predict(head,heldout)
        for key in ('reference_mse','condition','image_id','labels'):
            with self.assertRaises(TypeError): predict(head,heldout,**{key:None})
        self.assertEqual(before,predict(head,heldout))
        for p in head.parameters(): p.data.zero_()
        values,chosen=predict(head,heldout)
        self.assertEqual(chosen,[0,0]); self.assertEqual(len(set(values[0])),1)

    def test_fold_training_does_not_access_heldout_targets(self):
        torch.manual_seed(3); x=torch.randn(30,9,28)
        metadata=[dict(image_id=i) for i in range(10) for _ in range(3)]
        episodes=[str(i) for i in range(30)]; fold=grouped_folds(metadata)[0]
        refs={e:dict(reference_mse=[.01+.001*int(e)]*9) for e in episodes}
        heldout={episodes[i] for i in fold['heldout']}
        class GuardedReferences(dict):
            def __getitem__(self,key):
                if key in heldout:raise AssertionError('Held-out target read during fitting/scoring')
                return super().__getitem__(key)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            a,ra=train_fold(x,GuardedReferences(refs),episodes,fold,root/'first')
            altered=copy.deepcopy(refs)
            for key in heldout:altered[key]['reference_mse']=[1000.]*9
            b,rb=train_fold(x,altered,episodes,fold,root/'second')
            self.assertEqual(a,b);self.assertEqual(ra['normalization'],rb['normalization'])
            self.assertEqual(ra['final_train_huber'],rb['final_train_huber'])
            self.assertEqual(len(json.loads((root/'first/predictions.json').read_text())),6)

    def test_metrics_and_literal_three_way_interpretation(self):
        oof=[];refs=[]
        for i,c in enumerate(('left_right','quadrants','offset_left_right_40')):
            oof.append(dict(episode=str(i),fold=0,predictions=list(range(9)),selected_index=0))
            refs.append(dict(episode=str(i),image_id=i,condition=c,reference_mse=[.9]+[1.]*8,
                             region2_mse=1.,selected_mse=1.1))
        report,_=evaluate(oof,refs)
        self.assertTrue(report['qualified']);self.assertEqual(report['passed'],5)
        for r in oof:r['selected_index']=1
        report,_=evaluate(oof,refs)
        self.assertFalse(report['qualified']);self.assertEqual(report['passed'],2)
        self.assertEqual(interpretation(True,False),'existing_28d_development_sufficient')
        self.assertEqual(interpretation(True,True),'existing_28d_development_sufficient')
        self.assertEqual(interpretation(False,True),'explicit_geometry_restores_development_rankability')
        self.assertEqual(interpretation(False,False),'neither_probe_establishes_development_rankability')
        self.assertEqual(spearman([1,2,2],[3,1,1]),-1.)
        self.assertIsNone(spearman([1]*9,list(range(9))))


if __name__=='__main__':unittest.main()
