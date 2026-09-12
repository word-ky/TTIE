import unittest
from unittest.mock import patch
import copy
import json
from pathlib import Path
import tempfile
import torch
from ttie.boundary_rank import pairs,pair_loss,train_rank,tensor_sha
from ttie.boundary_probe import grouped_folds,probe_features,predict
from ttie.boundary_rank_run import fit_fold,finish,interpret,sha
from ttie.boundary_probe_metrics import evaluate


class RankTests(unittest.TestCase):
    def test_pairs_are_within_episode_lexicographic_and_only_exact_ties_removed(self):
        mse=torch.tensor([[0.,1.,1.,2.,3.,4.,5.,6.,7.],[1.]*9,[2.,1.,3.,4.,5.,6.,7.,8.,9.]],dtype=torch.float64)
        pair,sign=pairs(mse);expected=[];expected_sign=[]
        for e in range(3):
            for i in range(9):
                for j in range(i+1,9):
                    if mse[e,i]!=mse[e,j]:
                        expected.append([e,i,j]);expected_sign.append(1. if mse[e,i]<mse[e,j] else -1.)
        self.assertEqual(pair.tolist(),expected);self.assertEqual(sign.tolist(),expected_sign)
        self.assertEqual(len(pair),71);self.assertNotIn(1,pair[:,0].tolist())
        near=torch.ones(1,9,dtype=torch.float64);near[0,1]+=1e-12
        self.assertEqual(len(pairs(near)[0]),8)

    def test_logistic_sign_equation_and_gradient_direction(self):
        left=torch.tensor([-1.,1.],requires_grad=True);right=-left.detach();sign=torch.tensor([1.,-1.])
        loss=pair_loss(left,right,sign)
        self.assertAlmostEqual(loss.item(),torch.nn.functional.softplus(torch.tensor(-2.)).item())
        self.assertLess(loss.item(),pair_loss(right,left,sign).item())
        loss.backward();self.assertGreater(left.grad[0].item(),0);self.assertLess(left.grad[1].item(),0)

    def test_train_only_input_normalization_all_pairs_each_epoch_and_repeatability(self):
        torch.manual_seed(3);x=torch.randn(2,9,28);x[:,:,0]=2.;mse=torch.arange(18).reshape(2,9).double()
        head,history,receipt=train_rank(x,mse)
        same,again,other=train_rank(x,mse)
        self.assertEqual(history,again);self.assertEqual(receipt,other)
        self.assertEqual(len(history),100);self.assertEqual(receipt['non_tied_pairs'],72)
        generator=torch.Generator().manual_seed(7)
        for h in history:
            order=torch.randperm(72,generator=generator)
            self.assertEqual(h['pairs_seen'],72);self.assertEqual(h['permutation_sha256'],tensor_sha(order))
        self.assertTrue(torch.equal(head.x_mean,x.reshape(-1,28).double().mean(0).float()))
        self.assertEqual(head.x_scale[0].item(),1.)
        self.assertEqual(head.y_mean.item(),0.);self.assertEqual(head.y_scale.item(),1.)
        for k,v in head.state_dict().items():self.assertTrue(torch.equal(v,same.state_dict()[k]))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in head.parameters()))

    def test_folds_features_and_inference_metadata_ties(self):
        meta=[dict(image_id=i) for i in range(40) for _ in range(3)]
        for fold in grouped_folds(meta):
            self.assertEqual(fold['heldout_image_ids'],list(range(fold['fold'],40,5)))
            self.assertFalse(set(fold['train_image_ids'])&set(fold['heldout_image_ids']))
        raw=torch.zeros(2,9,28);candidates=[(x,y,0.) for x in (.4,.5,.6) for y in (.4,.5,.6)]
        x=probe_features(raw.tolist(),candidates,30)
        self.assertEqual(x.shape,(2,9,30));self.assertTrue(torch.equal(x[:,:,:28],raw))
        self.assertEqual(x[0,0,-2:].tolist(),[-1.,-1.]);self.assertEqual(x[0,4,-2:].tolist(),[0.,0.])
        from ttie.stop_quality import QualityHead
        head=QualityHead(30,torch.nn.SiLU)
        for p in head.parameters():p.data.zero_()
        self.assertEqual(predict(head,x)[1],[0,0])
        for key in ('reference_mse','image_id','condition','candidate_id'):
            with self.assertRaises(TypeError):predict(head,x,**{key:None})

    def test_heldout_targets_never_read_by_fold_training_or_scoring(self):
        torch.manual_seed(2);x=torch.randn(10,9,28)
        meta=[dict(image_id=i) for i in range(10)];fold=grouped_folds(meta)[0]
        episodes=list(map(str,range(10)));heldout={episodes[i] for i in fold['heldout']}
        refs={e:dict(reference_mse=[.01*(i+1) for i in range(9)]) for e in episodes}
        class Guarded(dict):
            def __getitem__(self,key):
                if key in heldout:raise AssertionError('Heldout target entered rank fitting/scoring')
                return super().__getitem__(key)
        with tempfile.TemporaryDirectory() as tmp:
            a,ra=fit_fold(x,Guarded(refs),episodes,fold,Path(tmp)/'a')
            changed=copy.deepcopy(refs)
            for e in heldout:changed[e]['reference_mse']=[1000.-i for i in range(9)]
            b,rb=fit_fold(x,changed,episodes,fold,Path(tmp)/'b')
            self.assertEqual(a,b);self.assertEqual(ra['normalization'],rb['normalization'])
            self.assertEqual(ra['pair_index_sha256'],rb['pair_index_sha256'])
            self.assertEqual(ra['pair_sign_sha256'],rb['pair_sign_sha256'])
            self.assertEqual(ra['final_train_pairwise_loss'],rb['final_train_pairwise_loss'])

    def test_both_oof_tables_frozen_before_evaluation_and_literal_interpretation(self):
        rows=[];reference=[]
        for i,c in enumerate(('left_right','quadrants','offset_left_right_40')):
            rows.append(dict(episode=str(i),fold=0,predictions=list(range(9)),selected_index=0))
            reference.append(dict(episode=str(i),image_id=i,condition=c,reference_mse=[.9]+[1.]*8,region2_mse=1.,selected_mse=1.1))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pred={n:copy.deepcopy(rows) for n in ('rank28','rank30')}
            training={n:[dict(final_train_pairwise_loss=.2,non_tied_pairs=8)]*5 for n in pred}
            for name in pred:(root/name).mkdir()
            def checked_evaluate(oof,refs):
                frozen=json.loads((root/'OOF_frozen.json').read_text())
                for name in pred:self.assertEqual(sha(root/name/'oof.json'),frozen['probes'][name]['oof_sha256'])
                return evaluate(oof,refs)
            with patch('ttie.boundary_rank_run.evaluate',side_effect=checked_evaluate) as checked:
                result=finish(root,pred,training,reference,dict(probe28=reference,probe30=reference))
                self.assertEqual(checked.call_count,2)
            self.assertTrue(result['probes']['rank28']['qualified'])
        self.assertEqual(interpret(True,True),'28d_development_rankable_under_pairwise_supervision')
        self.assertEqual(interpret(True,False),'28d_development_rankable_under_pairwise_supervision')
        self.assertEqual(interpret(False,True),'geometry_and_pairwise_supervision_jointly_restore_development_rankability')
        self.assertEqual(interpret(False,False),'neither_rank_probe_establishes_safe_development_ranking')


if __name__=='__main__':unittest.main()
