import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from ttie.boundary_probe import grouped_folds
from ttie.boundary_nested import inner_folds,run_outer
from ttie.boundary_rank_run import fit_fold
from ttie.boundary_rank import RECIPE
from ttie.boundary_nested_run import finish,evaluate_nested
from ttie.boundary_rank_run import sha,write


class NestedTests(unittest.TestCase):
    def test_exact_outer_and_inner_grouping(self):
        meta=[dict(image_id=i) for i in range(40) for _ in range(3)]
        for outer in grouped_folds(meta):
            self.assertEqual(outer['heldout_image_ids'],list(range(outer['fold'],40,5)))
            inner=inner_folds(meta,outer);seen=[]
            for k,fold in enumerate(inner):
                self.assertEqual(fold['heldout_image_ids'],outer['train_image_ids'][k::4])
                self.assertEqual((len(fold['train_image_ids']),len(fold['heldout_image_ids'])),(24,8))
                self.assertEqual((len(fold['train']),len(fold['heldout'])),(72,24))
                self.assertFalse(set(fold['train_image_ids'])&set(fold['heldout_image_ids']))
                self.assertFalse((set(fold['train_image_ids'])|set(fold['heldout_image_ids']))&set(outer['heldout_image_ids']))
                seen+=fold['heldout']
            self.assertEqual(sorted(seen),outer['train'])

    def test_five_real_heads_unchanged_by_outer_reference_mutation(self):
        torch.manual_seed(13);x=torch.randn(10,9,30)
        meta=[dict(image_id=i) for i in range(10)];episodes=[str(i) for i in range(10)];outer=grouped_folds(meta)[0]
        refs={e:dict(reference_mse=[.01*(j+1) for j in range(9)]) for e in episodes}
        allowed={episodes[i] for i in outer['train']}
        class Guard(dict):
            def __getitem__(self,key):
                if key not in allowed:raise AssertionError('outer-heldout reference read')
                return super().__getitem__(key)
        with tempfile.TemporaryDirectory() as directory:
            a=Path(directory)/'a';b=Path(directory)/'b'
            da,fa=run_outer(x,meta,episodes,Guard(refs),outer,a)
            changed=copy.deepcopy(refs)
            for i in outer['heldout']:changed[episodes[i]]['reference_mse']=[999.-j for j in range(9)]
            db,fb=run_outer(x,meta,episodes,changed,outer,b)
            self.assertEqual(da,db)
            for file in ('inner_oof.json','outer_scores.json','calibration.json','decisions.json'):
                self.assertEqual((a/file).read_bytes(),(b/file).read_bytes())
            inner=inner_folds(meta,outer)
            for name,fold in [('outer_head',outer)]+[(f'inner{k}',f) for k,f in enumerate(inner)]:
                sa=torch.load(a/name/'head.pt',weights_only=True);sb=torch.load(b/name/'head.pt',weights_only=True)
                self.assertEqual(sa['recipe'],dict(RECIPE,input_dim=30))
                for key,value in sa['state_dict'].items():self.assertTrue(torch.equal(value,sb['state_dict'][key]))
                flat=x[fold['train']].reshape(-1,30).double();scale=flat.std(0,unbiased=False)
                self.assertTrue(torch.equal(sa['state_dict']['x_mean'],flat.mean(0).float()))
                self.assertTrue(torch.equal(sa['state_dict']['x_scale'],torch.where(scale==0,torch.ones_like(scale),scale).float()))
                self.assertEqual(sa['state_dict']['y_mean'].item(),0);self.assertEqual(sa['state_dict']['y_scale'].item(),1)
                self.assertFalse(set(fa['head_receipts'][name]['train_image_ids'])&set(outer['heldout_image_ids']))
            # Reproduction of the outer head through the original, unchanged D helper.
            original,_=fit_fold(x,Guard(refs),episodes,outer,Path(directory)/'donor')
            self.assertEqual(original,json.loads((a/'outer_scores.json').read_text()))
            self.assertEqual(len(fa['files_sha256']),25)
            self.assertFalse(fa['heldout_reference_evaluation'])
            for file,digest in fa['files_sha256'].items():self.assertEqual(sha(a/file),digest)

    def test_all_folds_freeze_before_heldout_evaluation(self):
        rows=[];refs=[];old=[]
        for k in range(5):
            for j,c in enumerate(('left_right','quadrants','offset_left_right_40')):
                e=f'{k}/{j}';rows.append(dict(episode=e,fold=k,predictions=[0.,2.,2.,2.,1.,2.,2.,2.,2.],q=1.,selected_index=0))
                refs.append(dict(episode=e,image_id=k,condition=c,reference_mse=[.9]+[1.]*8,region2_mse=1.,selected_mse=1.))
                old.append(dict(episode=e,selected_mse=1.))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);receipts=[]
            for k in range(5):
                (root/f'fold{k}').mkdir();receipt=dict(threshold=.75,outer_fold=k)
                write(root/f'fold{k}'/'fold_frozen.json',receipt);receipts.append(receipt)
            def checked(decisions,reference,old_e):
                frozen=json.loads((root/'all_folds_frozen.json').read_text())
                self.assertEqual(frozen['folds'],receipts)
                self.assertEqual(frozen['decisions_sha256'],sha(root/'decisions.json'))
                self.assertTrue(all((root/f'fold{k}'/'fold_frozen.json').exists() for k in range(5)))
                return evaluate_nested(decisions,reference,old_e)
            with patch('ttie.boundary_nested_run.evaluate_nested',side_effect=checked):
                report=finish(root,rows,receipts,refs,old)
            self.assertEqual(report['passed'],5)
            self.assertEqual(report['groups']['spatial_pool']['ratios']['selected_over_old_t016e'],.9)
            self.assertEqual(report['groups']['spatial_pool']['ratios']['selected_over_ungated_nested'],1.)


if __name__=='__main__':unittest.main()
