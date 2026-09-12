"""Audit finished T012 source artifacts; run from the tested release root."""
import argparse
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path.cwd()))
from ttie.stop_receipt import sha,code_hashes
from ttie.stop_trajectory import feature_vectors,select_checkpoint,SCHEMA
from ttie.stop_quality import load_head,RECIPE
from ttie.stop_metrics import stage_a,choose_fixed_step


def main():
    parser=argparse.ArgumentParser();parser.add_argument('audit',type=Path);args=parser.parse_args()
    root=args.audit;torch.set_num_threads(1)
    entries=json.loads((root/'artifact_manifest.json').read_text());config=json.loads((root/'config.json').read_text())
    final=json.loads((root/'final_checks.json').read_text());receipt=json.loads((root/'T012_stopping_receipt.json').read_text())
    assert len(entries)==final['inputs']==500
    assert sum(e['split']=='train_t012_stop' for e in entries)==400
    assert sum(e['split']=='calibration_t012_stop' for e in entries)==100
    assert receipt['source_code_sha256']==code_hashes() and receipt['feature_schema']==SCHEMA and receipt['recipe']==RECIPE
    assert sha(root/'head.pt')==receipt['head_sha256']
    head=load_head(root/'head.pt');assert head.normalization()==receipt['normalization']
    totals=dict(inputs=0,checkpoints=0,trajectory_updates=0,no_active_inputs=0,hashed_files=0,
                checkpoint_image_bytes=0,output_bytes=0,selected_output_bytes=0,train_rows=0,calibration_rows=0)
    xs=[];ys=[]
    for e in entries:
        directory=root/e['directory'];files=e['files']
        assert files==json.loads((directory/'label_free_receipt.json').read_text())
        for name,r in files.items():
            path=directory/name;assert path.stat().st_size==r['bytes'] and sha(path)==r['sha256'];totals['hashed_files']+=1
        totals['checkpoint_image_bytes']+=files['checkpoint_images.pt']['bytes'];totals['output_bytes']+=files['outputs.pt']['bytes']
        small=torch.load(directory/'trajectory.pt',map_location='cpu',weights_only=True)
        images=torch.load(directory/'checkpoint_images.pt',map_location='cpu',weights_only=True)
        outputs=torch.load(directory/'outputs.pt',map_location='cpu',weights_only=True)
        decisions=json.loads((directory/'decisions.json').read_text());gate=decisions['gate'];d=decisions['trajectory']
        n=e['checkpoints'];assert len(images)==n==len(small['states'])==d['steps']+1
        assert small['features'].shape==(n,33) and len(d['gradient_norms'])==len(d['projections'])==n-1
        assert torch.equal(small['states'][0],torch.zeros_like(small['states'][0]))
        assert torch.isfinite(images).all() and (images>=0).all() and (images<=1).all()
        assert all(torch.isfinite(t).all() for t in small.values())
        assert torch.equal(images[-1],outputs['region2_ttt_projected']['image'])
        computed=feature_vectors(gate,small['scores'],small['grids'],d,config['frozen_receipt']['calibration'])
        assert torch.equal(computed,small['features'])
        if not any(gate['active']):
            assert n==1 and torch.equal(images[0],outputs['identity']['image']);totals['no_active_inputs']+=1
        targets=json.loads((directory/'checkpoint_metrics.json').read_text());assert len(targets)==n
        if e['split']=='train_t012_stop':
            xs.append(small['features']);ys.append(torch.tensor([r['mse'] for r in targets],dtype=torch.float64));totals['train_rows']+=n
        else:
            selection=json.loads((directory/'selection.json').read_text())
            assert selection['head_sha256']==receipt['head_sha256']
            t=dict(features=small['features'],gate=gate)
            assert selection['learned']==select_checkpoint(t,head)==select_checkpoint(t,head)
            selected=torch.load(directory/'selected_outputs.pt',weights_only=True)
            index=selection['learned']['selected_step'];fixed=min(selection['fixed_step'],n-1)
            assert torch.equal(selected['region2_ttt_learned_stop'],images[index])
            assert torch.equal(selected['fixed_step_source'],images[fixed])
            for name,r in json.loads((directory/'selection_receipt.json').read_text()).items():
                assert sha(directory/name)==r['sha256'] and (directory/name).stat().st_size==r['bytes'];totals['hashed_files']+=1
            totals['selected_output_bytes']+=(directory/'selected_outputs.pt').stat().st_size
            totals['calibration_rows']+=n
        totals['inputs']+=1;totals['checkpoints']+=n;totals['trajectory_updates']+=n-1
    x=torch.cat(xs).double();y=(torch.cat(ys)+1e-6).log()
    scale=x.std(0,unbiased=False);scale=torch.where(scale==0,torch.ones_like(scale),scale)
    yscale=y.std(unbiased=False);yscale=torch.where(yscale==0,torch.ones_like(yscale),yscale)
    assert torch.equal(head.x_mean,x.mean(0).float()) and torch.equal(head.x_scale,scale.float())
    assert torch.equal(head.y_mean,y.mean().float()) and torch.equal(head.y_scale,yscale.float())
    training=json.loads((root/'training.json').read_text());assert len(training['history'])==100 and training['rows']==len(x) and training['images']==80
    report=json.loads((root/'summary.json').read_text());rows=json.loads((root/'calibration_metrics.json').read_text())
    assert len(rows)==800 and stage_a(rows)==report
    fixed=json.loads((root/'fixed_step_calibration.json').read_text())
    assert choose_fixed_step(fixed['rows'])=={k:v for k,v in fixed.items() if k!='rows'}
    assert receipt['stage_a']==report and receipt['passes']==report['passes']
    assert totals['checkpoints']==final['checkpoints']
    result=dict(task='T012',stage='A',all_hashes_verified=True,feature_recomputation_exact=True,
                train_only_normalization_exact=True,frozen_head_scores_selections_exact=True,
                summary_fixed_step_recomputed=True,passes=report['passes'],failed=report['failed'],**totals)
    (root/'output_verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
