"""Post-run T014 evidence audit; run from the immutable experiment release.

Reuses the T013 audit's saved-pixel/receipt/feature/selection checks. Does not
rerun optimization, refit either head, modify a gate, or load fresh evaluation.
Full source Jacobians are checked as stored training records; direct-autograd
equivalence is the separately retained actual-CLIP fixture, not an all-row rerun.
"""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
sys.path.insert(0,str(Path.cwd()))
import torch
from ttie.energy_model import features,load_energy,SCHEMA,RECIPE
from ttie.energy_bank import BANK
from ttie.sobolev_source import JACOBIAN
from ttie.sobolev_train import DERIVATIVE_LOSS,training_statistics
from ttie.sobolev_receipt import code_hashes
from ttie.sobolev_metrics import stage_a,summarize_trajectories,trajectory_diagnostics
from ttie.sobolev_io import PRIMARY_METHOD,CONTROL
from ttie.stop_receipt import sha
from ttie.stop_trajectory import feature_vectors
from ttie.natural import load_image,degrade
from ttie.restoration_metrics import evaluate_outputs


def main():
    p=argparse.ArgumentParser();p.add_argument('audit',type=Path);p.add_argument('--images',type=Path,required=True);args=p.parse_args()
    root=args.audit;read=lambda p:json.loads(p.read_text());torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    config=read(root/'config.json');receipt=read(root/'T014_energy_receipt.json');final=read(root/'final_checks.json')
    assert receipt['source_code_sha256']==code_hashes() and receipt['feature_schema']==SCHEMA and receipt['recipe']==RECIPE and receipt['state_bank']==BANK
    assert receipt['jacobian_convention']==JACOBIAN and receipt['derivative_loss']==DERIVATIVE_LOSS
    heads={name:load_energy(root/(name+'.pt')) for name in ('value_only_control','sobolev_primary')}
    gpu_heads={name:load_energy(root/(name+'.pt')).cuda() for name in heads}
    hashes={name:sha(root/(name+'.pt')) for name in heads}
    assert hashes==final['energy_sha256'] and hashes['sobolev_primary']==receipt['energy_sha256'] and hashes['value_only_control']==receipt['value_only_sha256']
    assert heads['sobolev_primary'].normalization()==heads['value_only_control'].normalization()==receipt['normalization']==receipt['value_only_normalization']
    assert sha(Path(config['model_identity']['path']))==config['frozen_receipt']['model_identity']['sha256']
    prototype=Path('research_log/remote_runs/20260912-071126-ttie-t006-a6000/artifacts/audit/prototypes.pt')
    assert sha(prototype)==config['frozen_receipt']['prototype_identity']['sha256']
    assert read(Path('research_log/T007_joint_calibration.json'))==config['frozen_receipt']
    totals=dict(hashed_files=0,large_image_bytes=0,training_inputs=0,training_states=0,source_direction_rows=0,calibration_inputs=0,
                energy_checkpoints=0,energy_updates=0,semantic_checkpoints=0,inactive_region_checks=0,no_active_inputs=0,
                source_rerender_changed_rows=0,source_rerender_max_abs=0.)
    def verify_files(directory,files):
        for name,r in files.items():
            file=directory/name;assert file.stat().st_size==r['bytes'] and sha(file)==r['sha256'];totals['hashed_files']+=1
            if name in ('bank_images.pt','checkpoint_images.pt','outputs.pt'):totals['large_image_bytes']+=r['bytes']
    def obj(gate):
        return SimpleNamespace(active=torch.tensor(gate['active']),winner=torch.tensor(gate['winner']),
            evidence=torch.tensor(gate['evidence'],dtype=torch.float64),calibration=config['frozen_receipt']['calibration'])
    def inactive_checks(images,image,gate):
        h,w=image.shape[-2:]
        for i,active in enumerate(gate['active']):
            if active:continue
            y=slice(0,h//2) if i<2 else slice(h//2,h);x=slice(0,w//2) if i%2==0 else slice(w//2,w)
            assert torch.equal(images[...,y,x],image[...,y,x].expand(len(images),-1,-1,-1,-1))
            totals['inactive_region_checks']+=len(images)
    train_entries=read(root/'training_manifest.json');xs=[];ys=[];derivatives=[]
    assert receipt['source_records_manifest']==train_entries and receipt['source_records_manifest_sha256']==sha(root/'training_manifest.json')
    for e in train_entries:
        d=root/e['directory'];verify_files(d,e['files']);verify_files(d,e['source_supervision_files'])
        small=torch.load(d/'bank.pt',weights_only=True);images=torch.load(d/'bank_images.pt',weights_only=True)
        records=torch.load(d/'source_derivatives.pt',weights_only=True);dec=read(d/'bank_decisions.json');gate=dec['gate']
        expected=BANK['order'] if any(gate['active']) else ['identity']
        assert dec['names']==expected and len(images)==e['states']==len(expected)
        assert torch.isfinite(images).all() and (images>=0).all() and (images<=1).all()
        assert all(torch.isfinite(t).all() for t in (*small.values(),*records.values()))
        f=torch.stack([features(obj(gate),s,g) for s,g in zip(small['scores'],small['grids'])]);assert torch.equal(f,small['features'])
        assert torch.equal(records['features'],f) and records['jacobian'].shape==(len(f),28,8)
        assert (records['jacobian'][:,:12]==0).all()
        assert (records['active']==any(gate['active'])).all()
        assert torch.equal(records['direction_mask'],records['active']&(records['reference_gradient'].norm(dim=1)>0))
        assert int(records['direction_mask'].sum())==e['direction_rows']
        clean=load_image(args.images/f"{e['image_id']:012d}.jpg");image=degrade(clean,e['condition'])
        inactive_checks(images,image,gate)
        targets=read(d/'targets.json');values=[float((v-clean).square().mean()) for v in images]
        assert values==[t['mse'] for t in targets]
        convention=read(d/'derivative_convention.json');assert convention['convention']==JACOBIAN
        deltas=convention['inherited_pixel_max_abs_differences'];assert len(deltas)==len(images)
        totals['source_rerender_changed_rows']+=sum(v>0 for v in deltas)
        totals['source_rerender_max_abs']=max(totals['source_rerender_max_abs'],max(deltas))
        xs.append(f);ys.append(torch.tensor(values,dtype=torch.float64));derivatives.append(records)
        totals['training_inputs']+=1;totals['training_states']+=len(images);totals['source_direction_rows']+=e['direction_rows']
    x=torch.cat(xs);mse=torch.cat(ys);target=(mse+1e-6).log();head=heads['sobolev_primary']
    scale=x.double().std(0,unbiased=False);scale=torch.where(scale==0,torch.ones_like(scale),scale)
    yscale=target.std(unbiased=False);yscale=torch.where(yscale==0,torch.ones_like(yscale),yscale)
    assert torch.equal(head.x_mean,x.double().mean(0).float()) and torch.equal(head.x_scale,scale.float())
    assert torch.equal(head.y_mean,target.mean().float()) and torch.equal(head.y_scale,yscale.float())
    records={k:torch.cat([r[k] for r in derivatives]) for k in derivatives[0]}
    train=read(root/'training.json');assert train['rows']==len(x) and train['images']==80
    for name,head in heads.items():
        assert len(train['history'][name])==100
        assert training_statistics(head,x,mse,records)==train['final_train_statistics'][name]
    print('verified source banks and both final training fits',totals['training_states'],flush=True)
    entries=read(root/'artifact_manifest.json');rows=[];alignments=[]
    for e in entries:
        d=root/e['directory'];clean=load_image(args.images/f"{e['image_id']:012d}.jpg");image=degrade(clean,e['condition'])
        assert e['files']==read(d/'label_free_receipt.json');verify_files(d,e['files']['episode'])
        output=torch.load(d/'outputs.pt',weights_only=True);decisions=read(d/'decisions.json')
        results={m:dict(**r,diagnostics=decisions['methods'][m]) for m,r in output.items()}
        case=read(d/'metrics.json');measured=evaluate_outputs(results,clean,image_id=e['image_id'],condition=e['condition'])
        for actual,saved in zip(measured,case):assert all(actual[k]==saved[k] for k in actual)
        ts={};ds={}
        for method,files in e['files'].items():
            if method=='episode':continue
            folder=d/method;verify_files(folder,files)
            small=torch.load(folder/'trajectory.pt',weights_only=True);images=torch.load(folder/'checkpoint_images.pt',weights_only=True)
            info=read(folder/'decisions.json');gate=info['gate'];diag=info['trajectory'];n=len(images)
            assert all(torch.isfinite(t).all() for t in small.values())
            assert torch.isfinite(images).all() and (images>=0).all() and (images<=1).all()
            assert n==len(small['states'])==len(diag['gradient_vectors'])+1==len(diag['projections'])+1
            assert torch.equal(small['states'][0],torch.zeros_like(small['states'][0]))
            if method=='semantic':
                computed=feature_vectors(gate,small['scores'],small['grids'],diag,config['frozen_receipt']['calibration'])
                assert torch.equal(computed,small['features']);totals['semantic_checkpoints']+=n
                assert torch.equal(output['fixed_step_source']['image'],images[min(16,n-1)])
                assert torch.equal(output['region2_ttt_projected']['image'],images[-1]);continue
            assert n==(41 if any(gate['active']) else 1)
            f=torch.stack([features(obj(gate),s,g) for s,g in zip(small['scores'],small['grids'])]);assert torch.equal(f,small['features'])
            gpu=gpu_heads['value_only_control' if method==CONTROL else 'sobolev_primary']
            with torch.no_grad():energies=[float(gpu(row.cuda()).squeeze()) for row in f]
            assert energies==diag['loss_trajectory']==info['selection']['scores']
            index=min(range(n),key=lambda i:(energies[i],i));assert index==info['selection']['selected_step']
            assert torch.equal(output[method]['image'],images[index]) and torch.equal(output[method]['raw'],small['states'][index])
            assert torch.isfinite(torch.tensor(diag['gradient_vectors'])).all()
            lo=torch.tensor(diag['action_box']['lower']);hi=torch.tensor(diag['action_box']['upper'])
            assert (small['grids']>=lo-1e-7).all() and (small['grids']<=hi+1e-7).all()
            totals['energy_checkpoints']+=n;totals['energy_updates']+=n-1
            ts[method]=dict(**small,diagnostics=diag);ds[method]=info['selection']
            if method in (PRIMARY_METHOD,CONTROL):inactive_checks(images,image,gate)
            if method==PRIMARY_METHOD:
                if not any(gate['active']):totals['no_active_inputs']+=1
                values=[float((v-clean).square().mean()) for v in images]
                assert values==[r['mse'] for r in read(d/'checkpoint_metrics.json')]
                oracle=min(range(n),key=lambda i:(values[i],i));assert read(d/'oracle_diagnostic.json')['selected_step']==oracle
                assert case[-1]['mse']==values[oracle]
        assert trajectory_diagnostics(ts,ds)==e['trajectory_diagnostics']
        if (d/'alignments.json').exists():
            for a in read(d/'alignments.json'):
                g=torch.tensor(a['energy_gradient'],dtype=torch.float64);r=torch.tensor(a['reference_gradient'],dtype=torch.float64)
                assert torch.equal(g,torch.tensor(ts[a['method']]['diagnostics']['gradient_vectors'][0],dtype=torch.float64).flatten())
                denom=g.norm()*r.norm();cos=float(g@r/denom) if float(denom)>0 else 0.;assert cos==a['cosine'];alignments.append(a)
        rows.extend(case);totals['calibration_inputs']+=1
    assert rows==read(root/'metrics.json') and alignments==read(root/'alignments.json')
    computed=stage_a(rows,alignments);computed['trajectory_distributions']=summarize_trajectories(entries)
    report=read(root/'summary.json')
    # JSON integer histogram keys serialize as strings.
    assert json.loads(json.dumps(computed))==report==receipt['stage_a']
    assert totals['training_inputs']==400 and totals['calibration_inputs']==100
    assert totals['training_states']==final['training_states'] and totals['energy_checkpoints']==final['energy_checkpoints']
    result=dict(task='T014',stage='A',all_hashes_verified=True,all_stored_mse_recomputed_exact=True,
        train_only_normalization_and_final_fits_exact=True,cached_source_records_checked=True,
        jacobian_direct_equivalence_scope='separate fixed actual-CLIP24state feature / identity gradient fixture; not all-state derivative rerun',
        features_recomputed_from_saved_scores_exact=True,both_head_scores_selection_exact=True,
        summary_alignment_projection_recomputed_exact=True,passes=report['passes'],failed=report['failed'],**totals)
    (root/'output_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
