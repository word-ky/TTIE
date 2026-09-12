"""Verify completed fresh T014 results from their actual F-drive artifact root.

Run from the frozen StageB release. This only reads existing fresh results and
recomputes their offline metrics; no optimization, fitting, or new image split.
"""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
sys.path.insert(0,str(Path.cwd()))
import torch
from ttie.energy_model import features,load_energy
from ttie.sobolev_receipt import verify_receipt
from ttie.sobolev_metrics import stage_b,summarize_trajectories,trajectory_diagnostics
from ttie.sobolev_io import PRIMARY_METHOD,CONTROL
from ttie.stop_receipt import sha
from ttie.stop_trajectory import feature_vectors
from ttie.natural import load_image,degrade
from ttie.restoration_metrics import evaluate_outputs


def main():
    p=argparse.ArgumentParser();p.add_argument('audit',type=Path);p.add_argument('--images',type=Path,required=True);args=p.parse_args()
    root=args.audit;read=lambda p:json.loads(p.read_text());torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    config=read(root/'config.json');final=read(root/'final_checks.json');assert config['stage']==final['stage']=='B'
    receipt=verify_receipt(Path('research_log/T014_energy_receipt.json'),Path('research_log/T014_energy.pt'),
        Path('research_log/T014_value_only.pt'),Path('research_log/T014_source_manifest.json'),config['model_identity'],config['frozen_receipt'])
    manifest=read(Path('research_log/T014_manifest.json'));proof=read(Path('research_log/T014_git_freeze_verification.json'))
    assert config['manifest']==manifest and config['manifest_sha256']==sha(Path('research_log/T014_manifest.json'))
    assert manifest['energy_receipt_git_verification']==proof and proof['git_blob_verified'] and proof['both_checkpoint_blobs_verified']
    assert proof['receipt_sha256']==sha(Path('research_log/T014_energy_receipt.json'))
    assert config['source_sha']==receipt['source_sha'] and len(manifest['images'])==40 and len(manifest['excluded_prior_ids'])==608
    assert final['training_inputs']==final['training_states']==0
    head_files=dict(sobolev_primary=Path('research_log/T014_energy.pt'),value_only_control=Path('research_log/T014_value_only.pt'))
    hashes={n:sha(path) for n,path in head_files.items()};assert hashes==final['energy_sha256']
    assert hashes['sobolev_primary']==receipt['energy_sha256'] and hashes['value_only_control']==receipt['value_only_sha256']
    gpu_heads={n:load_energy(path).cuda() for n,path in head_files.items()}
    assert sha(Path(config['model_identity']['path']))==config['frozen_receipt']['model_identity']['sha256']
    assert sha(Path('research_log/remote_runs/20260912-071126-ttie-t006-a6000/artifacts/audit/prototypes.pt'))==config['frozen_receipt']['prototype_identity']['sha256']
    assert read(Path('research_log/T007_joint_calibration.json'))==config['frozen_receipt']
    totals=dict(hashed_files=0,large_image_bytes=0,evaluation_inputs=0,energy_checkpoints=0,energy_updates=0,
                semantic_checkpoints=0,inactive_region_checks=0,no_active_inputs=0)
    def files(directory,records):
        for name,r in records.items():
            path=directory/name;assert path.stat().st_size==r['bytes'] and sha(path)==r['sha256'];totals['hashed_files']+=1
            if name in ('checkpoint_images.pt','outputs.pt'):totals['large_image_bytes']+=r['bytes']
    def obj(gate):
        return SimpleNamespace(active=torch.tensor(gate['active']),winner=torch.tensor(gate['winner']),
            evidence=torch.tensor(gate['evidence'],dtype=torch.float64),calibration=config['frozen_receipt']['calibration'])
    def inactive(images,image,gate):
        h,w=image.shape[-2:]
        for q,active in enumerate(gate['active']):
            if active:continue
            y=slice(0,h//2) if q<2 else slice(h//2,h);x=slice(0,w//2) if q%2==0 else slice(w//2,w)
            assert torch.equal(images[...,y,x],image[...,y,x].expand(len(images),-1,-1,-1,-1));totals['inactive_region_checks']+=len(images)
    entries=read(root/'artifact_manifest.json');rows=[]
    for e in manifest['images']:assert sha(args.images/e['filename'])==e['sha256']
    for e in entries:
        d=root/e['directory'];clean=load_image(args.images/f"{e['image_id']:012d}.jpg");image=degrade(clean,e['condition'])
        assert e['files']==read(d/'label_free_receipt.json');files(d,e['files']['episode'])
        outputs=torch.load(d/'outputs.pt',weights_only=True);dec=read(d/'decisions.json')
        results={m:dict(**r,diagnostics=dec['methods'][m]) for m,r in outputs.items()};case=read(d/'metrics.json')
        actual=evaluate_outputs(results,clean,image_id=e['image_id'],condition=e['condition'])
        for a,b in zip(actual,case):assert all(a[k]==b[k] for k in a)
        ts={};ds={}
        for method,records in e['files'].items():
            if method=='episode':continue
            folder=d/method;files(folder,records)
            small=torch.load(folder/'trajectory.pt',weights_only=True);images=torch.load(folder/'checkpoint_images.pt',weights_only=True)
            info=read(folder/'decisions.json');gate=info['gate'];diag=info['trajectory'];n=len(images)
            assert all(torch.isfinite(v).all() for v in small.values()) and torch.isfinite(images).all()
            assert (images>=0).all() and (images<=1).all()
            assert n==len(small['states'])==len(diag['gradient_vectors'])+1==len(diag['projections'])+1
            assert torch.equal(small['states'][0],torch.zeros_like(small['states'][0]))
            if method=='semantic':
                f=feature_vectors(gate,small['scores'],small['grids'],diag,config['frozen_receipt']['calibration'])
                assert torch.equal(f,small['features']);totals['semantic_checkpoints']+=n
                assert torch.equal(outputs['fixed_step_source']['image'],images[min(16,n-1)])
                assert torch.equal(outputs['region2_ttt_projected']['image'],images[-1]);continue
            assert n==(41 if any(gate['active']) else 1)
            f=torch.stack([features(obj(gate),s,g) for s,g in zip(small['scores'],small['grids'])]);assert torch.equal(f,small['features'])
            head=gpu_heads['value_only_control' if method==CONTROL else 'sobolev_primary']
            with torch.no_grad():energies=[float(head(row.cuda()).squeeze()) for row in f]
            assert energies==diag['loss_trajectory']==info['selection']['scores']
            index=min(range(n),key=lambda i:(energies[i],i));assert index==info['selection']['selected_step']
            assert torch.equal(outputs[method]['image'],images[index]) and torch.equal(outputs[method]['raw'],small['states'][index])
            assert torch.isfinite(torch.tensor(diag['gradient_vectors'])).all()
            lo=torch.tensor(diag['action_box']['lower']);hi=torch.tensor(diag['action_box']['upper'])
            assert (small['grids']>=lo-1e-7).all() and (small['grids']<=hi+1e-7).all()
            ts[method]=dict(**small,diagnostics=diag);ds[method]=info['selection']
            totals['energy_checkpoints']+=n;totals['energy_updates']+=n-1
            if method in (CONTROL,PRIMARY_METHOD):inactive(images,image,gate)
            if method==PRIMARY_METHOD:
                values=[float((v-clean).square().mean()) for v in images]
                assert values==[r['mse'] for r in read(d/'checkpoint_metrics.json')]
                oracle=min(range(n),key=lambda i:(values[i],i));assert read(d/'oracle_diagnostic.json')['selected_step']==oracle
                assert case[-1]['mse']==values[oracle]
                if not any(gate['active']):totals['no_active_inputs']+=1
        assert trajectory_diagnostics(ts,ds)==e['trajectory_diagnostics']
        assert not (d/'alignments.json').exists();rows.extend(case);totals['evaluation_inputs']+=1
        if totals['evaluation_inputs']%20==0:print('verified fresh inputs',totals['evaluation_inputs'],flush=True)
    assert rows==read(root/'metrics.json') and read(root/'alignments.json')==[]
    computed=stage_b(rows);computed['trajectory_distributions']=summarize_trajectories(entries);report=read(root/'summary.json')
    assert json.loads(json.dumps(computed))==report
    assert totals['evaluation_inputs']==final['calibration_or_evaluation_inputs']==240
    assert totals['energy_checkpoints']==final['energy_checkpoints'] and len(report['criteria'])==12
    result=dict(task='T014',stage='B',all_hashes_verified=True,all_stored_pixel_mse_exact=True,both_head_scores_selection_exact=True,
        features_summary_projection_exact=True,frozen_receipt_models_assets_verified=True,no_refitting=True,
        offset_report_only=True,qualified=report['qualified'],failed=report['failed'],head_sha256=hashes,**totals)
    (root/'output_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
