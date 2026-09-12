"""Post-run T015 audit. Run from frozen release with full F-drive image packs."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
sys.path.insert(0,str(Path.cwd()))
import torch
from ttie.energy_model import features,load_energy
from ttie.sobolev_receipt import verify_receipt
from ttie.sobolev_metrics import trajectory_diagnostics
from ttie.routing.core import BASES,PRIMARY,ORACLE,route
from ttie.routing.metrics import summarize,routing_diagnostics,summarize_trajectories
from ttie.stop_receipt import sha
from ttie.stop_trajectory import feature_vectors
from ttie.natural import load_image,degrade
from ttie.restoration_metrics import evaluate_outputs


def main():
    p=argparse.ArgumentParser();p.add_argument('audit',type=Path);p.add_argument('--images',type=Path,required=True);a=p.parse_args()
    root=a.audit;read=lambda f:json.loads(f.read_text());torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    config=read(root/'config.json');final=read(root/'final_checks.json')
    receipt=verify_receipt(Path('research_log/T014_energy_receipt.json'),Path('research_log/T014_energy.pt'),
        Path('research_log/T014_value_only.pt'),Path('research_log/T014_source_manifest.json'),config['model_identity'],config['frozen_receipt'])
    manifest=read(Path('research_log/T015_manifest.json'));head=load_energy(Path('research_log/T014_energy.pt')).cuda()
    assert config['manifest']==manifest and config['manifest_sha256']==final['manifest_sha256']==sha(Path('research_log/T015_manifest.json'))
    assert len(manifest['images'])==40 and len(manifest['excluded_prior_ids'])==648
    assert config['source_sha']==final['source_sha']=='c4e58e5ad64bfce0bea72561997db8007e12b510'
    assert config['frozen_t014_source_sha']==receipt['source_sha']
    assert config['frozen_t014_receipt_sha256']==manifest['frozen_t014_receipt_sha256']==sha(Path('research_log/T014_energy_receipt.json'))
    assert config['training_inputs']==final['training_inputs']==0
    assert config['frozen_energy_sha256']==final['energy_sha256']==receipt['energy_sha256']==manifest['frozen_sobolev_sha256']==sha(Path('research_log/T014_energy.pt'))
    assert config['frozen_control_sha256']==receipt['value_only_sha256']==sha(Path('research_log/T014_value_only.pt'))
    for filename,digest in config['source_code_sha256'].items():assert sha(Path(filename))==digest
    assert sha(Path(config['model_identity']['path']))==config['frozen_receipt']['model_identity']['sha256']
    assert sha(Path('research_log/remote_runs/20260912-071126-ttie-t006-a6000/artifacts/audit/prototypes.pt'))==config['frozen_receipt']['prototype_identity']['sha256']
    totals=dict(hashed_files=0,large_image_bytes=0,evaluation_inputs=0,energy_checkpoints=0,energy_updates=0,
        semantic_checkpoints=0,inactive_region_checks=0,no_active_inputs=0)
    def files(directory,records):
        for name,r in records.items():
            path=directory/name;assert path.stat().st_size==r['bytes'] and sha(path)==r['sha256'];totals['hashed_files']+=1
            if name in ('checkpoint_images.pt','outputs.pt'):totals['large_image_bytes']+=r['bytes']
    def objective(gate):
        return SimpleNamespace(active=torch.tensor(gate['active']),winner=torch.tensor(gate['winner']),
            evidence=torch.tensor(gate['evidence'],dtype=torch.float64),calibration=config['frozen_receipt']['calibration'])
    entries=read(root/'artifact_manifest.json');rows=[]
    expected=[(e['image_id'],c) for e in manifest['images'] for c in ('clean','homogeneous_dark','homogeneous_bright','left_right','quadrants','offset_left_right_40')]
    assert [(e['image_id'],e['condition']) for e in entries]==expected
    for e in manifest['images']:assert sha(a.images/e['filename'])==e['sha256']
    for e in entries:
        d=root/e['directory'];clean=load_image(a.images/f"{e['image_id']:012d}.jpg");image=degrade(clean,e['condition'])
        assert e['files']==read(d/'label_free_receipt.json');files(d,e['files']['episode'])
        outputs=torch.load(d/'outputs.pt',weights_only=True);dec=read(d/'decisions.json');route_saved=read(d/'routing.json')
        results={m:dict(**r,diagnostics=dec['methods'][m]) for m,r in outputs.items()}
        actual=evaluate_outputs(results,clean,image_id=e['image_id'],condition=e['condition']);by_method={r['method']:r for r in actual}
        oracle_basis=min(BASES,key=lambda m:by_method[m]['mse']);actual.append(dict(by_method[oracle_basis],method=ORACLE))
        case=read(d/'metrics.json');assert len(actual)==len(case)==10
        for x,y in zip(actual,case):assert all(x[k]==y[k] for k in x)
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
                assert torch.equal(feature_vectors(gate,small['scores'],small['grids'],diag,config['frozen_receipt']['calibration']),small['features'])
                assert torch.equal(outputs['fixed_step_source']['image'],images[min(16,n-1)])
                assert torch.equal(outputs['region2_ttt_projected']['image'],images[-1]);totals['semantic_checkpoints']+=n;continue
            assert method in BASES and n==(41 if any(gate['active']) else 1)
            f=torch.stack([features(objective(gate),s,g) for s,g in zip(small['scores'],small['grids'])]);assert torch.equal(f,small['features'])
            with torch.no_grad():energies=[float(head(row.cuda()).squeeze()) for row in f]
            assert energies==diag['loss_trajectory']==info['selection']['scores']
            index=min(range(n),key=lambda i:(energies[i],i));assert index==info['selection']['selected_step']
            assert info['selection']==dec['selections'][method]==e['selections'][method]
            assert torch.equal(outputs[method]['image'],images[index]) and torch.equal(outputs[method]['raw'],small['states'][index])
            assert torch.isfinite(torch.tensor(diag['gradient_vectors'])).all()
            lo=torch.tensor(diag['action_box']['lower']);hi=torch.tensor(diag['action_box']['upper'])
            assert (small['grids']>=lo-1e-7).all() and (small['grids']<=hi+1e-7).all()
            if method==BASES[2]:
                h,w=image.shape[-2:]
                for q,active in enumerate(gate['active']):
                    if active:continue
                    y=slice(0,h//2) if q<2 else slice(h//2,h);x=slice(0,w//2) if q%2==0 else slice(w//2,w)
                    assert torch.equal(images[...,y,x],image[...,y,x].expand(n,-1,-1,-1,-1));totals['inactive_region_checks']+=n
                totals['no_active_inputs']+=not any(gate['active'])
            ts[method]=dict(**small,diagnostics=diag);ds[method]=info['selection']
            totals['energy_checkpoints']+=n;totals['energy_updates']+=n-1
        r=route(*(ds[m]['scores'][ds[m]['selected_step']] for m in BASES))
        assert r==route_saved==e['routing']==dec['selections']['routing']==dec['methods'][PRIMARY]['routing']
        for k in ('image','raw','grid'):assert torch.equal(outputs[PRIMARY][k],outputs[r['selected_basis']][k])
        oracle=read(d/'oracle_diagnostic.json');assert oracle==e['oracle'] and oracle['reference_only']
        assert oracle['selected_basis']==oracle_basis and oracle['basis_mse']=={m:by_method[m]['mse'] for m in BASES}
        assert oracle['routed_mse']==by_method[PRIMARY]['mse'] and oracle['oracle_mse']==by_method[oracle_basis]['mse']
        assert oracle['mse_regret']==oracle['routed_mse']-oracle['oracle_mse']
        assert oracle['disagreement']==(r['selected_basis']!=oracle_basis)
        assert ORACLE not in outputs and trajectory_diagnostics(ts,ds)==e['trajectory_diagnostics']
        rows.extend(case);totals['evaluation_inputs']+=1
        if totals['evaluation_inputs']%20==0:print('verified T015 inputs',totals['evaluation_inputs'],flush=True)
    assert rows==read(root/'metrics.json')
    report=summarize(rows);report['trajectory_distributions']=summarize_trajectories(entries)
    assert json.loads(json.dumps(report))==read(root/'summary.json')
    assert routing_diagnostics(entries)==read(root/'routing_diagnostics.json')
    assert totals['evaluation_inputs']==final['evaluation_inputs']==240
    assert totals['energy_checkpoints']==final['energy_checkpoints'] and len(report['criteria'])==10
    result=dict(task='T015',all_hashes_verified=True,all_stored_pixel_mse_exact=True,
        frozen_scores_checkpoints_routes_oracles_exact=True,features_projections_summary_exact=True,
        frozen_receipt_assets_manifest_verified=True,no_refitting=True,offset_is_primary=True,
        qualified=report['qualified'],failed=report['failed'],**totals)
    (root/'output_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
