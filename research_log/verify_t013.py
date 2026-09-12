"""Verify completed T013 Stage-A artifacts from the tested release root."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
sys.path.insert(0,str(Path.cwd()))
import torch
from ttie.energy_model import features,load_energy,SCHEMA,RECIPE
from ttie.energy_bank import BANK
from ttie.energy_receipt import code_hashes
from ttie.energy_metrics import stage_a,PRIMARY_METHOD
from ttie.stop_receipt import sha
from ttie.stop_trajectory import feature_vectors
from ttie.natural import load_image,degrade
from ttie.restoration_metrics import evaluate_outputs


def main():
    p=argparse.ArgumentParser();p.add_argument('audit',type=Path);p.add_argument('--images',type=Path,required=True);args=p.parse_args()
    root=args.audit;read=lambda p:json.loads(p.read_text());torch.set_num_threads(1)
    config=read(root/'config.json');receipt=read(root/'T013_energy_receipt.json');final=read(root/'final_checks.json')
    assert receipt['source_code_sha256']==code_hashes() and receipt['feature_schema']==SCHEMA and receipt['recipe']==RECIPE and receipt['state_bank']==BANK
    assert sha(root/'energy.pt')==receipt['energy_sha256']==final['energy_sha256']
    head=load_energy(root/'energy.pt');assert head.normalization()==receipt['normalization']
    gpu_head=load_energy(root/'energy.pt').cuda()
    totals=dict(hashed_files=0,large_image_bytes=0,training_inputs=0,training_states=0,calibration_inputs=0,
                energy_checkpoints=0,energy_updates=0,semantic_checkpoints=0,inactive_region_checks=0,no_active_inputs=0)
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
    train_entries=read(root/'training_manifest.json');xs=[];ys=[]
    for e in train_entries:
        d=root/e['directory'];verify_files(d,e['files'])
        small=torch.load(d/'bank.pt',weights_only=True);images=torch.load(d/'bank_images.pt',weights_only=True)
        dec=read(d/'bank_decisions.json');gate=dec['gate'];expected=BANK['order'] if any(gate['active']) else ['identity']
        assert dec['names']==expected and len(images)==e['states']==len(expected)
        assert torch.isfinite(images).all() and (images>=0).all() and (images<=1).all()
        assert all(torch.isfinite(t).all() for t in small.values())
        f=torch.stack([features(obj(gate),s,g) for s,g in zip(small['scores'],small['grids'])]);assert torch.equal(f,small['features'])
        clean=load_image(args.images/f"{e['image_id']:012d}.jpg");image=degrade(clean,e['condition'])
        assert torch.equal(images[0],image);inactive_checks(images,image,gate)
        targets=read(d/'targets.json');values=[float((v-clean).square().mean()) for v in images]
        assert values==[t['mse'] for t in targets]
        xs.append(small['features']);ys.append(torch.tensor(values,dtype=torch.float64))
        totals['training_inputs']+=1;totals['training_states']+=len(images)
    x=torch.cat(xs).double();y=(torch.cat(ys)+1e-6).log()
    scale=x.std(0,unbiased=False);scale=torch.where(scale==0,torch.ones_like(scale),scale)
    yscale=y.std(unbiased=False);yscale=torch.where(yscale==0,torch.ones_like(yscale),yscale)
    assert torch.equal(head.x_mean,x.mean(0).float()) and torch.equal(head.x_scale,scale.float())
    assert torch.equal(head.y_mean,y.mean().float()) and torch.equal(head.y_scale,yscale.float())
    train=read(root/'training.json');assert train['rows']==len(x) and train['images']==80 and len(train['history'])==100
    entries=read(root/'artifact_manifest.json');rows=[];alignments=[]
    for e in entries:
        d=root/e['directory'];clean=load_image(args.images/f"{e['image_id']:012d}.jpg");image=degrade(clean,e['condition'])
        assert e['files']==read(d/'label_free_receipt.json');verify_files(d,e['files']['episode'])
        output=torch.load(d/'outputs.pt',weights_only=True);decisions=read(d/'decisions.json')
        results={m:dict(**r,diagnostics=decisions['methods'][m]) for m,r in output.items()}
        case=read(d/'metrics.json');measured=evaluate_outputs(results,clean,image_id=e['image_id'],condition=e['condition'])
        for actual,saved in zip(measured,case):
            assert all(actual[k]==saved[k] for k in actual)
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
                assert torch.equal(output['region2_ttt_projected']['image'],images[-1])
                continue
            expected=41 if any(gate['active']) else 1;assert n==expected
            f=torch.stack([features(obj(gate),s,g) for s,g in zip(small['scores'],small['grids'])]);assert torch.equal(f,small['features'])
            with torch.no_grad():energies=[float(gpu_head(row.cuda()).squeeze()) for row in f]
            assert energies==diag['loss_trajectory']==info['selection']['scores']
            index=min(range(n),key=lambda i:(energies[i],i));assert index==info['selection']['selected_step']
            assert torch.equal(output[method]['image'],images[index]) and torch.equal(output[method]['raw'],small['states'][index])
            assert torch.isfinite(torch.tensor(diag['gradient_vectors'])).all()
            lo=torch.tensor(diag['action_box']['lower']);hi=torch.tensor(diag['action_box']['upper'])
            assert (small['grids']>=lo-1e-7).all() and (small['grids']<=hi+1e-7).all()
            totals['energy_checkpoints']+=n;totals['energy_updates']+=n-1
            if method==PRIMARY_METHOD:
                inactive_checks(images,image,gate)
                if not any(gate['active']):totals['no_active_inputs']+=1
                targets=read(d/'checkpoint_metrics.json');values=[float((v-clean).square().mean()) for v in images]
                assert values==[r['mse'] for r in targets]
                oracle=min(range(n),key=lambda i:(values[i],i));assert read(d/'oracle_diagnostic.json')['selected_step']==oracle
                assert case[-1]['mse']==values[oracle]
                if any(gate['active']) and e['condition']!='clean':
                    a=read(d/'alignment.json');g=torch.tensor(a['energy_gradient'],dtype=torch.float64);r=torch.tensor(a['reference_gradient'],dtype=torch.float64)
                    assert torch.equal(g,torch.tensor(diag['gradient_vectors'][0],dtype=torch.float64).flatten())
                    denom=g.norm()*r.norm();cos=float(g@r/denom) if float(denom)>0 else 0.;assert cos==a['cosine'];alignments.append(a)
        rows.extend(case);totals['calibration_inputs']+=1
    assert rows==read(root/'metrics.json') and alignments==read(root/'alignments.json')
    report=read(root/'summary.json');assert stage_a(rows,alignments)==report==receipt['stage_a']
    assert totals['training_inputs']==400 and totals['calibration_inputs']==100
    assert totals['training_states']==final['training_states'] and totals['energy_checkpoints']==final['energy_checkpoints']
    result=dict(task='T013',stage='A',all_hashes_verified=True,all_stored_mse_recomputed_exact=True,
        train_only_normalization_exact=True,features_recomputed_exact=True,energy_scores_selection_exact=True,
        summary_alignment_recomputed_exact=True,passes=report['passes'],failed=report['failed'],**totals)
    (root/'output_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
