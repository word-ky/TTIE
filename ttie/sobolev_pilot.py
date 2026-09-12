"""T014 fixed source derivative training and separately authorized calibration/evaluation."""
import argparse
import json
from pathlib import Path
import time
import torch
from .clip_signal import FrozenCLIP
from .learned_prototypes import Prototypes
from .semantic_ttt import SemanticScorer
from .natural import load_image,degrade
from .energy_model import save_energy,load_energy,SCHEMA,RECIPE
from .energy_bank import BANK
from .energy_io import save_bank,bank_targets,save_episode,hashes
from .sobolev_source import source_bank,source_derivatives,JACOBIAN
from .sobolev_train import train_pair,training_statistics,DERIVATIVE_LOSS
from .sobolev_io import run_label_free,evaluate_episode,ORACLE
from .sobolev_metrics import stage_a,stage_b,markdown,trajectory_diagnostics,summarize_trajectories
from .sobolev_receipt import create_receipt,verify_receipt
from .residual_metrics import PRIMARY,STRESS
from .stop_receipt import sha
from .residual_pilot import write,panel


def train_source(manifest,images,output,scorer,receipt,device,*,semantic_steps=40):
    entries=[];xs=[];ys=[];derivatives=[]
    for entry in manifest['images']:
        if entry['split']!='train_t014_sobolev':continue
        path=images/entry['filename'];assert sha(path)==entry['sha256'];clean=load_image(path)
        for condition in PRIMARY:
            start=time.monotonic();image=degrade(clean.to(device),condition)
            bank=source_bank(image,scorer,receipt,semantic_steps=semantic_steps)
            directory=output/'training_bank'/f'{len(entries):03d}';files=save_bank(directory,bank)
            # Freeze all label-free bank pixels/features/states before source targets.
            targets=bank_targets(directory,bank,clean);records=source_derivatives(image,clean,scorer,receipt,bank)
            torch.save(records,directory/'source_derivatives.pt')
            write(directory/'derivative_convention.json',dict(convention=JACOBIAN,
                inherited_pixel_max_abs_differences=bank['inherited_pixel_max_abs_differences']))
            derivative_files=hashes(directory,('source_derivatives.pt','targets.json','derivative_convention.json'))
            xs.append(bank['features']);ys.append(torch.tensor([r['mse'] for r in targets],dtype=torch.float64));derivatives.append(records)
            entries.append(dict(image_id=entry['image_id'],condition=condition,directory=directory.relative_to(output).as_posix(),
                states=len(targets),files=files,source_supervision_files=derivative_files,direction_rows=int(records['direction_mask'].sum()),
                seconds=time.monotonic()-start))
            write(output/'training_manifest.json',entries)
            print('T014 bank',len(entries),entry['image_id'],condition,'states',len(targets),'seconds',round(time.monotonic()-start,2),flush=True)
            del bank,image
    x=torch.cat(xs);y=torch.cat(ys);records={k:torch.cat([r[k] for r in derivatives]) for k in derivatives[0]}
    heads,histories=train_pair(x,y,records)
    for name,head in heads.items():save_energy(head,output/(name+'.pt'))
    training=dict(recipe=RECIPE,derivative_loss=DERIVATIVE_LOSS,jacobian_convention=JACOBIAN,normalization=heads['sobolev_primary'].normalization(),
        rows=len(x),images=len({e['image_id'] for e in entries}),history=histories,
        final_train_statistics={name:training_statistics(head,x,y,records) for name,head in heads.items()})
    write(output/'training.json',training)
    print('T014 trained both heads',len(x),'rows,100epochs',flush=True)
    return heads,entries


def calibrate(manifest,images,output,scorer,receipt,heads,device,*,stage='A',max_steps=40):
    entries=[];rows=[];alignments=[]
    selected=[e for e in manifest['images'] if stage=='B' or e['split']=='calibration_t014_sobolev']
    representative=selected[0]['image_id'];conditions=PRIMARY if stage=='A' else (*PRIMARY,STRESS)
    (output/'figures').mkdir(exist_ok=True)
    for entry in selected:
        path=images/entry['filename'];assert sha(path)==entry['sha256'];clean=load_image(path)
        for condition in conditions:
            start=time.monotonic();image=degrade(clean.to(device),condition)
            results,old,trajectories,decisions=run_label_free(image,scorer,receipt,heads,max_steps=max_steps)
            directory=output/'episodes'/f'{len(entries):03d}';files=save_episode(directory,results,old,trajectories,decisions)
            case,diagnostics,oracle=evaluate_episode(directory,results,trajectories,clean,image,image_id=entry['image_id'],condition=condition,diagnose=stage=='A')
            rows.extend(case);alignments.extend(diagnostics)
            entries.append(dict(image_id=entry['image_id'],condition=condition,directory=directory.relative_to(output).as_posix(),files=files,
                semantic_checkpoints=len(old['images']),energy_checkpoints={m:len(t['images']) for m,t in trajectories.items()},
                selections=decisions,trajectory_diagnostics=trajectory_diagnostics(trajectories,decisions),seconds=time.monotonic()-start))
            write(output/'artifact_manifest.json',entries);write(output/'metrics.json',rows);write(output/'alignments.json',alignments)
            if entry['image_id']==representative:
                results[ORACLE]=oracle;panel(results,clean,output/'figures'/(condition+'.png'),f'T014 {stage} fixed ID {representative}: {condition}')
            print('T014',stage,'finalized',len(entries),entry['image_id'],condition,'seconds',round(time.monotonic()-start,2),flush=True)
            del results,old,trajectories,image
    report=stage_a(rows,alignments) if stage=='A' else stage_b(rows)
    report['trajectory_distributions']=summarize_trajectories(entries)
    write(output/'summary.json',report);(output/'summary.md').write_text(markdown(report).replace('# T013 Stage','# T014 Stage',1))
    return report,entries


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=('A','B'),required=True)
    for key in ('manifest','source-manifest','images','model-identity','prototypes','receipt','output','t006-images'):
        parser.add_argument('--'+key,required=True,type=Path)
    parser.add_argument('--source-sha',required=True)
    for key in ('energy','control','energy-receipt'):parser.add_argument('--'+key,type=Path)
    parser.add_argument('--device',default='cuda:0');args=parser.parse_args()
    torch.manual_seed(7);torch.set_num_threads(1);torch.use_deterministic_algorithms(False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    receipt=json.loads(args.receipt.read_text());identity=json.loads(args.model_identity.read_text());frozen=None
    if args.stage=='B':
        frozen=verify_receipt(args.energy_receipt,args.energy,args.control,args.source_manifest,identity,receipt)
        assert frozen['source_sha']==args.source_sha
    manifest=json.loads(args.manifest.read_text())
    if frozen:
        verification=manifest['energy_receipt_git_verification']
        assert verification['git_blob_verified'] and verification['both_checkpoint_blobs_verified'] and verification['receipt_sha256']==sha(args.energy_receipt)
    args.output.mkdir(parents=True,exist_ok=True)
    assert sha(args.prototypes)==receipt['prototype_identity']['sha256'] and sha(identity['path'])==receipt['model_identity']['sha256']
    encoder=FrozenCLIP.from_checkpoint(identity['path'],args.device)
    saved=torch.load(args.prototypes,map_location=args.device,weights_only=True);prototypes=Prototypes(saved['raw']);scorer=SemanticScorer(encoder,prototypes)
    cal=load_image(args.t006_images/f"{receipt['image_ids'][0]:012d}.jpg").to(args.device)
    old=Path('research_log/remote_runs/20260912-071126-ttie-t006-a6000/artifacts/audit/calibration_scores.json')
    old_rows=[r for r in json.loads(old.read_text()) if r['image_id']==receipt['image_ids'][0] and r['view']!='full']
    expected=torch.tensor([[r['d_dark'],r['d_bright']] for r in old_rows])
    with torch.no_grad():actual=scorer(cal).cpu()
    assert torch.equal(actual,expected)
    write(args.output/'preflight.json',dict(original_calibration_bitwise_equal=True,energy_receipt_verified=bool(frozen)))
    config=dict(task='T014',stage=args.stage,source_sha=args.source_sha,manifest=manifest,manifest_sha256=sha(args.manifest),source_manifest_sha256=sha(args.source_manifest),
        frozen_receipt=receipt,model_identity=identity,schema=SCHEMA,recipe=RECIPE,state_bank=BANK,jacobian_convention=JACOBIAN,derivative_loss=DERIVATIVE_LOSS,
        fixed_step_source=16,seed=7,tf32=False,strict_cuda_determinism=False,energy_inference_device=args.device)
    write(args.output/'config.json',config);training_entries=[]
    if args.stage=='A':heads,training_entries=train_source(manifest,args.images,args.output,scorer,receipt,args.device)
    else:heads=dict(sobolev_primary=load_energy(args.energy),value_only_control=load_energy(args.control))
    head_files={n:args.output/(n+'.pt') for n in heads} if args.stage=='A' else dict(sobolev_primary=args.energy,value_only_control=args.control)
    head_hashes={n:sha(p) for n,p in head_files.items()}
    report,entries=calibrate(manifest,args.images,args.output,scorer,receipt,heads,args.device,stage=args.stage)
    assert head_hashes=={n:sha(p) for n,p in head_files.items()}
    assert all(not p.requires_grad and p.grad is None for head in heads.values() for p in head.parameters())
    assert all(not p.requires_grad and p.grad is None for p in scorer.parameters()) and torch.equal(prototypes.vectors,saved['raw'])
    if args.stage=='A':write(args.output/'T014_energy_receipt.json',create_receipt(head_files['sobolev_primary'],head_files['value_only_control'],
        args.source_manifest,args.source_sha,report,identity,receipt,args.output/'training_manifest.json'))
    write(args.output/'final_checks.json',dict(stage=args.stage,training_inputs=len(training_entries),training_states=sum(e['states'] for e in training_entries),
        calibration_or_evaluation_inputs=len(entries),energy_checkpoints=sum(sum(e['energy_checkpoints'].values()) for e in entries),
        frozen_assets_unchanged=True,energy_sha256=head_hashes,both_heads_frozen=True))
    print('T014 Stage',args.stage,'passes',report.get('passes',report.get('qualified')),'failed',report['failed'],flush=True)
    print('STOP: no new manifest or next task in this process.',flush=True)


if __name__=='__main__':main()
