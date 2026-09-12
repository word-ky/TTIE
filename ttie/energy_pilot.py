"""T013 fixed source bank/training, calibration and separately authorized fresh run."""
import argparse
import json
from pathlib import Path
import time
import torch
from .clip_signal import FrozenCLIP
from .learned_prototypes import Prototypes
from .semantic_ttt import SemanticScorer
from .natural import load_image,degrade
from .energy_bank import state_bank,BANK
from .energy_model import train_energy,save_energy,load_energy,SCHEMA,RECIPE
from .energy_io import save_bank,bank_targets,run_label_free,save_episode,evaluate_episode
from .energy_metrics import stage_a,stage_b_energy,markdown,PRIMARY,STRESS,PRIMARY_METHOD,ORACLE
from .energy_receipt import create_receipt,verify_receipt
from .stop_receipt import sha
from .residual_pilot import write,panel


def train_source(manifest,images,output,scorer,receipt,device,*,semantic_steps=40):
    entries=[];xs=[];ys=[]
    for entry in manifest['images']:
        if entry['split']!='train_t013_energy':continue
        path=images/entry['filename'];assert sha(path)==entry['sha256'];clean=load_image(path)
        for condition in PRIMARY:
            start=time.monotonic();image=degrade(clean.to(device),condition)
            bank=state_bank(image,scorer,receipt,semantic_steps=semantic_steps)
            directory=output/'training_bank'/f'{len(entries):03d}';files=save_bank(directory,bank)
            targets=bank_targets(directory,bank,clean)
            xs.append(bank['features']);ys.append(torch.tensor([r['mse'] for r in targets],dtype=torch.float64))
            entries.append(dict(image_id=entry['image_id'],condition=condition,directory=directory.relative_to(output).as_posix(),
                states=len(targets),files=files,seconds=time.monotonic()-start))
            write(output/'training_manifest.json',entries)
            print('T013 bank',len(entries),entry['image_id'],condition,'states',len(targets),flush=True)
            del bank,image
    x=torch.cat(xs);y=torch.cat(ys);head,history=train_energy(x,y);save_energy(head,output/'energy.pt')
    training=dict(recipe=RECIPE,normalization=head.normalization(),rows=len(x),images=len({e['image_id'] for e in entries}),history=history)
    write(output/'training.json',training)
    print('T013 trained energy',len(x),'rows',len(history),'epochs',flush=True)
    return head,entries


def calibrate(manifest,images,output,scorer,receipt,head,device,*,stage='A',max_steps=40):
    entries=[];rows=[];alignments=[]
    selected=[e for e in manifest['images'] if stage=='B' or e['split']=='calibration_t013_energy']
    representative=selected[0]['image_id'];conditions=PRIMARY if stage=='A' else (*PRIMARY,STRESS)
    (output/'figures').mkdir(exist_ok=True)
    for entry in selected:
        path=images/entry['filename'];assert sha(path)==entry['sha256'];clean=load_image(path)
        for condition in conditions:
            start=time.monotonic();image=degrade(clean.to(device),condition)
            results,old,trajectories,decisions=run_label_free(image,scorer,receipt,head,max_steps=max_steps)
            directory=output/'episodes'/f'{len(entries):03d}';files=save_episode(directory,results,old,trajectories,decisions)
            case,diagnostic,oracle=evaluate_episode(directory,results,trajectories,clean,image,image_id=entry['image_id'],condition=condition,diagnose=stage=='A')
            rows.extend(case)
            if diagnostic is not None:alignments.append(diagnostic)
            entries.append(dict(image_id=entry['image_id'],condition=condition,directory=directory.relative_to(output).as_posix(),files=files,
                semantic_checkpoints=len(old['images']),energy_checkpoints={m:len(t['images']) for m,t in trajectories.items()},
                selections=decisions,seconds=time.monotonic()-start))
            write(output/'artifact_manifest.json',entries);write(output/'metrics.json',rows);write(output/'alignments.json',alignments)
            if entry['image_id']==representative:
                results[ORACLE]=oracle;panel(results,clean,output/'figures'/(condition+'.png'),f'T013 {stage} fixed ID {representative}: {condition}')
            print('T013',stage,'finalized',len(entries),entry['image_id'],condition,'seconds',round(time.monotonic()-start,2),flush=True)
            del results,old,trajectories,image
    report=stage_a(rows,alignments) if stage=='A' else stage_b_energy(rows)
    write(output/'summary.json',report);(output/'summary.md').write_text(markdown(report))
    return report,entries


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=('A','B'),required=True)
    for key in ('manifest','source-manifest','images','model-identity','prototypes','receipt','output','t006-images'):
        parser.add_argument('--'+key,required=True,type=Path)
    parser.add_argument('--source-sha',required=True);parser.add_argument('--energy',type=Path);parser.add_argument('--energy-receipt',type=Path)
    parser.add_argument('--device',default='cuda:0');args=parser.parse_args()
    torch.manual_seed(7);torch.set_num_threads(1);torch.use_deterministic_algorithms(False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    receipt=json.loads(args.receipt.read_text());identity=json.loads(args.model_identity.read_text());frozen=None;head=None
    if args.stage=='B':
        frozen=verify_receipt(args.energy_receipt,args.energy,args.source_manifest,identity,receipt)
        assert frozen['source_sha']==args.source_sha;head=load_energy(args.energy)
    manifest=json.loads(args.manifest.read_text())
    if frozen:
        verified=manifest['energy_receipt_git_verification']
        assert verified['git_blob_verified'] and verified['receipt_sha256']==sha(args.energy_receipt)
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
    config=dict(stage=args.stage,source_sha=args.source_sha,manifest=manifest,manifest_sha256=sha(args.manifest),source_manifest_sha256=sha(args.source_manifest),
        frozen_receipt=receipt,model_identity=identity,schema=SCHEMA,recipe=RECIPE,state_bank=BANK,fixed_step_source=16,seed=7,tf32=False,
        strict_cuda_determinism=False,energy_inference_device=args.device,energy_receipt_sha256=sha(args.energy_receipt) if frozen else None)
    write(args.output/'config.json',config);training_entries=[]
    if args.stage=='A':head,training_entries=train_source(manifest,args.images,args.output,scorer,receipt,args.device)
    energy_file=args.output/'energy.pt' if args.stage=='A' else args.energy;energy_hash=sha(energy_file)
    report,entries=calibrate(manifest,args.images,args.output,scorer,receipt,head,args.device,stage=args.stage)
    assert sha(energy_file)==energy_hash and all(not p.requires_grad and p.grad is None for p in head.parameters())
    assert all(not p.requires_grad and p.grad is None for p in scorer.parameters()) and torch.equal(prototypes.vectors,saved['raw'])
    if args.stage=='A':write(args.output/'T013_energy_receipt.json',create_receipt(energy_file,args.source_manifest,args.source_sha,report,identity,receipt))
    write(args.output/'final_checks.json',dict(stage=args.stage,training_inputs=len(training_entries),training_states=sum(e['states'] for e in training_entries),
        calibration_or_evaluation_inputs=len(entries),energy_checkpoints=sum(sum(e['energy_checkpoints'].values()) for e in entries),
        frozen_assets_unchanged=True,energy_sha256=energy_hash,energy_frozen=True))
    print('T013 Stage',args.stage,'passes',report.get('passes',report.get('qualified')),'failed',report['failed'],flush=True)
    print('STOP: no new manifest or next task in this process.',flush=True)


if __name__=='__main__':main()
