"""T012 source training/calibration, then separately frozen fresh evaluation."""
import argparse
import json
from pathlib import Path
import time
import torch
from .clip_signal import FrozenCLIP
from .learned_prototypes import Prototypes
from .natural import load_image,degrade
from .semantic_ttt import SemanticScorer,FixedObjective
from .projected_ttt import run_candidate
from .stop_trajectory import capture,checkpoint,select_checkpoint,SCHEMA
from .stop_quality import train_head,save_head,load_head,RECIPE
from .stop_metrics import FIXED_STEPS,fixed_name,choose_fixed_step,stage_a,stage_b_stop,markdown,PRIMARY_METHOD,PRIMARY,STRESS
from .stop_receipt import sha,create_receipt,verify_receipt
from .stop_io import save_label_free,evaluate_reference,load_trajectory,renamed_row
from .residual_pilot import write,panel


def controls(image,scorer,receipt,trajectory,*,full):
    objective=FixedObjective(scorer,image,receipt)
    results={name:run_candidate(image,objective,name) for name in ('identity','region2_direct','region2_discrete_projected')}
    if full:
        for name in ('global_ttt_projected','bilinear2_ttt_projected'):results[name]=run_candidate(image,objective,name)
    results['region2_ttt_projected']=checkpoint(trajectory,40)
    if not full:results['region2_ttt_projected_1step']=checkpoint(trajectory,1)
    return results


def source_calibration(output,manifest,entries,source_sha,identity,receipt,images):
    xs=[];ys=[]
    for entry in entries:
        if entry['split']!='train_t012_stop':continue
        directory=output/entry['directory']
        xs.append(torch.load(directory/'trajectory.pt',weights_only=True)['features'])
        ys.append(torch.tensor([r['mse'] for r in json.loads((directory/'checkpoint_metrics.json').read_text())],dtype=torch.float64))
    x=torch.cat(xs);y=torch.cat(ys);head,history=train_head(x,y)
    head_file=output/'head.pt';save_head(head,head_file);head_hash=sha(head_file)
    write(output/'training.json',dict(recipe=RECIPE,rows=len(x),images=len({e['image_id'] for e in entries if e['split']=='train_t012_stop'}),normalization=head.normalization(),history=history))
    candidates=[];cal=[e for e in entries if e['split']=='calibration_t012_stop']
    for entry in cal:
        rows=json.loads((output/entry['directory']/'checkpoint_metrics.json').read_text())
        for step in FIXED_STEPS:candidates.append(renamed_row(rows[min(step,len(rows)-1)],fixed_name(step)))
    fixed=choose_fixed_step(candidates)
    write(output/'fixed_step_calibration.json',dict(**fixed,rows=candidates))
    rows=[];selections=[];first_cal_id=cal[0]['image_id']
    for entry in cal:
        directory=output/entry['directory'];trajectory=load_trajectory(directory)
        decision=select_checkpoint(trajectory,head)
        assert decision==select_checkpoint(trajectory,head)
        selected=checkpoint(trajectory,decision['selected_step']);fixed_result=checkpoint(trajectory,fixed['selected_step'])
        torch.save({PRIMARY_METHOD:selected['image'],'fixed_step_source':fixed_result['image']},directory/'selected_outputs.pt')
        write(directory/'selection.json',dict(learned=decision,fixed_step=fixed['selected_step'],head_sha256=head_hash))
        selection_receipt={n:dict(file=n,bytes=(directory/n).stat().st_size,sha256=sha(directory/n)) for n in ('selected_outputs.pt','selection.json')}
        write(directory/'selection_receipt.json',selection_receipt)
        # All learned decisions and selected outputs exist before indexing source targets.
        checkpoint_rows=json.loads((directory/'checkpoint_metrics.json').read_text())
        oracle_step=min(range(len(checkpoint_rows)),key=lambda i:checkpoint_rows[i]['mse'])
        case_rows=json.loads((directory/'metrics.json').read_text())
        for name,index in ((PRIMARY_METHOD,decision['selected_step']),('fixed_step_source',min(fixed['selected_step'],len(checkpoint_rows)-1)),('oracle_best_checkpoint',oracle_step)):
            case_rows.append(renamed_row(checkpoint_rows[index],name))
        write(directory/'calibration_metrics.json',case_rows);rows.extend(case_rows)
        selections.append(dict(image_id=entry['image_id'],condition=entry['condition'],learned=decision,
                               fixed_step=fixed['selected_step'],oracle_step=oracle_step,files=selection_receipt))
        if entry['image_id']==first_cal_id:
            saved=torch.load(directory/'outputs.pt',weights_only=True)
            saved.update({PRIMARY_METHOD:selected,'fixed_step_source':fixed_result,'oracle_best_checkpoint':checkpoint(trajectory,oracle_step)})
            clean=load_image(images/f"{entry['image_id']:012d}.jpg")
            panel(saved,clean,output/'figures'/(entry['condition']+'.png'),f"T012 source calibration fixed ID {first_cal_id}: {entry['condition']}")
    assert sha(head_file)==head_hash and all(p.grad is None for p in head.parameters())
    report=stage_a(rows);write(output/'calibration_metrics.json',rows);write(output/'calibration_selections.json',selections)
    write(output/'summary.json',report);(output/'summary.md').write_text(markdown(report))
    frozen=create_receipt(head_file,manifest,source_sha,report,fixed,identity,receipt)
    write(output/'T012_stopping_receipt.json',frozen)
    print('Stage A passes',report['passes'],'failed',report['failed'],'fixed step',fixed['selected_step'],flush=True)
    print('STOP: no fresh T012 manifest/scoring in this process. Commit a passing frozen receipt first.',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=('A','B'),required=True)
    for key in ('manifest','source-manifest','images','model-identity','prototypes','receipt','output','t006-images'):
        parser.add_argument('--'+key,required=True,type=Path)
    parser.add_argument('--source-sha',required=True);parser.add_argument('--head',type=Path)
    parser.add_argument('--stopping-receipt',type=Path);parser.add_argument('--device',default='cuda:0');args=parser.parse_args()
    torch.manual_seed(7);torch.set_num_threads(1);torch.use_deterministic_algorithms(False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    receipt=json.loads(args.receipt.read_text());identity=json.loads(args.model_identity.read_text())
    frozen=None;head=None
    if args.stage=='B':
        frozen=verify_receipt(args.stopping_receipt,args.head,args.source_manifest,identity,receipt)
        assert frozen['source_sha']==args.source_sha
        head=load_head(args.head)
    manifest=json.loads(args.manifest.read_text())
    if frozen:
        verified=manifest['stopping_receipt_git_verification']
        assert verified['git_blob_verified'] and verified['receipt_sha256']==sha(args.stopping_receipt)
    args.output.mkdir(parents=True,exist_ok=True);(args.output/'figures').mkdir(exist_ok=True)
    assert sha(args.prototypes)==receipt['prototype_identity']['sha256']
    assert sha(identity['path'])==receipt['model_identity']['sha256']
    encoder=FrozenCLIP.from_checkpoint(identity['path'],args.device)
    saved=torch.load(args.prototypes,map_location=args.device,weights_only=True)
    prototypes=Prototypes(saved['raw']);scorer=SemanticScorer(encoder,prototypes)
    cal=load_image(args.t006_images/f"{receipt['image_ids'][0]:012d}.jpg").to(args.device)
    old=Path('research_log/remote_runs/20260912-071126-ttie-t006-a6000/artifacts/audit/calibration_scores.json')
    old_rows=[r for r in json.loads(old.read_text()) if r['image_id']==receipt['image_ids'][0] and r['view']!='full']
    expected=torch.tensor([[r['d_dark'],r['d_bright']] for r in old_rows])
    with torch.no_grad():actual=scorer(cal).cpu()
    assert torch.equal(actual,expected)
    write(args.output/'preflight.json',dict(original_calibration_bitwise_equal=True,prototype_sha256=sha(args.prototypes),
          stopping_receipt_verified=bool(frozen)))
    conditions=PRIMARY if args.stage=='A' else (*PRIMARY,STRESS)
    config=dict(stage=args.stage,source_sha=args.source_sha,manifest=manifest,manifest_sha256=sha(args.manifest),
        source_manifest_sha256=sha(args.source_manifest),frozen_receipt=receipt,model_identity=identity,
        schema=SCHEMA,recipe=RECIPE,conditions=conditions,seed=7,tf32=False,strict_cuda_determinism=False,
        stopping_receipt_sha256=sha(args.stopping_receipt) if frozen else None,
        fixed_step=frozen['fixed_step_source']['selected_step'] if frozen else None,
        representative_image_id=next(r['image_id'] for r in manifest['images'] if r['split']=='calibration_t012_stop') if args.stage=='A' else manifest['images'][0]['image_id'])
    write(args.output/'config.json',config);entries=[];all_rows=[]
    for entry in manifest['images']:
        path=args.images/entry['filename'];assert sha(path)==entry['sha256'];clean=load_image(path)
        for condition in conditions:
            start=time.monotonic();index=len(entries);image=degrade(clean.to(args.device),condition)
            trajectory=capture(image,scorer,receipt)
            if args.stage=='A' and entry['split']=='train_t012_stop':
                obj=FixedObjective(scorer,image,receipt)
                results={'identity':run_candidate(image,obj,'identity'),'region2_ttt_projected':checkpoint(trajectory,40)}
            else:results=controls(image,scorer,receipt,trajectory,full=args.stage=='B')
            decision=None
            if args.stage=='B':
                decision=select_checkpoint(trajectory,head)
                assert decision==select_checkpoint(trajectory,head)
                results['fixed_step_source']=checkpoint(trajectory,config['fixed_step'])
                results[PRIMARY_METHOD]=checkpoint(trajectory,decision['selected_step'])
            directory=args.output/'episodes'/f'{index:03d}'
            files=save_label_free(directory,trajectory,results,decision)
            rows,checkpoint_rows=evaluate_reference(directory,trajectory,results,clean,image_id=entry['image_id'],condition=condition)
            if args.stage=='B':
                for r in rows:
                    if r['method']==PRIMARY_METHOD:r['selected_step']=decision['selected_step']
                oracle_step=min(range(len(checkpoint_rows)),key=lambda i:checkpoint_rows[i]['mse'])
                rows.append(renamed_row(checkpoint_rows[oracle_step],'oracle_best_checkpoint'))
                write(directory/'oracle_diagnostic.json',dict(selected_step=oracle_step,reference_only=True,checkpoint_file='checkpoint_images.pt'))
                write(directory/'metrics.json',rows)
                results['oracle_best_checkpoint']=checkpoint(trajectory,oracle_step)
                all_rows.extend(rows);write(args.output/'metrics.json',all_rows)
                if entry['image_id']==config['representative_image_id']:
                    panel(results,clean,args.output/'figures'/(condition+'.png'),f"T012 B fixed ID {entry['image_id']}: {condition}")
            entries.append(dict(index=index,image_id=entry['image_id'],split=entry['split'],condition=condition,
                directory=str(directory.relative_to(args.output)),checkpoints=len(trajectory['images']),seconds=time.monotonic()-start,files=files))
            write(args.output/'artifact_manifest.json',entries)
            print('Finalized T012',args.stage,entry['image_id'],condition,'inputs',len(entries),'checkpoints',len(trajectory['images']),
                  'seconds',round(time.monotonic()-start,2),flush=True)
            del trajectory,results,image
    assert all(not p.requires_grad and p.grad is None for p in scorer.parameters()) and torch.equal(prototypes.vectors,saved['raw'])
    if args.stage=='A':source_calibration(args.output,args.source_manifest,entries,args.source_sha,identity,receipt,args.images)
    else:
        assert sha(args.head)==frozen['head_sha256'] and all(not p.requires_grad and p.grad is None for p in head.parameters())
        report=stage_b_stop(all_rows);write(args.output/'summary.json',report);(args.output/'summary.md').write_text(markdown(report))
        print('Stage B qualified',report['qualified'],'failed',report['failed'],'STOP: no next task.',flush=True)
    write(args.output/'final_checks.json',dict(stage=args.stage,inputs=len(entries),checkpoints=sum(e['checkpoints'] for e in entries),
          frozen_assets_unchanged=True,head_frozen=True))


if __name__=='__main__':main()
