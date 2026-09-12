"""T010 source calibration / frozen pilot. Persist decisions before reference use."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
from PIL import Image,ImageDraw,ImageOps
import torch
from .clip_signal import FrozenCLIP
from .learned_prototypes import Prototypes
from .natural import load_image,degrade
from .semantic_ttt import SemanticScorer,FixedObjective
from .residual_ttt import Rhos,ResidualObjective,run_candidate,run_all
from .residual_metrics import PRIMARY,STRESS,PAIRS,candidate_name,stage_a,stage_b,markdown
from .restoration_metrics import evaluate_outputs


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False))


def run_source(image,scorer,receipt,*,max_steps=40):
    """All source actions still use only pixels and fixed candidate constants."""
    envelope=FixedObjective(scorer,image,receipt)
    results={name:run_candidate(image,envelope,name,max_steps=max_steps,record_states=True)
             for name in ('identity','region2_direct','region2_ttt_envelope')}
    for dark,bright in PAIRS:
        objective=ResidualObjective(scorer,image,receipt,Rhos(dark,bright))
        result=run_candidate(image,objective,'region2_ttt_rho',max_steps=max_steps,record_states=True)
        result['diagnostics']['rhos']=asdict(objective.rhos)
        results[candidate_name(dark,bright)]=result
    gate=dict(scores=objective.original_scores.cpu().tolist(),winner=objective.winner.cpu().tolist(),
              active=objective.active.cpu().tolist(),e0=objective.e0.cpu().tolist())
    return results,gate


def persist_then_evaluate(results,gate,clean,*,directory,image_id,condition):
    directory.mkdir(parents=True)
    pack={name:{k:r[k].cpu() for k in ('image','raw','grid')} for name,r in results.items()}
    states={name:dict(states=torch.stack(r['states']),final_raw=r['raw'].cpu())
            for name,r in results.items() if r['states']}
    outputs=directory/'outputs.pt';state_file=directory/'states.pt'
    torch.save(pack,outputs);torch.save(states,state_file)
    write(directory/'decisions.json',dict(gate=gate,methods={name:r['diagnostics'] for name,r in results.items()}))
    receipt={name:dict(file=path.name,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
             for name,path in (('outputs',outputs),('states',state_file),('decisions',directory/'decisions.json'))}
    write(directory/'label_free_receipt.json',receipt)
    # All methods, decisions, states and hashes now exist; only now read clean.
    rows=evaluate_outputs(results,clean,image_id=image_id,condition=condition)
    write(directory/'metrics.json',rows)
    return rows,receipt


def panel(results,clean,path,title):
    names=('clean',*results);columns=4;cell_w,cell_h=224,190
    canvas=Image.new('RGB',(columns*cell_w,35+((len(names)+columns-1)//columns)*cell_h),'white');draw=ImageDraw.Draw(canvas)
    draw.text((5,8),title,fill='black')
    for i,name in enumerate(names):
        image=clean if name=='clean' else results[name]['image'].cpu()
        array=(image[0].permute(1,2,0).clamp(0,1)*255).round().byte().numpy()
        tile=ImageOps.contain(Image.fromarray(array),(cell_w-8,cell_h-28))
        x=(i%columns)*cell_w;y=35+(i//columns)*cell_h
        draw.text((x+4,y+4),name,fill='black');canvas.paste(tile,(x+4,y+22))
    canvas.save(path)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',required=True,choices=('A','B'))
    for key in ('manifest','images','model-identity','prototypes','receipt','output','t006-images'):
        parser.add_argument('--'+key,required=True,type=Path)
    parser.add_argument('--source-sha',required=True)
    parser.add_argument('--calibration',type=Path);parser.add_argument('--calibration-commit')
    parser.add_argument('--device',default='cuda:0');args=parser.parse_args()
    calibration=None;rhos=None
    if args.stage=='B':
        calibration=json.loads(args.calibration.read_text())
        assert calibration['passes'] and args.calibration_commit
        selected=calibration['selected'];rhos=Rhos(selected['rho_dark'],selected['rho_bright'])
    torch.manual_seed(7);torch.set_num_threads(1);torch.use_deterministic_algorithms(False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    args.output.mkdir(parents=True,exist_ok=True);(args.output/'figures').mkdir(exist_ok=True)
    manifest=json.loads(args.manifest.read_text());receipt=json.loads(args.receipt.read_text());identity=json.loads(args.model_identity.read_text())
    prototype_sha=hashlib.sha256(args.prototypes.read_bytes()).hexdigest()
    assert prototype_sha==receipt['prototype_identity']['sha256']
    assert hashlib.sha256(Path(identity['path']).read_bytes()).hexdigest()==receipt['model_identity']['sha256']
    if calibration:
        assert calibration['prototype_identity']==receipt['prototype_identity'] and calibration['frozen_receipt']==receipt
        assert calibration['model_identity']==identity
    encoder=FrozenCLIP.from_checkpoint(identity['path'],args.device)
    saved=torch.load(args.prototypes,map_location=args.device,weights_only=True)
    prototypes=Prototypes(saved['raw']);scorer=SemanticScorer(encoder,prototypes)
    cal=load_image(args.t006_images/f"{receipt['image_ids'][0]:012d}.jpg").to(args.device)
    old=Path('research_log/remote_runs/20260912-071126-ttie-t006-a6000/artifacts/audit/calibration_scores.json')
    old_rows=[r for r in json.loads(old.read_text()) if r['image_id']==receipt['image_ids'][0] and r['view']!='full']
    expected=torch.tensor([[r['d_dark'],r['d_bright']] for r in old_rows])
    with torch.no_grad():actual=scorer(cal).cpu()
    assert torch.equal(actual,expected)
    write(args.output/'preflight.json',dict(original_calibration_bitwise_equal=True,prototype_sha256=prototype_sha))
    conditions=PRIMARY if args.stage=='A' else (*PRIMARY,STRESS)
    config=dict(stage=args.stage,manifest=manifest,manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        frozen_receipt=receipt,model_identity=identity,prototype_identity=receipt['prototype_identity'],source_sha=args.source_sha,
        conditions=conditions,candidate_pairs=PAIRS if args.stage=='A' else None,selected_rhos=asdict(rhos) if rhos else None,
        calibration=calibration,calibration_commit=args.calibration_commit,
        calibration_sha256=hashlib.sha256(args.calibration.read_bytes()).hexdigest() if args.calibration else None,
        optimizer=dict(name='Adam',lr=.03,max_updates=40,loss_stop=1e-8),seed=7,tf32=False,strict_determinism=False,
        representative_image_id=manifest['images'][0]['image_id'],storage='all float32 outputs and raw states before reference metrics')
    write(args.output/'config.json',config)
    rows=[];artifacts=[]
    for entry in manifest['images']:
        clean=load_image(args.images/entry['filename'])
        for condition in conditions:
            start=time.monotonic();index=len(artifacts);image=degrade(clean.to(args.device),condition)
            results,gate=run_source(image,scorer,receipt) if args.stage=='A' else run_all(image,scorer,receipt,rhos,record_states=True)
            directory=args.output/'episodes'/f'{index:03d}'
            episode_rows,pack_receipt=persist_then_evaluate(results,gate,clean,directory=directory,image_id=entry['image_id'],condition=condition)
            rows.extend(episode_rows)
            artifacts.append(dict(index=index,image_id=entry['image_id'],condition=condition,directory=str(directory.relative_to(args.output)),
                                  seconds=time.monotonic()-start,**pack_receipt))
            write(args.output/'metrics.json',rows);write(args.output/'artifact_manifest.json',artifacts)
            if entry['image_id']==config['representative_image_id']:
                panel(results,clean,args.output/'figures'/(condition+'.png'),f"T010 {args.stage} fixed ID {entry['image_id']}: {condition}")
            print('Finalized Stage',args.stage,entry['image_id'],condition,'inputs',len(artifacts),'seconds',round(time.monotonic()-start,2),flush=True)
            del results,image
    immutable=all(not p.requires_grad and p.grad is None for p in scorer.parameters()) and torch.equal(prototypes.vectors,saved['raw'])
    assert immutable
    report=stage_a(rows) if args.stage=='A' else stage_b(rows)
    write(args.output/'summary.json',report);(args.output/'summary.md').write_text(markdown(report))
    write(args.output/'final_checks.json',dict(frozen_assets_unchanged=True,inputs=len(artifacts),rows=len(rows),stage=args.stage,
          calibration_constants_unchanged=asdict(rhos)==config['selected_rhos'] if rhos else True))
    if args.stage=='A':
        write(args.output/'T010_calibration.json',dict(passes=report['passes'],selected=report['selected'],
          candidates=report['candidates'],selection=report['selection'],source_sha=args.source_sha,
          development_manifest_sha256=config['manifest_sha256'],model_identity=identity,prototype_identity=receipt['prototype_identity'],
          frozen_receipt=receipt,development_only=True))
        print('Stage A passes',report['passes'],'feasible',report['feasible_count'],'selected',report['selected'],flush=True)
        print('STOP: no Stage B in this process; commit a passing receipt before any fresh evaluation.',flush=True)
    else:print('Stage B qualified',report['qualified'],'failed',report['failed'],'STOP: no T011.',flush=True)


if __name__=='__main__':main()
