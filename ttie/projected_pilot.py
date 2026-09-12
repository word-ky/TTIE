"""T011 fixed fresh pilot. No calibration, selector or reference inside adaptation."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
from .clip_signal import FrozenCLIP
from .learned_prototypes import Prototypes
from .natural import load_image,degrade
from .semantic_ttt import SemanticScorer
from .projected_ttt import run_all,METHODS
from .projected_metrics import PRIMARY,STRESS,summarize,markdown
from .residual_pilot import persist_then_evaluate,panel,write


def main():
    parser=argparse.ArgumentParser()
    for key in ('manifest','images','model-identity','prototypes','receipt','output','t006-images'):
        parser.add_argument('--'+key,required=True,type=Path)
    parser.add_argument('--source-sha',required=True)
    parser.add_argument('--device',default='cuda:0');args=parser.parse_args()
    torch.manual_seed(7);torch.set_num_threads(1);torch.use_deterministic_algorithms(False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    args.output.mkdir(parents=True,exist_ok=True);(args.output/'figures').mkdir(exist_ok=True)
    manifest=json.loads(args.manifest.read_text());receipt=json.loads(args.receipt.read_text())
    identity=json.loads(args.model_identity.read_text())
    prototype_sha=hashlib.sha256(args.prototypes.read_bytes()).hexdigest()
    assert prototype_sha==receipt['prototype_identity']['sha256']
    assert hashlib.sha256(Path(identity['path']).read_bytes()).hexdigest()==receipt['model_identity']['sha256']
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
    config=dict(task='T011',manifest=manifest,manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        frozen_receipt=receipt,model_identity=identity,prototype_identity=receipt['prototype_identity'],source_sha=args.source_sha,
        conditions=(*PRIMARY,STRESS),methods=METHODS,optimizer=dict(name='Adam',lr=.03,max_updates=40,loss_stop=1e-8),
        seed=7,tf32=False,strict_determinism=False,representative_image_id=manifest['images'][0]['image_id'],
        projection='Frozen gate, EV dark[0,.5]/bright[-.5,0], gamma[.8,1.25], inactive identity; raw inverse box; Adam moments unchanged',
        global_sign='Agreeing active winners restrict sign, conflicts two-sided',
        storage='Float32 outputs, raw/physical pre/post states, gradients, decisions and hashes persisted before reference metrics')
    write(args.output/'config.json',config)
    rows=[];artifacts=[]
    for entry in manifest['images']:
        path=args.images/entry['filename']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256']
        clean=load_image(path)
        for condition in (*PRIMARY,STRESS):
            start=time.monotonic();index=len(artifacts);image=degrade(clean.to(args.device),condition)
            results,gate=run_all(image,scorer,receipt)
            directory=args.output/'episodes'/f'{index:03d}'
            episode_rows,pack_receipt=persist_then_evaluate(results,gate,clean,directory=directory,image_id=entry['image_id'],condition=condition)
            rows.extend(episode_rows)
            artifacts.append(dict(index=index,image_id=entry['image_id'],condition=condition,directory=str(directory.relative_to(args.output)),
                                  seconds=time.monotonic()-start,**pack_receipt))
            write(args.output/'metrics.json',rows);write(args.output/'artifact_manifest.json',artifacts)
            if entry['image_id']==config['representative_image_id']:
                panel(results,clean,args.output/'figures'/(condition+'.png'),f"T011 fixed ID {entry['image_id']}: {condition}")
            print('Finalized T011',entry['image_id'],condition,'inputs',len(artifacts),'seconds',round(time.monotonic()-start,2),flush=True)
            del results,image
    immutable=all(not p.requires_grad and p.grad is None for p in scorer.parameters()) and torch.equal(prototypes.vectors,saved['raw'])
    assert immutable
    report=summarize(rows);write(args.output/'summary.json',report);(args.output/'summary.md').write_text(markdown(report))
    write(args.output/'final_checks.json',dict(frozen_assets_unchanged=True,inputs=len(artifacts),rows=len(rows),task='T011'))
    print('T011 qualified',report['qualified'],'failed',report['failed'],'STOP: no automatic next task.',flush=True)


if __name__=='__main__':main()
