"""T008 fixed seven-method restoration pilot; outputs precede reference metrics."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

from PIL import Image,ImageDraw
import torch
from torch.nn import functional as F

from .clip_signal import FrozenCLIP
from .learned_prototypes import Prototypes
from .natural import CONDITIONS,load_image,degrade
from .semantic_ttt import METHODS,SemanticScorer,FixedObjective,EVGamma,run_all
from .restoration_metrics import evaluate_outputs,summarize_restoration,markdown_summary


def save_csv(rows,path):
    with path.open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def panel(results,clean,path,condition):
    # Fixed first numeric ID for every condition; shared physical color ranges.
    width,height=176,132
    canvas=Image.new('RGB',(width*8,460),'white');draw=ImageDraw.Draw(canvas)
    draw.text((5,3),condition+' | RGB; EV blue(-2) white(0) red(+2); gamma black(.5) white(2)',fill='black')
    for index,name in enumerate(('clean',*METHODS)):
        image=clean if name=='clean' else results[name]['image'].cpu()
        array=(image[0].permute(1,2,0).clamp(0,1)*255).round().byte().numpy()
        canvas.paste(Image.fromarray(array).resize((width,height)),(index*width,44))
        draw.text((index*width+2,24),name,fill='black')
        if name=='clean':continue
        field=F.interpolate(results[name]['grid'].cpu(),size=(height,width),mode='bilinear',align_corners=False)[0]
        ev=(field[0]/2).clamp(-1,1)
        rgb=torch.stack((1+ev.clamp_max(0),1-ev.abs(),1-ev.clamp_min(0)),dim=-1)
        canvas.paste(Image.fromarray((rgb*255).round().byte().numpy()),(index*width,190))
        gray=((field[1]-.5)/1.5).clamp(0,1)
        canvas.paste(Image.fromarray((gray*255).round().byte().numpy()).convert('RGB'),(index*width,328))
    canvas.save(path)


def main():
    parser=argparse.ArgumentParser()
    for key in ('manifest','images','model-identity','prototypes','receipt','output','t006-images'):
        parser.add_argument('--'+key,required=True,type=Path)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    torch.manual_seed(7);torch.set_num_threads(1)
    # Observed in T004: torch2.4 antialiased bicubic CUDA backward lacks a strict
    # deterministic implementation. Permit that backward; keep seed/TF32 fixed.
    torch.use_deterministic_algorithms(False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'episodes').mkdir(exist_ok=True);(args.output/'figures').mkdir(exist_ok=True)
    manifest=json.loads(args.manifest.read_text());receipt=json.loads(args.receipt.read_text())
    identity=json.loads(args.model_identity.read_text())
    assert hashlib.sha256(args.prototypes.read_bytes()).hexdigest()==receipt['prototype_identity']['sha256']
    assert hashlib.sha256(Path(identity['path']).read_bytes()).hexdigest()==receipt['model_identity']['sha256']
    encoder=FrozenCLIP.from_checkpoint(identity['path'],args.device)
    weights=torch.load(args.prototypes,map_location=args.device,weights_only=True)
    prototypes=Prototypes(weights['raw'])
    scorer=SemanticScorer(encoder,prototypes)
    config=dict(manifest=manifest,receipt=receipt,model_identity=identity,methods=METHODS,conditions=CONDITIONS,
         views='four quadrants; original mask frozen',optimizer=dict(name='Adam',lr=.03,max_steps=40,stop=1e-8),
         objective='mean_active(sum_types(relu((d-tau)/scale)^2))',dtype='float32 scores/ISP;float64 normalized loss',
         seed=7,tf32=False,strict_deterministic_algorithms=False,reason='observed unsupported antialiased bicubic CUDA backward',
         output_storage='all seven float32 outputs persisted per input before metrics; no clean reference in decision tensor pack',
         representative_image_id=manifest['images'][0]['image_id'])
    (args.output/'config.json').write_text(json.dumps(config,indent=2))
    first=receipt['image_ids'][0]
    cal=load_image(args.t006_images/f'{first:012d}.jpg').to(args.device)
    old=Path('research_log/remote_runs/20260912-071126-ttie-t006-a6000/artifacts/audit')
    old_rows=[r for r in json.loads((old/'calibration_scores.json').read_text()) if r['image_id']==first and r['view']!='full']
    expected=torch.tensor([[r['d_dark'],r['d_bright']] for r in old_rows])
    with torch.no_grad():actual=scorer(cal).cpu()
    assert torch.equal(actual,expected)
    probe=cal*.45
    objective=FixedObjective(scorer,probe,receipt);model=EVGamma(2).to(probe)
    loss=objective(model(probe));gradient,=torch.autograd.grad(loss,model.raw,retain_graph=True)
    repeat,=torch.autograd.grad(loss,model.raw)
    checks=dict(old_calibration_scores_bitwise_equal=True,active_probe_quadrants=int(objective.active.sum()),
                gradient_norm=float(gradient.norm()),gradient_per_coordinate=gradient.abs().sum(dim=(0,2,3)).tolist(),
                gradient_repeat_max_delta=float((gradient-repeat).abs().max()),
                frozen_no_grad=all(not p.requires_grad and p.grad is None for p in scorer.parameters()))
    (args.output/'preflight.json').write_text(json.dumps(checks,indent=2))
    assert torch.isfinite(gradient).all() and (gradient.abs().sum(dim=(0,2,3))>0).all() and checks['frozen_no_grad']
    del gradient,repeat,loss,objective,model,probe,cal
    rows=[];outputs=[]
    for entry in manifest['images']:
        clean=load_image(args.images/entry['filename'])
        for condition in CONDITIONS:
            index=len(outputs);start=time.monotonic()
            image=degrade(clean.to(args.device),condition)
            results,gate=run_all(image,scorer,receipt)
            pack={name:{key:result[key].cpu() for key in ('image','raw','grid')} for name,result in results.items()}
            directory=args.output/'episodes'/f'{index:03d}';directory.mkdir()
            path=directory/'outputs.pt';torch.save(pack,path)
            diagnostics=dict(gate=gate,methods={name:result['diagnostics'] for name,result in results.items()})
            (directory/'decisions.json').write_text(json.dumps(diagnostics,indent=2))
            # All methods finalized and persisted before reference/condition metrics.
            rows.extend(evaluate_outputs(results,clean,image_id=entry['image_id'],condition=condition))
            outputs.append(dict(index=index,image_id=entry['image_id'],condition=condition,path=str(path.relative_to(args.output)),
                                bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),seconds=time.monotonic()-start))
            (args.output/'metrics.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
            (args.output/'output_manifest.json').write_text(json.dumps(outputs,indent=2))
            if entry['image_id']==config['representative_image_id']:
                panel(results,clean,args.output/'figures'/(condition+'.png'),condition)
            print('Finalized',entry['image_id'],condition,'episodes',len(outputs),'seconds',round(outputs[-1]['seconds'],2),flush=True)
            del results,pack,image
    immutable=all(not p.requires_grad and p.grad is None for p in scorer.parameters()) and torch.equal(prototypes.vectors,weights['raw'])
    report,paired=summarize_restoration(rows,immutable=immutable)
    (args.output/'summary.json').write_text(json.dumps(report,indent=2))
    (args.output/'summary.md').write_text(markdown_summary(report))
    (args.output/'paired.json').write_text(json.dumps(paired,indent=2))
    save_csv(rows,args.output/'metrics.csv');save_csv(paired,args.output/'paired.csv')
    (args.output/'final_checks.json').write_text(json.dumps(dict(frozen_no_grad_unchanged=immutable,rows=len(rows),episodes=len(outputs),
         finite=all(r['all_finite'] for o in outputs for r in json.loads((args.output/'episodes'/f"{o['index']:03d}"/'decisions.json').read_text())['methods'].values()),
         prototype_sha256=hashlib.sha256(args.prototypes.read_bytes()).hexdigest()),indent=2))
    print('T008 verdict',report['qualifies_later_detector'],report['failed_criteria'],flush=True)
    print('T008 complete. Stop; no detector/T009 authorized.',flush=True)


if __name__=='__main__':main()
