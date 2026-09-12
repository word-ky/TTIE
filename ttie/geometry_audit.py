"""T009 diagnostic orchestration. Semantic simulation finishes before oracles."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch

from .clip_signal import FrozenCLIP
from .learned_prototypes import Prototypes
from .natural import load_image,degrade
from .semantic_ttt import SemanticScorer,FixedObjective,run_method,direct_grid,EVGamma
from .offline_geometry import (Piecewise2,gradient_alignment,attach_trajectory_mse,surface_semantics,
                              label_surface,oracle_renderer,region_mask,SURFACE_EV,SURFACE_GAMMA)
from .geometry_summary import CONDITIONS,COORDINATES,diagnose,markdown


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False))


def main():
    parser=argparse.ArgumentParser()
    for key in ('manifest','images','model-identity','prototypes','receipt','output','t006-images'):
        parser.add_argument('--'+key,required=True,type=Path)
    parser.add_argument('--device',default='cuda:0');args=parser.parse_args()
    torch.manual_seed(7);torch.set_num_threads(1);torch.use_deterministic_algorithms(False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    args.output.mkdir(parents=True,exist_ok=True);(args.output/'episodes').mkdir(exist_ok=True)
    manifest=json.loads(args.manifest.read_text());receipt=json.loads(args.receipt.read_text());identity=json.loads(args.model_identity.read_text())
    assert hashlib.sha256(args.prototypes.read_bytes()).hexdigest()==receipt['prototype_identity']['sha256']
    assert hashlib.sha256(Path(identity['path']).read_bytes()).hexdigest()==receipt['model_identity']['sha256']
    encoder=FrozenCLIP.from_checkpoint(identity['path'],args.device)
    saved=torch.load(args.prototypes,map_location=args.device,weights_only=True)
    prototypes=Prototypes(saved['raw']);scorer=SemanticScorer(encoder,prototypes)
    first=receipt['image_ids'][0]
    cal=load_image(args.t006_images/f'{first:012d}.jpg').to(args.device)
    old=Path('research_log/remote_runs/20260912-071126-ttie-t006-a6000/artifacts/audit')
    old_rows=[r for r in json.loads((old/'calibration_scores.json').read_text()) if r['image_id']==first and r['view']!='full']
    expected=torch.tensor([[r['d_dark'],r['d_bright']] for r in old_rows])
    with torch.no_grad():actual=scorer(cal).cpu()
    assert torch.equal(actual,expected)
    write(args.output/'preflight.json',dict(original_calibration_bitwise_equal=True,prototype_sha256=receipt['prototype_identity']['sha256']))
    write(args.output/'config.json',dict(manifest=manifest,receipt=receipt,model_identity=identity,conditions=CONDITIONS,
        coordinates=COORDINATES,sizes=[1,2],ttt=dict(optimizer='Adam',lr=.03,max_updates=40,stop=1e-8),
        surface_ev=SURFACE_EV,surface_gamma=SURFACE_GAMMA,oracle=dict(diagnostic_only=True,optimizer='Adam',lr=.03,updates=100),
        representative_image_id=manifest['images'][0]['image_id'],development_only=True,seed=7,tf32=False,strict_determinism=False))
    records=[];surfaces=[];renderers=[];artifacts=[]
    for entry in manifest['images']:
        clean=load_image(args.images/entry['filename']).to(args.device)
        for condition in CONDITIONS:
            start=time.monotonic();index=len(artifacts)
            image=degrade(clean,condition);objective=FixedObjective(scorer,image,receipt)
            simulations={}
            for size in (1,2):
                for coordinates in COORDINATES:
                    method='global_ttt' if size==1 else 'spatial2_ttt'
                    simulations[f'{size}_{coordinates}']=run_method(image,objective,method,coordinates=coordinates,record_states=True)
            directory=args.output/'episodes'/f'{index:03d}';directory.mkdir()
            pack={name:dict(states=torch.stack(result['states']),final_raw=result['raw'].cpu()) for name,result in simulations.items()}
            state_path=directory/'semantic_states.pt';torch.save(pack,state_path)
            write(directory/'semantic_decisions.json',dict(gate=dict(scores=objective.original_scores.cpu().tolist(),
                winner=objective.winner.cpu().tolist(),active=objective.active.cpu().tolist()),
                methods={name:result['diagnostics'] for name,result in simulations.items()}))
            # Reference data first enters the structurally separate offline APIs here.
            case_records=[]
            for size in (1,2):
                for coordinates in COORDINATES:
                    result=simulations[f'{size}_{coordinates}']
                    trace=attach_trajectory_mse(image,clean,result,size=size,coordinates=coordinates,condition=condition)
                    alignment=gradient_alignment(image,clean,objective,size=size,coordinates=coordinates,condition=condition) if objective.active.any() and condition!='clean' else None
                    case_records.append(dict(image_id=entry['image_id'],condition=condition,size=size,coordinates=coordinates,
                        active_count=int(objective.active.sum()),trace=trace,alignment=alignment,stop_reason=result['diagnostics']['stop_reason']))
            write(directory/'offline_trajectories.json',case_records);records.extend(case_records)
            if condition in ('homogeneous_dark','homogeneous_bright'):
                sem=surface_semantics(image,objective)
                write(directory/'surface_before_reference.json',sem)
                labeled,summary=label_surface(image,clean,sem,condition=condition)
                item=dict(image_id=entry['image_id'],condition=condition,active_count=int(objective.active.sum()),rows=labeled,summary=summary)
                surfaces.append(item);write(directory/'surface_offline.json',item)
            if condition in ('left_right','quadrants'):
                node_values=direct_grid(objective,2);mask=region_mask(image,condition)
                comparisons=[]
                for renderer in ('bilinear2','piecewise2'):
                    model=(EVGamma(2) if renderer=='bilinear2' else Piecewise2()).to(image)
                    model.set_grid(node_values)
                    with torch.no_grad():direct=model(image)
                    oracle=oracle_renderer(image,clean,renderer)
                    write(directory/(renderer+'_oracle.json'),dict(diagnostic_only=True,initial_raw_zero=True,steps=100,
                        loss_trajectory=oracle['loss_trajectory'],final_raw=oracle['raw'].cpu().tolist(),final_grid=oracle['grid'].cpu().tolist()))
                    for method,output in (('direct',direct),('oracle',oracle['image'])):
                        square=(output-clean).square()
                        comparisons.append(dict(image_id=entry['image_id'],condition=condition,method=method,renderer=renderer,
                            mse=float(square.mean()),dark_mse=float(square[...,mask].mean()),bright_mse=float(square[...,~mask].mean()),
                            direct_node_grid=node_values.cpu().tolist() if method=='direct' else None,diagnostic_only=True))
                renderers.extend(comparisons);write(directory/'renderer_comparisons.json',comparisons)
            artifacts.append(dict(index=index,image_id=entry['image_id'],condition=condition,state_path=str(state_path.relative_to(args.output)),
                          bytes=state_path.stat().st_size,sha256=hashlib.sha256(state_path.read_bytes()).hexdigest()))
            write(args.output/'artifact_manifest.json',artifacts)
            print('Finalized development',entry['image_id'],condition,'episodes',len(artifacts),'seconds',round(time.monotonic()-start,2),flush=True)
            del simulations,pack,objective,image,case_records
    summary=diagnose(records,surfaces,renderers)
    write(args.output/'trajectories.json',records);write(args.output/'surfaces.json',surfaces);write(args.output/'renderers.json',renderers)
    write(args.output/'summary.json',summary);(args.output/'summary.md').write_text(markdown(summary))
    immutable=all(not p.requires_grad and p.grad is None for p in scorer.parameters()) and torch.equal(prototypes.vectors,saved['raw'])
    assert immutable
    write(args.output/'final_checks.json',dict(frozen_assets_unchanged=True,episodes=len(artifacts),coordinate_trajectories=len(records),
            surfaces=len(surfaces),surface_points=sum(len(r['rows']) for r in surfaces),renderer_rows=len(renderers),development_only=True))
    print('T009 diagnosis rules',summary['rules'],flush=True)
    print('Development diagnosis complete; no T010/held-out qualification.',flush=True)


if __name__=='__main__':main()
