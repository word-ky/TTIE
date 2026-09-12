"""Fresh T015 execution composed from frozen T014 runners and asset checks."""
import argparse
import json
from pathlib import Path
import time
import torch
from ..clip_signal import FrozenCLIP
from ..learned_prototypes import Prototypes
from ..semantic_ttt import SemanticScorer
from ..natural import load_image,degrade
from ..energy_model import load_energy,SCHEMA
from ..sobolev_receipt import verify_receipt
from ..sobolev_metrics import trajectory_diagnostics
from ..residual_metrics import PRIMARY as CONDITIONS,STRESS
from ..residual_pilot import write,panel
from ..stop_receipt import sha
from .io import run_label_free,persist_then_evaluate
from .core import BASES,PRIMARY,ORACLE
from .metrics import summarize,routing_diagnostics,markdown,summarize_trajectories
from .provenance import T015_SOURCES,verify_source


def evaluate(manifest,images,output,scorer,receipt,head,device,*,max_steps=40):
    entries=[];rows=[];representative=manifest['images'][0]['image_id'];(output/'figures').mkdir(parents=True)
    for entry in manifest['images']:
        path=images/entry['filename'];assert sha(path)==entry['sha256'];clean=load_image(path)
        for condition in (*CONDITIONS,STRESS):
            start=time.monotonic();image=degrade(clean.to(device),condition)
            results,old,ts,decisions=run_label_free(image,scorer,receipt,head,max_steps=max_steps)
            directory=output/'episodes'/f'{len(entries):03d}'
            files,case,oracle_info,oracle=persist_then_evaluate(directory,results,old,ts,decisions,lambda:clean,
                image_id=entry['image_id'],condition=condition)
            rows.extend(case)
            entries.append(dict(image_id=entry['image_id'],condition=condition,directory=directory.relative_to(output).as_posix(),
                files=files,selections={m:decisions[m] for m in BASES},routing=decisions['routing'],oracle=oracle_info,
                semantic_checkpoints=len(old['images']),energy_checkpoints={m:len(t['images']) for m,t in ts.items()},
                trajectory_diagnostics=trajectory_diagnostics(ts,decisions),seconds=time.monotonic()-start))
            write(output/'artifact_manifest.json',entries);write(output/'metrics.json',rows)
            if entry['image_id']==representative:
                panel(dict(results,**{ORACLE:oracle}),clean,output/'figures'/(condition+'.png'),f'T015 fixed ID {representative}: {condition}')
            print('T015 finalized',len(entries),entry['image_id'],condition,'basis',decisions['routing']['selected_basis'],
                'seconds',round(time.monotonic()-start,2),flush=True)
            del results,old,ts,image,oracle
    report=summarize(rows);report['trajectory_distributions']=summarize_trajectories(entries)
    write(output/'routing_diagnostics.json',routing_diagnostics(entries))
    write(output/'summary.json',report);(output/'summary.md').write_text(markdown(report))
    return report,entries


def main():
    p=argparse.ArgumentParser()
    for key in ('manifest','source-manifest','images','model-identity','prototypes','receipt','output','t006-images','energy','control','energy-receipt'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--source-sha',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    code=verify_source(a.source_sha,T015_SOURCES)
    torch.manual_seed(7);torch.set_num_threads(1);torch.use_deterministic_algorithms(False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    receipt=json.loads(a.receipt.read_text());identity=json.loads(a.model_identity.read_text())
    frozen=verify_receipt(a.energy_receipt,a.energy,a.control,a.source_manifest,identity,receipt)
    manifest=json.loads(a.manifest.read_text())
    assert len(manifest['images'])==40 and len(manifest['excluded_prior_ids'])==648
    assert manifest['frozen_t014_receipt_sha256']==sha(a.energy_receipt)
    assert manifest['frozen_sobolev_sha256']==frozen['energy_sha256']
    assert sha(a.prototypes)==receipt['prototype_identity']['sha256'] and sha(Path(identity['path']))==receipt['model_identity']['sha256']
    encoder=FrozenCLIP.from_checkpoint(identity['path'],a.device)
    saved=torch.load(a.prototypes,map_location=a.device,weights_only=True);prototypes=Prototypes(saved['raw']);scorer=SemanticScorer(encoder,prototypes)
    cal=load_image(a.t006_images/f"{receipt['image_ids'][0]:012d}.jpg").to(a.device)
    old=Path('research_log/remote_runs/20260912-071126-ttie-t006-a6000/artifacts/audit/calibration_scores.json')
    old_rows=[r for r in json.loads(old.read_text()) if r['image_id']==receipt['image_ids'][0] and r['view']!='full']
    expected=torch.tensor([[r['d_dark'],r['d_bright']] for r in old_rows])
    with torch.no_grad():actual=scorer(cal).cpu()
    assert torch.equal(actual,expected)
    a.output.mkdir(parents=True,exist_ok=True)
    write(a.output/'preflight.json',dict(original_calibration_bitwise_equal=True,frozen_t014_receipt_verified=True))
    write(a.output/'config.json',dict(task='T015',source_sha=a.source_sha,source_code_sha256=code,
        frozen_t014_source_sha=frozen['source_sha'],manifest=manifest,manifest_sha256=sha(a.manifest),
        frozen_t014_receipt_sha256=sha(a.energy_receipt),frozen_receipt=receipt,model_identity=identity,
        frozen_energy_sha256=sha(a.energy),frozen_control_sha256=sha(a.control),schema=SCHEMA,basis_tie_order=list(BASES),
        fixed_step_source=16,seed=7,tf32=False,strict_cuda_determinism=False,device=a.device,training_inputs=0))
    head=load_energy(a.energy);digest=sha(a.energy)
    report,entries=evaluate(manifest,a.images,a.output,scorer,receipt,head,a.device)
    assert digest==sha(a.energy)==frozen['energy_sha256']
    assert all(not q.requires_grad and q.grad is None for q in head.parameters())
    assert all(not q.requires_grad and q.grad is None for q in scorer.parameters()) and torch.equal(prototypes.vectors,saved['raw'])
    write(a.output/'final_checks.json',dict(task='T015',training_inputs=0,evaluation_inputs=len(entries),
        energy_checkpoints=sum(sum(e['energy_checkpoints'].values()) for e in entries),frozen_assets_unchanged=True,
        energy_sha256=digest,source_sha=a.source_sha,manifest_sha256=sha(a.manifest),offset_is_primary=True))
    print('T015 qualified',report['qualified'],'failed',report['failed'],flush=True)
    print('STOP: no rerun, calibration, fitting, new split or next task.',flush=True)


if __name__=='__main__':main()
