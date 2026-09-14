"""Fresh low-only qualification; both selections frozen before reference deployment."""
import argparse,json,time,hashlib
from pathlib import Path
from datetime import datetime,timezone
import torch
from PIL import Image
from ttie.lolv2_gamma_core import native_rgb,low_image_opener
from ttie.clip_signal import FrozenCLIP
from ttie.learned_prototypes import Prototypes
from ttie.semantic_ttt import SemanticScorer,FixedObjective,Region2
from ttie.energy_model import load_energy
from ttie.gamma_range_ttt import trajectory,evaluate_energy
from ttie.self_reversal import SPEC,select

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def utc():return datetime.now(timezone.utc).isoformat()
def main():
    p=argparse.ArgumentParser()
    for k in ['low-root','manifest','assets','spec','out']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();torch.manual_seed(7);torch.set_num_threads(1)
    torch.use_deterministic_algorithms(False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.is_available();assert json.loads(a.spec.read_bytes())==SPEC
    manifest=json.loads(a.manifest.read_bytes());rows=manifest['selected'];assert len(rows)==100
    assets=json.loads(a.assets.read_bytes());files=assets['files']
    for v in files.values():assert sha(v['path'])==v['sha256']
    for r in rows:assert sha(a.low_root/r['low'])==r['low_sha256']
    opened=[];allowed={str((a.low_root/r['low']).resolve()) for r in rows}
    Image.open=low_image_opener(allowed,opened)
    encoder=FrozenCLIP.from_checkpoint(files['clip']['path'],'cuda:0')
    protos=torch.load(files['prototypes']['path'],map_location='cuda:0',weights_only=True)
    scorer=SemanticScorer(encoder,Prototypes(protos['raw']))
    head=load_energy(files['energy']['path']).cuda();receipt=json.loads(Path(files['gate']['path']).read_bytes())
    a.out.mkdir(parents=True,exist_ok=False)
    write(a.out/'config.json',dict(task='T030-A',assets=assets,spec=SPEC,spec_sha256=sha(a.spec),
        manifest_sha256=sha(a.manifest),frozen_before_inference_utc=utc(),seed=7,tf32=False,
        accepted_source='b2359721c89db732d17e03be273e0bdb71bb377a',max_steps=40,optimizer='Adam lr=.03',
        gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda))
    frozen=[]
    for i,r in enumerate(rows):
        low=native_rgb(a.low_root/r['low']).cuda()
        torch.cuda.synchronize();start=time.perf_counter()
        original,t,original_decision=trajectory(low,scorer,receipt,head,basis='region2',max_steps=40)
        torch.cuda.synchronize();trajectory_seconds=time.perf_counter()-start;start=time.perf_counter()
        objective=FixedObjective(scorer,low,receipt);model=Region2(objective.active).to(low);gradients=[]
        assert objective.active.cpu().tolist()==t['gate']['active']
        for state in t['states']:
            with torch.no_grad():model.raw.copy_(state.cuda())
            value,*_=evaluate_energy(model,low,objective,head)
            gradient,=torch.autograd.grad(value,model.raw)
            assert torch.isfinite(gradient).all();gradients.append(gradient.detach().cpu())
        gradients=torch.stack(gradients)
        decision=select(original_decision['scores'],gradients,objective.active.cpu())
        assert decision['original_step']==original_decision['selected_step']
        step=decision['selected_step'];guarded={k:t[v][step].clone() for k,v in [('image','images'),('raw','states'),('grid','grids')]}
        assert all(torch.isfinite(x).all() for x in [t['states'],t['images'],gradients])
        torch.cuda.synchronize();guard_seconds=time.perf_counter()-start
        d=a.out/f'{i:03d}';d.mkdir()
        torch.save({k:original[k].clone() for k in ['image','raw','grid']},d/'original.pt')
        torch.save(guarded,d/'guarded.pt')
        torch.save(dict(states=t['states'],grids=t['grids'],features=t['features'],gradients=gradients),d/'trajectory.pt')
        write(d/'decision.json',dict(original=original_decision,guarded=decision,gate=t['gate'],diagnostics=t['diagnostics']))
        frozen.append(dict(index=i,low=r['low'],updates=t['diagnostics']['steps'],states=len(t['states']),
            original_step=decision['original_step'],selected_step=step,cutoff=decision['cutoff'],crossing=decision['crossing'],fallback=decision['fallback'],
            trajectory_seconds=trajectory_seconds,guard_seconds=guard_seconds,persisted_utc=utc(),
            files={n:dict(sha256=sha(d/n),bytes=(d/n).stat().st_size) for n in ['original.pt','guarded.pt','trajectory.pt','decision.json']}))
        write(a.out/'progress.json',frozen);print(f'{i+1}/100 updates={frozen[-1]["updates"]} cutoff={decision["cutoff"]}',flush=True)
        del t,original,guarded,low
    for v in files.values():assert sha(v['path'])==v['sha256']
    assert opened==[str((a.low_root/r['low']).resolve()) for r in rows]
    write(a.out/'freeze.json',dict(completed_utc=utc(),rows=frozen,opened_images=opened,normal_decodes=0,
        manifest_sha256=sha(a.manifest),spec_sha256=sha(a.spec),config_sha256=sha(a.out/'config.json'),assets_unchanged=True))
    print('100 original and guarded outputs frozen; normal decodes0',flush=True)
if __name__=='__main__':main()
