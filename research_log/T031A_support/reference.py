"""REFERENCE_GRADIENT_DIAGNOSTIC_ONLY after frozen support scores; never a selector."""
import argparse,json,hashlib,time
from pathlib import Path
from datetime import datetime,timezone
import torch
from PIL import Image
from ttie.lolv2_gamma_core import native_rgb
from ttie.clip_signal import FrozenCLIP
from ttie.learned_prototypes import Prototypes
from ttie.semantic_ttt import SemanticScorer,FixedObjective,Region2
from ttie.energy_model import load_energy
from ttie.gamma_range_ttt import evaluate_energy
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,indent=2)+'\n')
def state_hash(m):return {n:hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest() for n,t in m.state_dict().items()}
def main():
    p=argparse.ArgumentParser()
    for k in ['support','manifest','assets','low-root','normal-root','out']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();torch.manual_seed(7);torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    start=time.perf_counter();f=json.loads((a.support/'freeze.json').read_bytes())
    assert sha(a.support/'scores.json')==f['scores_sha256'] and sha(a.support/'bound_features.pt')==f['bound_features_sha256']
    assert sha(a.manifest)==f['cohort_sha256'];manifest=json.loads(a.manifest.read_bytes())['selected']
    bound=torch.load(a.support/'bound_features.pt',weights_only=True,map_location='cpu')
    assets=json.loads(a.assets.read_bytes())['files']
    for v in assets.values():assert sha(v['path'])==v['sha256']
    assert assets['energy']['sha256']==f['checkpoint_sha256']
    scorer=SemanticScorer(FrozenCLIP.from_checkpoint(assets['clip']['path'],'cuda:0'),Prototypes(torch.load(assets['prototypes']['path'],weights_only=True,map_location='cuda:0')['raw']))
    head=load_energy(assets['energy']['path']).cuda();receipt=json.loads(Path(assets['gate']['path']).read_bytes())
    before=dict(scorer=state_hash(scorer),head=state_hash(head));opened=[];normal_names={str((a.normal_root/r['normal']).resolve()) for r in manifest}
    allowed=normal_names|{str((a.low_root/r['low']).resolve()) for r in manifest};original=Image.open
    def guarded(path,*args,**kwargs):
        name=str(Path(path).resolve());assert name in allowed
        opened.append(dict(path=name,normal=name in normal_names,utc=datetime.now(timezone.utc).isoformat()))
        return original(path,*args,**kwargs)
    Image.open=guarded;a.out.mkdir(parents=True,exist_ok=False);rows=[];all_gradients=[]
    for i,r in enumerate(manifest):
        assert sha(a.low_root/r['low'])==r['low_sha256'] and sha(a.normal_root/r['normal'])==r['normal_sha256']
        low=native_rgb(a.low_root/r['low']).cuda();objective=FixedObjective(scorer,low,receipt)
        assert objective.active.cpu().tolist()==f['accepted_bindings'][i]['gate']['active']
        model=Region2(objective.active).to(low);normal=native_rgb(a.normal_root/r['normal']).cuda()
        mask=objective.active.flatten().repeat(2);gradients=[]
        for step,raw in enumerate(bound['raw_states'][i]):
            with torch.no_grad():model.raw.copy_(raw.cuda())
            version=model.raw._version;energy,output,_,_,features=evaluate_energy(model,low,objective,head)
            assert torch.equal(features.detach().cpu(),bound['state_features'][i*41+step])
            g_e,=torch.autograd.grad(energy,model.raw)
            # A separate renderer graph; no update or selection consumes this derivative.
            loss=(model(low).double()-normal.double()).square().mean();g_r,=torch.autograd.grad(loss,model.raw)
            assert torch.isfinite(output).all() and torch.isfinite(g_e).all() and torch.isfinite(g_r).all()
            assert model.raw._version==version and torch.equal(model.raw.detach().cpu(),raw) and model.raw.grad is None
            e=g_e.flatten()[mask].double();rgrad=g_r.flatten()[mask].double();en=float(e.norm());rn=float(rgrad.norm());dot=float(e@rgrad)
            rows.append(dict(index=i,step=step,active_count=int(mask.sum()),energy_norm=en,reference_norm=rn,dot=dot,valid=dot>0,
                cosine=dot/(en*rn) if en>1e-12 and rn>1e-12 else None))
            gradients.append(torch.stack([g_e.detach().cpu(),g_r.detach().cpu()]))
        all_gradients.append(torch.stack(gradients));print(f'{i+1}/100 post-freeze diagnostic',flush=True)
    assert before==dict(scorer=state_hash(scorer),head=state_hash(head))
    for v in assets.values():assert sha(v['path'])==v['sha256']
    assert sha(a.support/'scores.json')==f['scores_sha256']
    write(a.out/'validity.json',rows);torch.save(torch.stack(all_gradients),a.out/'gradients.pt')
    write(a.out/'receipt.json',dict(label='REFERENCE_GRADIENT_DIAGNOSTIC_ONLY',completed_utc=datetime.now(timezone.utc).isoformat(),
        support_freeze_sha256=sha(a.support/'freeze.json'),support_completed_utc=f['completed_utc'],opened=opened,states=4100,
        optimizer_updates=0,selection_decisions=0,model_unchanged=True,raw_unchanged_during_gradients=True,features_exact=True,all_finite=True,
        seconds=time.perf_counter()-start,torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
        validity_sha256=sha(a.out/'validity.json'),gradients_sha256=sha(a.out/'gradients.pt')))
if __name__=='__main__':main()
