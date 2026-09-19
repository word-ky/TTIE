"""Target-free stage: one shared online J, two separately normalized heads."""
import argparse
from ttie.clip_signal import FrozenCLIP
from ttie.learned_prototypes import Prototypes
from ttie.semantic_ttt import SemanticScorer
from ttie.energy_model import load_energy,features
from research_log.T060B.core import *

def main(out):
    out.mkdir(parents=True,exist_ok=False);setup();start=utc()
    binding=json.loads((HERE/'source_binding.json').read_bytes());s=json.loads((HERE/'selection.json').read_bytes());sf=json.loads((HERE/'selection_freeze.json').read_bytes())
    inputs=dict(s['inputs']);inputs.update({str(p):HEAD_SHA[h] for h,p in HEADS.items()})
    firewall(set(inputs)|{str(Path(n).resolve()) for n in binding}|{str((HERE/'source_binding.json').resolve())},out)
    validate(binding);validate(inputs);assert sf['sha256']==sha(HERE/'selection.json') and sf['utc']<start
    if len(s['anchors'])<40:
        atomic_json(out/'result.json',dict(classification='insufficient target-free common-gain-active coverage',rows=len(s['anchors']),clean_reference_reads=0));return
    config=json.loads((B/'config.json').read_bytes());encoder=FrozenCLIP.from_checkpoint(config['model_identity']['path'],'cuda:0')
    proto=ROOT/'research_log/remote_runs/20260912-071126-ttie-t006-a6000/artifacts/audit/prototypes.pt'
    scorer=SemanticScorer(encoder,Prototypes(torch.load(proto,weights_only=True,map_location='cuda:0')['raw'])).eval().requires_grad_(False)
    heads={h:load_energy(p) for h,p in HEADS.items()};before={h:{k:thash(t) for k,t in head.state_dict().items()} for h,head in heads.items()};records=[]
    for r in s['anchors']:
        directory=Path(r['directory']);bank=torch.load(directory/'bank.pt',weights_only=True,map_location='cpu');low=torch.load(directory/'bank_images.pt',weights_only=True,map_location='cpu')[0].cuda()
        gate=r['gate'];active=torch.tensor(gate['active'],device='cuda:0',dtype=torch.bool)
        obj=SimpleNamespace(active=active,winner=torch.tensor(gate['winner'],device='cuda:0'),evidence=torch.tensor(gate['evidence'],device='cuda:0',dtype=torch.float64),calibration=s['calibration'])
        raw=torch.cat((bank['states'][0],torch.zeros_like(bank['states'][0][:,:1])),1).cuda();model=Gain(low,raw,active);y0=model()
        assert thash(y0)==r['state_before']['y0'] and thash(model.grid)==r['state_before']['grid']
        x=features(obj,scorer(y0),model.grid);J=jacobian(x,model.gain).detach().cpu()
        tensors=dict(x=x.detach().cpu(),J=J,low=low.cpu(),base=y0.detach().cpu(),grid=model.grid.cpu(),raw=raw.cpu(),gain=model.gain.detach().cpu())
        norms={}
        for h,head in heads.items():
            leaf=x.detach().cpu().float().requires_grad_(True);q,=torch.autograd.grad(head(leaf).sum(),leaf);g=torch.einsum('fi,f->i',J,q).detach()
            tensors['q_'+h]=q.detach();tensors['g_'+h]=g;norms[h]=float(g.double().norm())
        assert all(torch.isfinite(t).all() for t in tensors.values());assert torch.equal(raw,model.fixed) and torch.count_nonzero(model.gain)==0
        name=f"prediction_{r['bank_index']:03d}.pt";save_tensor(out/name,tensors)
        records.append(dict(**r,file=name,sha256=sha(out/name),tensor_hashes={k:thash(t) for k,t in tensors.items()},norms=norms,persisted_utc=utc()))
        print(json.dumps(dict(completed=len(records),bank=r['bank_index'])),flush=True)
    after={h:{k:thash(t) for k,t in head.state_dict().items()} for h,head in heads.items()};assert before==after
    validate(binding);validate(inputs);atomic_json(out/'predictions.json',records)
    atomic_json(out/'prediction_freeze.json',dict(utc=utc(),started_utc=start,rows=len(records),selection_freeze=sf,files={n:sha(out/n) for n in ['predictions.json']+[r['file'] for r in records]},inputs=inputs,source_bindings=binding,head_state_before=before,head_state_after=after,normalizations={h:head.normalization() for h,head in heads.items()},optimizer_steps=0,clean_reference_reads=0,reference_gradient_reads=0,target_domain_access=0,lolv2_access=0,official_test_access=0,inference_reference_leakage=0,physical_gpu=1))
    print('ALL_PREDICTED_GRADIENTS_FROZEN',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
