"""T060-D-R1 required exact source gate reproduction, before either trajectory."""
import argparse
from ttie.clip_signal import FrozenCLIP
from ttie.learned_prototypes import Prototypes
from ttie.semantic_ttt import SemanticScorer,FixedObjective
from research_log.T060B.core import *

HERE=Path('research_log/T060DR1')

def main(out):
    out.mkdir(parents=True,exist_ok=False);setup();start=utc()
    selection=Path('research_log/T060B/selection.json')
    assert sha(selection)=='494fdafcf0db462d7d8157175bd50a879217f6e5f0c973c9af833b50ac726297'
    s=json.loads(selection.read_bytes());assert len(s['anchors'])==60
    binding=json.loads((HERE/'source_binding.json').read_bytes())
    inputs={n:h for n,h in s['inputs'].items() if n.endswith('config.json') or n.endswith('open_clip_pytorch_model.bin') or n.endswith('prototypes.pt') or any(n==r['directory']+'/bank_images.pt' for r in s['anchors'])}
    inputs.update({str(v):HEAD_SHA[k] for k,v in HEADS.items()})
    firewall(set(inputs)|{str(Path(n).resolve()) for n in binding}|{str((HERE/'source_binding.json').resolve())},out)
    validate(binding);validate(inputs)
    config=json.loads((B/'config.json').read_bytes());receipt=config['frozen_receipt'];assert receipt['calibration']==s['calibration']
    encoder=FrozenCLIP.from_checkpoint(config['model_identity']['path'],'cuda:0')
    proto=ROOT/'research_log/remote_runs/20260912-071126-ttie-t006-a6000/artifacts/audit/prototypes.pt'
    scorer=SemanticScorer(encoder,Prototypes(torch.load(proto,weights_only=True,map_location='cuda:0')['raw'])).eval().requires_grad_(False)
    checks=[]
    for i,r in enumerate(s['anchors']):
        low=torch.load(Path(r['directory'])/'bank_images.pt',weights_only=True,map_location='cpu')[0].cuda()
        obj=FixedObjective(scorer,low,receipt)
        gate={k:v.detach().cpu().tolist() for k,v in dict(scores=obj.original_scores,active=obj.active,winner=obj.winner,evidence=obj.evidence).items()}
        errors={k:float(np.max(np.abs(np.array(gate[k])-np.array(r['gate'][k])))) for k in ['scores','evidence']}
        equal={k:gate[k]==r['gate'][k] for k in ['active','winner']}
        equal.update({k:errors[k]<=1e-5 for k in errors});equal['low_hash']=thash(low)==r['state_before']['y0']
        row=dict(order=i,index=r['index'],bank_index=r['bank_index'],low_hash=thash(low),expected=r['gate'],actual=gate,equal=equal,errors=errors)
        checks.append(row)
        if not all(equal.values()):break
    ok=len(checks)==60 and all(all(r['equal'].values()) for r in checks)
    validate(binding);validate(inputs)
    result=dict(task='T060-D-R1',status='PASS' if ok else 'BLOCKED',started_utc=start,completed_utc=utc(),selection_sha256=sha(selection),source_bindings=binding,inputs=inputs,checks=checks,source_clean_reads=0,reference_gradient_reads=0,metric_reads=0,target_domain_access=0,official_test_access=0,optimizer_steps=0,physical_gpu=1,gpu=torch.cuda.get_device_name(),reason=None if ok else 'Recomputed gate fails fixed 1e-5 floating tolerance or exact hash/discrete decision check; stopped before any trajectory or source-clean read.')
    atomic_json(out/'preflight.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
