"""Primitive score and actual target-free gain equivalence, before trajectories."""
import argparse
from ttie.clip_signal import FrozenCLIP
from ttie.learned_prototypes import Prototypes
from ttie.semantic_ttt import SemanticScorer,FixedObjective
from ttie.common_gain import CommonRegion2
from ttie.energy_model import load_energy,features
from research_log.T060B.core import *

HERE=Path('research_log/T060DR2')
PRED=ROOT/'runs/20260919-114102-ttie-t060b-common/artifacts/T060B'

def models():
    c=json.loads((B/'config.json').read_bytes())
    encoder=FrozenCLIP.from_checkpoint(c['model_identity']['path'],'cuda:0')
    proto=ROOT/'research_log/remote_runs/20260912-071126-ttie-t006-a6000/artifacts/audit/prototypes.pt'
    scorer=SemanticScorer(encoder,Prototypes(torch.load(proto,weights_only=True,map_location='cuda:0')['raw'])).eval().requires_grad_(False)
    return scorer,c['frozen_receipt'],load_energy(HEADS['014']).cuda(),load_energy(HEADS['E'])

def fresh_gain(low,obj,scorer,head):
    model=CommonRegion2(obj.active).to(low)
    x=features(obj,scorer(model(low)),model.physical_grid()[:,:2]);J=x.new_zeros(28,4)
    for j in range(12,20):J[j]=torch.autograd.grad(x[j],model.raw,retain_graph=True)[0][:,2:3].flatten()
    leaf=x.detach().cpu().requires_grad_(True);q,=torch.autograd.grad(head(leaf).sum(),leaf)
    gain=torch.einsum('fi,f->i',J.detach().cpu(),q).detach()
    return gain,dict(x=x.detach().cpu(),J=J.detach().cpu(),q=q.detach(),gain=gain)

def comparison(fresh,frozen):
    x=fresh.double().flatten();y=frozen.double().flatten();nx=float(x.norm());ny=float(y.norm())
    if nx==0 or ny==0:
        return dict(fresh_norm=nx,frozen_norm=ny,zero_convention='both exact zero pass; one zero fails',cosine=None,relative_l2=None,passed=nx==ny==0)
    cosine=float(torch.dot(x,y)/(x.norm()*y.norm()));relative=float((x-y).norm()/y.norm())
    return dict(fresh_norm=nx,frozen_norm=ny,zero_convention='nonzero',cosine=cosine,relative_l2=relative,passed=cosine>=.999 and relative<=.01)

def main(out):
    out.mkdir(parents=True,exist_ok=False);setup();start=utc()
    selection=Path('research_log/T060B/selection.json');assert sha(selection)=='494fdafcf0db462d7d8157175bd50a879217f6e5f0c973c9af833b50ac726297'
    s=json.loads(selection.read_bytes());assert len(s['anchors'])==60
    binding=json.loads((HERE/'source_binding.json').read_bytes());frozen=json.loads((HERE/'frozen_prediction_manifest.json').read_bytes())
    inputs={n:h for n,h in s['inputs'].items() if n.endswith('config.json') or n.endswith('open_clip_pytorch_model.bin') or n.endswith('prototypes.pt') or any(n==r['directory']+'/bank_images.pt' for r in s['anchors'])}
    inputs.update({str(v):HEAD_SHA[k] for k,v in HEADS.items()})
    inputs.update({str(PRED/n):h for n,h in frozen['files'].items()})
    firewall(set(inputs)|{str(Path(n).resolve()) for n in binding}|{str((HERE/'source_binding.json').resolve())},out)
    validate(binding);validate(inputs)
    scorer,receipt,head014,headE=models();assert receipt['calibration']==s['calibration'];before={h:{k:thash(t) for k,t in head.state_dict().items()} for h,head in [('014',head014),('E',headE)]}
    records=json.loads((PRED/'predictions.json').read_bytes());assert [r['bank_index'] for r in records]==[r['bank_index'] for r in s['anchors']]
    checks=[]
    for i,(r,old) in enumerate(zip(s['anchors'],records)):
        low=torch.load(Path(r['directory'])/'bank_images.pt',weights_only=True,map_location='cpu')[0].cuda();obj=FixedObjective(scorer,low,receipt)
        gate={k:v.detach().cpu().tolist() for k,v in dict(scores=obj.original_scores,active=obj.active,winner=obj.winner,evidence=obj.evidence).items()}
        errors={k:float(np.max(np.abs(np.array(gate[k])-np.array(r['gate'][k])))) for k in ['scores','evidence']}
        equal={k:gate[k]==r['gate'][k] for k in ['active','winner']};equal['scores']=errors['scores']<=1e-5;equal['low_hash']=thash(low)==r['state_before']['y0']
        # Evidence is computed by the unchanged FixedObjective/both_gates, not copied.
        expected=(obj.original_scores.double().gather(1,obj.winner[:,None]).squeeze(1)-obj.evidence.new_tensor(receipt['calibration']['tau'])[obj.winner])/obj.evidence.new_tensor(receipt['calibration']['scale'])[obj.winner]
        assert torch.equal(expected,obj.evidence)
        row=dict(order=i,index=r['index'],bank_index=r['bank_index'],image_id=r['image_id'],low_hash=thash(low),expected=r['gate'],actual=gate,equal=equal,errors=errors)
        if all(equal.values()):
            saved=torch.load(PRED/old['file'],weights_only=True,map_location='cpu');assert thash(saved['low'])==thash(low)
            fresh,trace=fresh_gain(low,obj,scorer,headE);assert torch.isfinite(fresh).all();check=comparison(fresh,saved['g_E']);row['gradient']=check;equal['gradient']=check['passed']
            trace['frozen_gain']=saved['g_E'];name=f'{i:03d}.pt';save_tensor(out/name,trace);row['gradient_file']=name;row['gradient_sha256']=sha(out/name)
        checks.append(row)
        print(json.dumps(dict(order=i,bank=r['bank_index'],passed=all(equal.values()),gradient=row.get('gradient'))),flush=True)
        if not all(equal.values()):break
    ok=len(checks)==60 and all(all(r['equal'].values()) for r in checks)
    validate(binding);validate(inputs);after={h:{k:thash(t) for k,t in head.state_dict().items()} for h,head in [('014',head014),('E',headE)]};assert before==after
    result=dict(task='T060-D-R2',status='PASS' if ok else 'BLOCKED',started_utc=start,completed_utc=utc(),selection_sha256=sha(selection),source_bindings=binding,inputs=inputs,head_state_before=before,head_state_after=after,checks=checks,source_clean_reads=0,reference_gradient_reads=0,metric_reads=0,target_domain_access=0,official_test_access=0,optimizer_steps=0,physical_gpu=1,gpu=torch.cuda.get_device_name(),reason=None if ok else 'Primitive score/hash/discrete gate or downstream gain-gradient equivalence failed; stopped before trajectories.')
    atomic_json(out/'preflight.json',result);print('PREFLIGHT_'+result['status'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
