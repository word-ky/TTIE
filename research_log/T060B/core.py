"""T060-B: exact CommonRegion2 renderer, gain-only source direction audit."""
import hashlib,json,os,sys
from pathlib import Path
from datetime import datetime,timezone
from types import SimpleNamespace
import numpy as np
import torch
from ttie.common_gain import CommonRegion2

ROOT=Path('/home/wenchang/asdasdsad/wjq/TTIE')
B=ROOT/'runs/20260912-181047-ttie-t014-stage-a-repaired/artifacts/audit'
HEADS={'E':ROOT/'runs/20260918-023558-ttie-t059e-relative/artifacts/T059E/head.pt','014':ROOT/'research_log/T014_energy.pt'}
HEAD_SHA={'E':'e15e91c4e9be401bcac6d3de039ec2d1ae7141fb6cec40c388d22549652d43d0','014':'c3d1eef9f20af163e1268823cf16d3fdaed315e7723fc94db3b1138ad1336521'}
HERE=Path('research_log/T060B')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def thash(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def utc():return datetime.now(timezone.utc).isoformat()
def atomic_json(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as f:json.dump(obj,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
def save_tensor(path,obj):
    with path.open('wb') as f:torch.save(obj,f);f.flush();os.fsync(f.fileno())
def validate(inputs):
    for n,h in inputs.items():assert sha(n)==h,n
def setup():
    torch.manual_seed(7);torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
def firewall(allowed,out):
    allowed={str(Path(n).resolve()) for n in allowed};out=str(out.resolve());root=str(ROOT)
    def audit(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        name=str(Path(os.fsdecode(args[0])).resolve())
        if name.startswith(root+'/') and not(name in allowed or name.startswith(out+'/') or name.startswith(root+'/.venv/')):raise PermissionError('T060B target-free boundary: '+name)
    sys.addaudithook(audit)
def select(anchors):
    table=[dict(index=r['index'],bank_index=r['bank_index'],image_id=r['image_id'],active=r['gate']['active'],included=any(r['gate']['active'])) for r in anchors]
    return table,[r for r in anchors if any(r['gate']['active'])]

class Gain(torch.nn.Module):
    def __init__(self,low,raw,active):
        super().__init__();self.legacy=CommonRegion2(active).to(low).requires_grad_(False)
        with torch.no_grad():self.legacy.raw.copy_(raw)
        self.register_buffer('low',low.detach().clone());self.register_buffer('fixed',raw.detach().clone())
        self.register_buffer('grid',self.legacy.physical_grid()[:,:2].detach().clone())
        self.gain=torch.nn.Parameter(raw[:,2:3].detach().clone());assert torch.count_nonzero(self.gain)==0
    def forward(self):
        raw=torch.cat((self.fixed[:,:2],self.gain),dim=1)
        return torch.func.functional_call(self.legacy,{'raw':raw},(self.low,))

def jacobian(x,gain):
    assert x.shape==(28,) and gain.shape==(1,1,2,2)
    J=x.new_zeros(28,4)
    for i in range(12,20):J[i]=torch.autograd.grad(x[i],gain,retain_graph=True)[0].flatten()
    return J
def stats(values):
    if not values:return None
    a=np.array(values);return {k:float(v) for k,v in dict(mean=a.mean(),median=np.median(a),p10=np.quantile(a,.1),p90=np.quantile(a,.9),min=a.min(),max=a.max()).items()}
def summarize(rows):
    eligible=[r for r in rows if r['nondegenerate']];n=len(eligible);heads={}
    for h in HEADS:
        heads[h]=dict(positive_dot_fraction=sum(r[h]['positive_dot'] for r in eligible)/n if n else 0.,wrong_sign_count=sum(r[h]['dot']<0 for r in eligible),zero_dot_count=sum(r[h]['dot']==0 for r in eligible),cosine=stats([r[h]['cosine'] for r in eligible]),dot=stats([r[h]['dot'] for r in eligible]),norm_all=stats([r[h]['norm'] for r in rows]),norm_nondegenerate=stats([r[h]['norm'] for r in eligible]))
    delta=dict(positive_fraction_E_minus_014=heads['E']['positive_dot_fraction']-heads['014']['positive_dot_fraction'],median_cosine_E_minus_014=heads['E']['cosine']['median']-heads['014']['cosine']['median'] if n else None,wrong_sign_014_minus_E=heads['014']['wrong_sign_count']-heads['E']['wrong_sign_count'])
    gates=dict(positive_E=heads['E']['positive_dot_fraction']>=.90,median_E=n>0 and heads['E']['cosine']['median']>=.60,not_worse_positive=delta['positive_fraction_E_minus_014']>=0,not_worse_median=n>0 and delta['median_cosine_E_minus_014']>=0,material=n>0 and (delta['median_cosine_E_minus_014']>=.05 or delta['wrong_sign_014_minus_E']>=2))
    label='common-gain source direction audit lacks nondegenerate coverage' if n<40 else 'T059-E is a common-gain direction candidate for one later finite-step test' if all(gates.values()) else 'T059-E does not improve the deployed common-gain direction enough to justify integration'
    return dict(rows=len(rows),nondegenerate=n,heads=heads,deltas=delta,gates=gates,reference_norm_all=stats([r['reference_norm'] for r in rows]),reference_norm_nondegenerate=stats([r['reference_norm'] for r in eligible]),classification=label)
