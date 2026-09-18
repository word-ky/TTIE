"""One inherited local-detail Adam step; action helpers have no reference input."""
import ast,json,os
from pathlib import Path
import torch
from research_log.T058A_tangent.core import sha,thash,utc,Detail,probe_raw
from research_log.T059A.support import atomic_json
ROOT=Path('/home/wenchang/asdasdsad/wjq/TTIE')
E=ROOT/'runs/20260918-023558-ttie-t059e-relative/artifacts/T059E'
B=ROOT/'runs/20260912-181047-ttie-t014-stage-a-repaired/artifacts/audit'
JR=ROOT/'runs/20260917-165444-ttie-t059a-jacobian'
RR=ROOT/'runs/20260917-153308-ttie-t058af-stageb'
HEAD='e15e91c4e9be401bcac6d3de039ec2d1ae7141fb6cec40c388d22549652d43d0'
# Compile the literal accepted methods, without importing the target-domain oracle.
pinned=Path(__file__).with_name('pinned_t054_core.py.txt')
assert sha(pinned)=='e56f359bc353b1d5ccf374e0980a27780e827d67fc9cc36b78dfaf6887100ce1'
tree=ast.parse(pinned.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Detail')
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['blur','optimize']]
nodes += [n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ['coefficient','forward']]
namespace={'torch':torch};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(pinned),'exec'),namespace)
optimizer_node=next(n for n in ast.walk(next(n for n in nodes if n.name=='optimize')) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='optimizer' for t in n.targets))
optimizer_code=compile(ast.Module(body=[optimizer_node],type_ignores=[]),str(pinned),'exec')
class Bridge(Detail):
    coefficient=namespace['coefficient']
    forward=namespace['forward']

def step(model,gradient):
    assert torch.count_nonzero(model.v)==0 and model.v.grad is None
    scope={'torch':torch,'model':model};exec(optimizer_code,scope)
    optimizer=scope['optimizer'];optimizer.zero_grad();model.v.grad=gradient.to(model.v).reshape_as(model.v).clone();optimizer.step()
    return optimizer

def save_tensor(path,value):
    with path.open('wb') as f:torch.save(value,f);f.flush();os.fsync(f.fileno())

def stats(rows):
    eligible=[r for r in rows if r['eligible']]
    def group(values):
        rel=torch.tensor([r['relative_mse_change'] for r in values],dtype=torch.float64)
        return dict(n=len(values),wins=sum(r['mse_change']<0 for r in values),equals=sum(r['mse_change']==0 for r in values),losses=sum(r['mse_change']>0 for r in values),mean_relative_change=float(rel.mean()),median_relative_change=float(torch.quantile(rel,.5)),p90_relative_harm=float(torch.quantile(rel.clamp_min(0),.9)),max_relative_harm=float(rel.clamp_min(0).max()))
    result=group(eligible);result['win_fraction']=result['wins']/result['n']
    positive=result['win_fraction']>=.75 and result['median_relative_change']<0 and result['mean_relative_change']<0
    return dict(eligible=result,all_banks=group(rows),per_image={str(i):group([r for r in rows if r['image_id']==i]) for i in sorted({r['image_id'] for r in rows})},classification='one-step detail-direction transfer is supported as a source-only mechanism candidate' if positive else 'one-step detail-direction transfer is not supported; do not extend to multi-step detail TTT')
