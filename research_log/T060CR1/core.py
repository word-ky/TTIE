"""One gain-gradient call substitution in the literal accepted T036 trajectory."""
import ast,json,hashlib
from pathlib import Path
import torch
import numpy as np
from ttie import common_gain_ttt as accepted
from research_log.T060B.core import sha,thash,utc,atomic_json,save_tensor,validate,firewall,ROOT,HEADS,HEAD_SHA,setup,stats
HERE=Path('research_log/T060CR1')
BASE=ROOT/'runs/20260915-035341-ttie-t036a-common/artifacts/audit'
LOW=ROOT/'shared/t036a/low'
NORMAL=ROOT/'shared/t036a/normal'

def trajectory(image,scorer,receipt,head014,headE):
    source=Path(__file__).with_name('pinned_trajectory.py.txt').read_bytes()
    assert hashlib.sha256(source).hexdigest()==sha(accepted.__file__)
    tree=ast.parse(source);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='trajectory')
    calls=[n for n in ast.walk(node) if isinstance(n,ast.Call) and ast.unparse(n.func)=='torch.autograd.grad'];assert len(calls)==1
    call=calls[0];assert ast.unparse(call)=='torch.autograd.grad(value, model.raw)'
    call.func=ast.Name(id='gain_substitution',ctx=ast.Load());call.args.append(ast.Name(id='f',ctx=ast.Load()))
    traces=[]
    def gain_substitution(value,raw,f):
        g014,=torch.autograd.grad(value,raw,retain_graph=True)
        J=f.new_zeros(28,4)
        for j in range(12,20):J[j]=torch.autograd.grad(f[j],raw,retain_graph=True)[0][:,2:3].flatten()
        leaf=f.detach().cpu().requires_grad_(True);q,=torch.autograd.grad(headE(leaf).sum(),leaf)
        J=J.detach().cpu();gain=torch.einsum('fi,f->i',J,q).reshape(1,1,2,2)
        hybrid=g014.detach().clone();hybrid[:,2:3]=gain.to(hybrid)
        values=dict(x=f.detach().cpu(),J_gain=J,q_E=q.detach(),g014=g014.detach().cpu(),g_E_gain=gain.detach(),hybrid=hybrid.cpu())
        assert all(torch.isfinite(t).all() for t in values.values());traces.append(values)
        return (hybrid,)
    scope=dict(vars(accepted));scope['gain_substitution']=gain_substitution
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'literal_T036_with_gain_substitution','exec'),scope)
    result,t,decision=scope['trajectory'](image,scorer,receipt,head014,basis='region2',max_steps=40)
    assert len(traces)==t['diagnostics']['steps'];return result,t,decision,traces

def summarize(rows):
    metrics={k:stats([r[k] for r in rows]) for k in rows[0] if k not in ['index','low']}
    counts={m:dict(improve=sum(r[m]>0 for r in rows),regress=sum(r[m]<0 for r in rows),tie=sum(r[m]==0 for r in rows)) for m in ['vs026_psnr','vs026_ssim','vs036_psnr','vs036_ssim']}
    worst=min(rows,key=lambda r:(r['vs026_psnr'],r['index']))
    gates=dict(mean_psnr=metrics['vs026_psnr']['mean']>=.80,regressions=counts['vs026_psnr']['regress']<=20,worst_regression=worst['vs026_psnr']>=-3.,mean_ssim=metrics['vs026_ssim']['mean']>=.006)
    label='T059-E gain-slice substitution meets fixed T036 finite-step criteria' if all(gates.values()) else 'T059-E gain-direction advantage does not translate into a sufficiently safe/material fixed T036 trajectory improvement'
    return dict(rows=len(rows),metrics=metrics,counts=counts,worst_index=worst['index'],worst_gain=worst['vs026_psnr'],gates=gates,classification=label)
