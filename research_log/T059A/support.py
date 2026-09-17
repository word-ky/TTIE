"""Exact detail-feature VJPs, fixed chain gate, immutable Stage-A readback."""
import json,os
from pathlib import Path
import torch
from research_log.T058A_tangent.core import sha,thash,utc
from research_log.T058AE.storage import atomic_json,reopen as reopen_energy,verify_shard0,coverage
from research_log.T058AD.storage import continuity
ROOT=Path('/home/wenchang/asdasdsad/wjq/TTIE')
AE=ROOT/'runs/20260917-135400-ttie-t058ae-stagea'
AD=ROOT/'runs/20260917-131739-ttie-t058ad-shard0'
def verify_stage_a(banks):
    files=json.loads(Path('research_log/T058AE_archives.json').read_bytes())['files']
    for n,h in files.items(): assert sha(AE/n)==h,n
    out=AE/'artifacts/T058AE'; marker=json.loads((out/'complete.json').read_bytes())
    assert sha(out/'complete.json')==sha('research_log/T058AE_complete.json')
    assert marker['combined_rows']==7346
    for n,k in [('combined_reopen.json','combined_reopen_sha256'),('receipt.json','receipt_sha256'),('manifest.json','manifest_sha256')]: assert sha(out/n)==marker[k]
    old=verify_shard0(AD,'research_log/T058AD_archives.json'); new=reopen_energy(out,6322,1024)
    prior=json.loads((out/'receipt.json').read_bytes())
    assert old==prior['shard0_before']==prior['shard0_after']
    old_m=json.loads((AD/'artifacts/T058AD/manifest.json').read_bytes()); new_m=json.loads((out/'manifest.json').read_bytes())
    rows=old_m['rows']+new_m['rows']; combined=coverage(old_m['rows'],new_m['rows'],banks); assert combined['rows']==7346
    gradients=[]
    for base,m in [(AD/'artifacts/T058AD',old_m),(out,new_m)]:
        for chunk in m['chunks']:
            saved=torch.load(base/chunk['file'],map_location='cpu',weights_only=True)
            for index,g in zip(saved['indices'],saved['g_E']):
                assert index==len(gradients) and thash(g)==rows[index]['gradient_sha256']
                assert rows[index]['gradient_dtype']==str(g.dtype) and rows[index]['gradient_shape']==list(g.shape) and rows[index]['finite']
                gradients.append(g)
    receipt=dict(verified_utc=utc(),rows=len(rows),complete_sha256=sha(out/'complete.json'),marker=marker,immutable_shard0=old,new_shard=new,combined=combined,new_shard_files=files,learned_gradient_recomputations=0)
    return rows,gradients,receipt


def detail_jacobian(phi,v):
    # T014 exact evidence VJPs. Gate/evidence constants and frozen legacy grid
    # have exactly zero derivative in the new detail coordinate.
    assert torch.is_grad_enabled() and phi.dtype==v.dtype==torch.float32 and phi.shape==(28,)
    jac=phi.new_zeros(28,64)
    for j in range(12,20):jac[j]=torch.autograd.grad(phi[j],v,retain_graph=True)[0].flatten()
    return jac

def reconstruct(head,phi,jac):
    leaf=phi.detach().clone().requires_grad_(True)
    q,=torch.autograd.grad(head(leaf).squeeze(),leaf)
    return q.detach(),torch.einsum('fi,f->i',jac,q).reshape(1,1,8,8).detach()

def compare(recon,ref):
    a=recon.detach().cpu();b=ref.detach().cpu();delta=a.double()-b.double()
    an=float(a.double().norm());bn=float(b.double().norm());az=bool(torch.count_nonzero(a)==0);bz=bool(torch.count_nonzero(b)==0)
    return dict(allclose=bool(torch.allclose(a,b,atol=2e-6,rtol=2e-5)),atol=2e-6,rtol=2e-5,max_abs=float(delta.abs().max()),l2=float(delta.norm()),cosine=None if an*bn==0 else float((a.double()*b.double()).sum()/(an*bn)),reconstructed_norm=an,energy_norm=bn,reconstructed_zero=az,energy_zero=bz,zero_status_matches=az==bz)

def save_chunk(out,indices,values):
    path=out/f'jacobian_{indices[0]:04d}_{indices[-1]:04d}.pt';tmp=path.with_suffix('.pt.tmp')
    with tmp.open('wb') as f:
        torch.save(dict(indices=indices,**{k:torch.stack([v[k] for v in values]) for k in values[0]}),f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
    return dict(file=path.name,sha256=sha(path),first=indices[0],last=indices[-1],count=len(indices))

def reopen_cache(out,expected):
    m=json.loads((out/'manifest.json').read_bytes());assert len(m['rows'])==expected;order=[]
    for chunk in m['chunks']:
        file=out/chunk['file'];assert sha(file)==chunk['sha256'];saved=torch.load(file,map_location='cpu',weights_only=True)
        assert saved['indices']==list(range(chunk['first'],chunk['last']+1)) and len(saved['indices'])==chunk['count']
        assert saved['J'].shape==(chunk['count'],28,64)
        for pos,index in enumerate(saved['indices']):
            row=m['rows'][index];assert row['index']==index==len(order)
            for key in ['phi','J','q','reconstructed','accepted']:
                t=saved[key][pos];assert t.dtype==torch.float32 and torch.isfinite(t).all() and thash(t)==row[key+'_sha256']
            result=compare(saved['reconstructed'][pos],saved['accepted'][pos]);assert result==row['chain'] and result['allclose'] and result['zero_status_matches']
            order.append(index)
    assert order==list(range(expected))
    return dict(rows=expected,all_tensor_hashes_verified=True,all_chain_results_verified=True,order_exact=True,chunks=len(m['chunks']),manifest_sha256=sha(out/'manifest.json'))
