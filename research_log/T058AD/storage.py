"""Checkpointed Stage-A gradient chunks and independent readback."""
import os,json
from pathlib import Path
import torch
from research_log.T058A_tangent.core import sha,thash
from research_log.T058AC.chain import vector_checks

def atomic_json(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w',encoding='utf8') as f:
        json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def continuity(g,ref):
    result=vector_checks(g,ref,ref)
    checks={k:result['criteria'][k] for k in ['float32_float64','orientation']}
    checks['finite']=dict(passed=bool(torch.isfinite(g).all() and torch.isfinite(ref).all()))
    return dict(norm_current=result['norm32'],norm_reference=result['norm64'],comparison_copies=result['comparison_copies'],criteria=checks,passed=all(c['passed'] for c in checks.values()))

def save_chunk(out,indices,gradients):
    path=out/f'gradients_{indices[0]:04d}_{indices[-1]:04d}.pt';tmp=path.with_suffix('.pt.tmp')
    with tmp.open('wb') as f:
        torch.save(dict(indices=indices,g_E=torch.stack(gradients)),f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
    return dict(file=path.name,sha256=sha(path),first=indices[0],last=indices[-1],count=len(indices))

def reopen(out,expected):
    out=Path(out);manifest=json.loads((out/'manifest.json').read_bytes());rows=manifest['rows'];order=[]
    assert len(rows)==expected and [x['index'] for x in rows]==list(range(expected))
    for item in manifest['chunks']:
        file=out/item['file'];assert sha(file)==item['sha256']
        saved=torch.load(file,map_location='cpu',weights_only=True);grad=saved['g_E']
        assert grad.dtype==torch.float32 and grad.shape==(len(saved['indices']),1,1,8,8) and torch.isfinite(grad).all()
        assert len(saved['indices'])==item['count'] and saved['indices']==list(range(item['first'],item['last']+1))
        for i,g in zip(saved['indices'],grad):
            assert thash(g)==rows[i]['gradient_sha256'];assert float(g.double().norm())==rows[i]['gradient_norm'];order.append(i)
    assert order==list(range(expected))
    return dict(rows=expected,execution_order_exact=True,all_finite=True,all_gradient_hashes_match=True,all_norms_match=True,manifest_sha256=sha(out/'manifest.json'),chunks=len(manifest['chunks']))
