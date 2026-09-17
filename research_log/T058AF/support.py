"""Frozen Stage-A readback and source-only Stage-B persistence."""
import json,os
from pathlib import Path
import torch
from PIL import Image
from research_log.T058A_tangent.core import sha,thash,utc,alignment,aggregate,classify
from research_log.T058AE.storage import atomic_json,reopen,verify_shard0,coverage

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
    old=verify_shard0(AD,'research_log/T058AD_archives.json'); new=reopen(out,6322,1024)
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

class SourceOpens:
    def __init__(self,images,sources):
        self.allowed={str((images/r['filename']).resolve()):r for r in sources.values()};self.opened=[];self.verified_utc=None;self.original=Image.open
    def __call__(self,path,*args,**kwargs):
        assert self.verified_utc is not None,'STAGE_A_NOT_VERIFIED'
        name=str(Path(path).resolve());assert name in self.allowed,'UNAUTHORIZED_IMAGE'
        row=self.allowed[name];assert row['split']=='train_t014_sobolev' and sha(name)==row['sha256'],'SOURCE_HASH_OR_SPLIT_MISMATCH'
        assert not any(x['path']==name for x in self.opened),'DUPLICATE_SOURCE_OPEN'
        stamp=utc();assert stamp>self.verified_utc
        self.opened.append(dict(path=name,image_id=row['image_id'],filename=row['filename'],split=row['split'],sha256=row['sha256'],utc=stamp))
        return self.original(path,*args,**kwargs)

def reference_loss(output,clean):
    return (output.double()-clean.double()).square().mean()

def save_reference(out,indices,gradients):
    path=out/f'reference_{indices[0]:04d}_{indices[-1]:04d}.pt';tmp=path.with_suffix('.pt.tmp')
    with tmp.open('wb') as f:
        torch.save(dict(indices=indices,g_R=torch.stack(gradients)),f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
    return dict(file=path.name,sha256=sha(path),indices=indices.copy())

def replay(out,learned,expected):
    m=json.loads((out/'manifest.json').read_bytes()); rows=m['rows']; assert len(rows)==expected and [r['index'] for r in rows]==list(range(expected))
    checked=[]
    for chunk in m['chunks']:
        assert sha(out/chunk['file'])==chunk['sha256']
        saved=torch.load(out/chunk['file'],map_location='cpu',weights_only=True);assert saved['indices']==chunk['indices']
        assert saved['g_R'].dtype==torch.float32 and saved['g_R'].shape==(len(saved['indices']),1,1,8,8) and torch.isfinite(saved['g_R']).all()
        for i,g in zip(saved['indices'],saved['g_R']):
            r=rows[i];assert i==len(checked) and thash(g)==r['reference_sha256'] and thash(learned[i])==r['energy_sha256']
            metric=alignment(learned[i],g,torch.ones_like(g,dtype=torch.bool))
            for k,v in metric.items():assert r[k]==v,(i,k)
            checked.append(r)
    assert len(checked)==expected
    return dict(rows=expected,reference_hashes_and_alignment_verified=True,canonical_order_exact=True,manifest_sha256=sha(out/'manifest.json'))

def summary(rows):
    overall=aggregate(rows)
    return dict(label='SOURCE_REFERENCE_GRADIENT_DIAGNOSTIC_ONLY',overall=overall,classification=classify(overall),thresholds=dict(positive_dot_fraction=.75,cosine_median=.50),stratifications={k:{str(v):aggregate([r for r in rows if r[k]==v]) for v in sorted({r[k] for r in rows})} for k in ['condition','image_id','state_name']},stratifications_used_for_gating=False)
