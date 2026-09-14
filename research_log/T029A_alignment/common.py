"""Frozen-state T029-A diagnostics; no deployable code is modified."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from PIL import Image
from ttie.semantic_ttt import Region2
from ttie.gamma_range_box import Gamma05Box

LABEL = 'REFERENCE_GRADIENT_DIAGNOSTIC_ONLY'
SPLIT_SHA = 'b88c8347005984b5523b117b52c0c068672fe172eb7c9aa5b60102d350e2d85b'
ENERGY_SHA = 'c3d1eef9f20af163e1268823cf16d3fdaed315e7723fc94db3b1138ad1336521'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def thash(t): return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def utc(): return datetime.now(timezone.utc).isoformat()
def write(p,v): Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def setup():
    torch.manual_seed(7); torch.set_num_threads(1)
    torch.use_deterministic_algorithms(False)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.is_available()
def native(p):
    with Image.open(p) as im: a=np.asarray(im.convert('RGB')).copy()
    assert a.shape==(400,600,3)
    return (torch.from_numpy(a).permute(2,0,1).unsqueeze(0).float()/255.).cuda()
def model_for(decision,low):
    obj=SimpleNamespace(active=torch.tensor(decision['gate']['active'],device=low.device,dtype=torch.bool),
                        winner=torch.tensor(decision['gate']['winner'],device=low.device))
    model=Region2(obj.active).to(low);box=Gamma05Box(obj,2)
    for k in ['lower','upper']:
        assert torch.equal(getattr(box,k).cpu(),torch.tensor(decision['diagnostics']['action_box'][k]))
    return model,box
def alignment(e,r,mask):
    a=e.detach().double().flatten()[mask.flatten()];b=r.detach().double().flatten()[mask.flatten()]
    en=float(a.norm());rn=float(b.norm());dot=float(a@b);deg=en<=1e-12 or rn<=1e-12
    return dict(active_count=a.numel(),energy_norm=en,reference_norm=rn,dot=dot,
                cosine=None if deg else dot/(en*rn),degenerate=deg,positive_dot=dot>0)
def summarize(rows):
    valid=[r for r in rows if not r['degenerate']];c=np.array([r['cosine'] for r in valid])
    return dict(count=len(rows),nondegenerate=len(valid),cosine_mean=float(c.mean()) if len(c) else None,
        cosine_median=float(np.median(c)) if len(c) else None,
        cosine_p10=float(np.quantile(c,.1)) if len(c) else None,cosine_p90=float(np.quantile(c,.9)) if len(c) else None,
        positive_dot_fraction=sum(r['positive_dot'] for r in valid)/len(valid) if valid else None,
        energy_zero_fraction=sum(r['energy_norm']<=1e-12 for r in rows)/len(rows),
        reference_zero_fraction=sum(r['reference_norm']<=1e-12 for r in rows)/len(rows),
        either_zero_fraction=sum(r['degenerate'] for r in rows)/len(rows))
def classify(s):
    if not s['nondegenerate']:return 'structurally blocked'
    if s['cosine_median']<=0 or s['positive_dot_fraction']<.5:return 'strong field-direction mismatch'
    if s['cosine_median']<.25 or s['positive_dot_fraction']<.75:return 'weak/mixed field alignment'
    return 'broad field-direction alignment'
