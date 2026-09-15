"""Fixed source-bank tangent audit, with accepted T029 scalar convention."""
from research_log.T029A_alignment.common import alignment,summarize,thash,utc,write
import json,hashlib,math
from pathlib import Path
from types import SimpleNamespace
import torch
import numpy as np
LABEL='SOURCE_REFERENCE_GRADIENT_DIAGNOSTIC_ONLY'
RECEIPT='d4379330bb937bd30345f657f90743f0b23f077e49a93d7fb3c51e6bdfc92de2'
MANIFEST='4dbf8c604bc44574a5823ec8e22142c8ac39574b6103c4bbd0a69647f15a2257'
BANK='92fd60d01ca0780880d724464c4d0a76ba1721ab5b8a2d054d4035a141673125'
ENERGY='c3d1eef9f20af163e1268823cf16d3fdaed315e7723fc94db3b1138ad1336521'
GAINS=[.75,1.,1.25]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def setup():
    torch.manual_seed(7);torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.is_available()
def masks(active):
    total=active.reshape(1,1,2,2).expand(1,3,2,2).clone();legacy=total.clone();legacy[:,2]=False;gain=total.clone();gain[:,:2]=False
    return dict(legacy=legacy,gain=gain,total=total)
def probe_raw(legacy,active,gain):
    channel=torch.full_like(legacy[:,:1],math.atanh(math.log(gain)/math.log(2)))
    channel.masked_fill_(~active.reshape(1,1,2,2),0.)
    return torch.cat([legacy,channel],1)
def objective(scorer,gate,calibration,device):
    return SimpleNamespace(scorer=scorer,calibration=calibration,active=torch.tensor(gate['active'],dtype=torch.bool,device=device),winner=torch.tensor(gate['winner'],device=device),evidence=torch.tensor(gate['evidence'],dtype=torch.float64,device=device))
def aggregate(rows):
    return {g:dict(**summarize([r['groups'][g] for r in rows]),**{k+'_median':float(np.median([r['groups'][g][k] for r in rows])) for k in ['energy_norm','reference_norm','dot']}) for g in ['legacy','gain','total']}
def classify(s):
    a,b=s['legacy'],s['gain']
    yes=(a['cosine_median'] is not None and b['cosine_median'] is not None and a['positive_dot_fraction']-b['positive_dot_fraction']>=.2 and a['cosine_median']-b['cosine_median']>=.25)
    return 'source gain-tangent deficit supported' if yes else 'source gain-tangent deficit not supported / real-domain effect remains plausible'
