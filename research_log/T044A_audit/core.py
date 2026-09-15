"""The sole predeclared low-only legacy displacement, using accepted mapping."""
import hashlib,json
from pathlib import Path
from datetime import datetime,timezone
import torch
from ttie.isp import physical_parameters

COHORT='279c74b335999d6631d436c79ea80e9e2cccfc4b97cb7c0b992829d04cfaf40b'
FREEZE='46e667ded785e4f3f8ba341d40d00a7e02ca652e2778fcc7ead1750665161be4'
DEFINITION='sqrt(mean(concat(((EV_selected-EV_step10)/4)^2,(log2(gamma_selected/gamma_step10)/2)^2) over active Region2 coordinates)); zero active => 0'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def thash(t):return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()
def utc():return datetime.now(timezone.utc).isoformat()
def write(p,d):Path(p).write_text(json.dumps(d,indent=2)+'\n')
def physical(raw):
    # Same accepted renderer map; float64 makes the scalar audit reproducible.
    x=raw[:,:2].double()
    return physical_parameters(torch.cat([x,torch.zeros_like(x).repeat(1,2,1,1)],1))[:,:2]
def score(anchor,selected,active):
    active=torch.as_tensor(active,dtype=torch.bool).reshape(2,2)
    if not active.any():return 0.
    a,b=physical(anchor),physical(selected)
    ev=(b[0,0]-a[0,0])/4
    gamma=torch.log2(b[0,1]/a[0,1])/2
    return float(torch.cat([ev[active],gamma[active]]).square().mean().sqrt())
