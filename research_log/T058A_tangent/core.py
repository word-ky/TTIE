"""T058 source-only local-detail derivatives, with frozen legacy features."""
from research_log.T039A_tangent.core import (torch,np,json,Path,sha,thash,utc,write,setup,objective,probe_raw,alignment,summarize,LABEL,RECEIPT,MANIFEST,BANK,ENERGY)
from ttie.energy_model import features

def blur(y):
    h,w=y.shape[-2:]; k=(1/16,4/16,6/16,4/16,1/16)
    padded=torch.nn.functional.pad(y,(2,2,0,0),mode='reflect');horizontal=torch.zeros_like(y)
    for i,weight in enumerate(k):horizontal=horizontal+padded[...,i:i+w]*weight
    padded=torch.nn.functional.pad(horizontal,(0,0,2,2),mode='reflect');result=torch.zeros_like(y)
    for i,weight in enumerate(k):result=result+padded[...,i:i+h,:]*weight
    return result

class Detail(torch.nn.Module):
    def __init__(self,y0,active,grid):
        super().__init__();h,w=y0.shape[-2:]
        yy=(torch.arange(h,device=y0.device)>=h//2).long();xx=(torch.arange(w,device=y0.device)>=w//2).long()
        for n,t in dict(y0=y0,detail=y0-blur(y0),grid=grid,mask=active.reshape(2,2)[yy[:,None],xx[None,:]]).items():self.register_buffer(n,t.detach().clone())
        self.v=torch.nn.Parameter(y0.new_zeros(1,1,8,8))
    def render(self,v):
        c=torch.nn.functional.interpolate(v,size=self.y0.shape[-2:],mode='bilinear',align_corners=False).tanh()
        return torch.where(self.mask,(self.y0+c*self.detail).clamp(0,1),self.y0)
    def forward(self):return self.render(self.v)

def energy(model,obj,head,v=None):
    y=model() if v is None else model.render(v)
    phi=features(obj,obj.scorer(y),model.grid)
    return head(phi).squeeze(),y,phi

def fd(model,fn,g):
    direction=(torch.arange(64,device=model.v.device).reshape_as(model.v)%2*2-1).to(model.v)/8
    h=.001
    with torch.no_grad():plus=float(fn(h*direction));minus=float(fn(-h*direction))
    central=(plus-minus)/(2*h);ad=float((g.double()*direction.double()).sum())
    error=abs(central-ad);limit=.0005+.05*abs(ad)
    return dict(step=h,direction='alternating64_unit_l2',plus=plus,minus=minus,central=central,autograd=ad,abs_error=error,tolerance=limit,passed=error<=limit)

def aggregate(rows):
    s=summarize(rows)
    s['degenerate']=len(rows)-s['nondegenerate']
    for k in ['energy_norm','reference_norm','dot']:
        x=np.array([r[k] for r in rows]);s[k+'_summary']={n:float(v) for n,v in dict(min=x.min(),max=x.max(),mean=x.mean(),median=np.median(x),p10=np.quantile(x,.1),p90=np.quantile(x,.9)).items()}
    return s

def classify(s):
    good=s['nondegenerate']>0 and s['positive_dot_fraction']>=.75 and s['cosine_median']>=.50
    return 'frozen-energy detail tangent supported on source' if good else 'frozen-energy detail tangent not ready'
