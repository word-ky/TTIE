import numpy as np
import torch

STEPS=(.004,.002,.001,.0005)

def ulp(value):
    x=np.float32(value)
    return float(max(abs(np.nextafter(x,np.float32(np.inf))-x),abs(x-np.nextafter(x,np.float32(-np.inf)))))

def boundaries(y,mask):
    values=y.detach().cpu().numpy();active=np.broadcast_to(mask.detach().cpu().numpy(),values.shape);x=values[active]
    zero=np.nextafter(np.float32(0),np.float32(1));lo=np.nextafter(np.float32(1),np.float32(0));hi=np.nextafter(np.float32(1),np.float32(np.inf))
    return dict(active_rgb_elements=len(x),exact_zero=int(np.sum(x==0)),exact_one=int(np.sum(x==1)),within_one_ulp_zero=int(np.sum(np.abs(x)<=zero)),within_one_ulp_one=int(np.sum((x>=lo)&(x<=hi))))

def crossings(model,v):
    with torch.no_grad():
        c=torch.nn.functional.interpolate(v,size=model.y0.shape[-2:],mode='bilinear',align_corners=False).tanh()
        pre=model.y0+c*model.detail;zero=(model.y0<0)|(model.y0>1);now=(pre<0)|(pre>1);mask=model.mask.expand_as(pre)
        return dict(changed=int(((now!=zero)&mask).sum()),below_zero=int(((pre<0)&mask).sum()),above_one=int(((pre>1)&mask).sum()))

def agreement(rev,fwd):return abs(rev-fwd)<=1e-6+1e-4*abs(rev)
