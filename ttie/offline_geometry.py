"""T009 DEVELOPMENT ONLY: reference gradients, trajectory labels and oracle.

Never imported by semantic_ttt.py. Clean references appear only in this module's
offline APIs after a semantic trajectory/action has been finalized.
"""
import math
import torch
from torch.nn import functional as F
from .semantic_ttt import EVGamma, CoordinateISP

SURFACE_EV=(-1.25,-1.,-.75,-.5,-.25,0.,.25,.5,.75,1.,1.25)
SURFACE_GAMMA=(.8,.9,1.,1.1,1.25)


class Piecewise2(EVGamma):
    def __init__(self):super().__init__(2)
    def parameter_field(self,size):
        h,w=size
        y=(torch.arange(h,device=self.raw.device)>=h//2).long()
        x=(torch.arange(w,device=self.raw.device)>=w//2).long()
        return self.physical_grid()[:,:,y[:,None],x[None,:]]


def coordinate_model(size,coordinates):
    return EVGamma(size) if coordinates=='ev_gamma' else CoordinateISP(size,coordinates)


def region_mask(image,condition):
    h,w=image.shape[-2:]
    y,x=torch.meshgrid(torch.arange(h,device=image.device),torch.arange(w,device=image.device),indexing='ij')
    return x<w//2 if condition=='left_right' else ((x>=w//2).int()+(y>=h//2).int())%2==0


def reference_gradient(image, clean, *, size=2, coordinates='ev_gamma', mask=None):
    model=coordinate_model(size,coordinates).to(image)
    square=(model(image.detach())-clean.detach()).square()
    loss=square.mean() if mask is None else square[...,mask].mean()
    gradient,=torch.autograd.grad(loss,model.raw)
    return gradient.detach()


def cosine(a,b):
    a=a.flatten().double();b=b.flatten().double()
    denom=a.norm()*b.norm()
    return float(a.dot(b)/denom) if denom>0 else None


def gradient_alignment(image,clean,objective,*,size,coordinates,condition):
    model=coordinate_model(size,coordinates).to(image)
    sem,=torch.autograd.grad(objective(model(image)),model.raw)
    ref=reference_gradient(image,clean,size=size,coordinates=coordinates)
    result=dict(cosine=cosine(sem,ref),semantic_gradient=sem.detach().cpu().tolist(),reference_gradient=ref.cpu().tolist())
    for channel,name in enumerate(('ev','gamma') if coordinates=='ev_gamma' else ('ev',) if coordinates=='ev_only' else ('gamma',)):
        a=sem[:,channel];b=ref[:,channel]
        result[name]=dict(sign_agreement=float((a.sign()==b.sign()).float().mean()),
                          semantic_abs_sum=float(a.abs().sum()),reference_abs_sum=float(b.abs().sum()))
    if condition in ('left_right','quadrants'):
        mask=region_mask(image,condition)
        result['dark_cosine']=cosine(sem,reference_gradient(image,clean,size=size,coordinates=coordinates,mask=mask))
        result['bright_cosine']=cosine(sem,reference_gradient(image,clean,size=size,coordinates=coordinates,mask=~mask))
    return result


def attach_trajectory_mse(image,clean,result,*,size,coordinates,condition):
    model=coordinate_model(size,coordinates).to(image)
    rows=[]
    with torch.no_grad():
        for step,raw in enumerate(result['states']):
            model.raw.copy_(raw.to(image))
            output=image if result['diagnostics']['stop_reason']=='no_active' else model(image)
            square=(output-clean).square()
            item=dict(step=step,semantic_loss=result['diagnostics']['loss_trajectory'][step],mse=float(square.mean()))
            if condition in ('left_right','quadrants'):
                mask=region_mask(image,condition)
                item.update(dark_mse=float(square[...,mask].mean()),bright_mse=float(square[...,~mask].mean()))
            rows.append(item)
    return rows


def rank(values):
    order=sorted(range(len(values)),key=values.__getitem__);ranks=[0.]*len(values)
    start=0
    while start<len(order):
        end=start+1
        while end<len(order) and values[order[end]]==values[order[start]]:end+=1
        for i in order[start:end]:ranks[i]=(start+end-1)/2
        start=end
    return ranks


def correlation(a,b):
    if len(a)<2:return None
    aa=torch.tensor(a,dtype=torch.float64);bb=torch.tensor(b,dtype=torch.float64)
    return cosine(aa-aa.mean(),bb-bb.mean())


def surface_semantics(image,objective):
    # No clean reference or condition argument: score the literal grid first.
    model=EVGamma(1).to(image);rows=[]
    with torch.no_grad():
        for ev in SURFACE_EV:
            for gamma in SURFACE_GAMMA:
                model.set_grid(image.new_tensor([ev,gamma]).reshape(1,2,1,1))
                rows.append(dict(ev=ev,gamma=gamma,semantic_loss=float(objective(model(image)))))
    return rows


def label_surface(image,clean,rows,*,condition):
    model=EVGamma(1).to(image);labeled=[]
    with torch.no_grad():
        for row in rows:
            model.set_grid(image.new_tensor([row['ev'],row['gamma']]).reshape(1,2,1,1))
            labeled.append(dict(row,mse=float((model(image)-clean).square().mean())))
    tie=lambda r:(abs(r['ev'])+abs(r['gamma']-1),abs(r['ev']),abs(r['gamma']-1),r['ev'],r['gamma'])
    sem=min(labeled,key=lambda r:(r['semantic_loss'],*tie(r)))
    ref=min(labeled,key=lambda r:(r['mse'],*tie(r)))
    sign=1 if condition=='homogeneous_dark' else -1
    direct=next(r for r in labeled if r['ev']==sign*.5 and r['gamma']==1)
    path=sorted([r for r in labeled if r['gamma']==1 and r['ev']*sign>=0],key=lambda r:abs(r['ev']))
    cross=next((r for r in path if r['semantic_loss']<=1e-8),None)
    best=min(path,key=lambda r:(r['mse'],abs(r['ev'])))
    delta=None if cross is None else abs(cross['ev'])-abs(best['ev'])
    return labeled,dict(semantic_argmin=sem,mse_argmin=ref,ev_distance=abs(sem['ev']-ref['ev']),gamma_distance=abs(sem['gamma']-ref['gamma']),
      action_euclidean_distance=math.hypot(sem['ev']-ref['ev'],sem['gamma']-ref['gamma']),
      spearman=correlation(rank([r['semantic_loss'] for r in labeled]),rank([r['mse'] for r in labeled])),
      semantic_argmin_worse_than_fixed_direct=sem['mse']>direct['mse'],fixed_direct_mse=direct['mse'],
      crossing_gamma1=cross,mse_optimal_gamma1=best,crossing_category='none' if delta is None else 'near' if abs(delta)<=.25 else 'before' if delta<0 else 'beyond')


def oracle_renderer(image,clean,renderer,*,steps=100):
    """Development-only reference optimizer, never a deployable policy."""
    model=(EVGamma(2) if renderer=='bilinear2' else Piecewise2()).to(image)
    optimizer=torch.optim.Adam([model.raw],lr=.03);losses=[]
    for step in range(steps+1):
        output=model(image.detach());loss=(output-clean.detach()).square().mean();losses.append(float(loss.detach()))
        if step==steps:break
        optimizer.zero_grad(set_to_none=True)
        gradient,=torch.autograd.grad(loss,model.raw);model.raw.grad=gradient;optimizer.step()
    return dict(image=output.detach(),raw=model.raw.detach(),grid=model.physical_grid().detach()[:,:2],
                diagnostic_only=True,loss_trajectory=losses,steps=steps,initial_raw_zero=True)
