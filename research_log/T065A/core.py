"""Fixed 11 target-free features and one development-only ridge quality head."""
import numpy as np
import torch
from research_log.T063C.core import choose as base_choose
from research_log.T063B.core import summarize
RHO=0.9857470621423519
LAMBDA=1e-3
FEATURES=['step_fraction','objective_progress','one_step_progress','luma_mean','luma_std_population','luma_ge_098','luma_le_002','luma_forward_gradient_ratio','rgb_absolute_change','raw_l2','raw_max_abs']


def luminance(rgb):
    return rgb[:,0]*.299+rgb[:,1]*.587+rgb[:,2]*.114


def gradient(y):
    dh=(y[:,1:,:]-y[:,:-1,:]).abs().reshape(-1)
    dw=(y[:,:,1:]-y[:,:,:-1]).abs().reshape(-1)
    return torch.cat([dh,dw]).mean()


def features(low,images,states,totals):
    x=low.double();denom=max(float(gradient(luminance(x))),1e-8)
    l=np.asarray(totals,dtype=np.float64);reduction=max(float(l[0]-l.min()),1e-8);rows=[]
    for k,(image,state) in enumerate(zip(images,states)):
        y=image.to(x);Y=luminance(y);raw=state.to(x).reshape(-1)
        rows.append([k/27,float((l[0]-l[k])/reduction),0. if k==0 else float((l[k-1]-l[k])/reduction),
            float(Y.mean()),float(Y.std(correction=0)),float((Y>=.98).double().mean()),float((Y<=.02).double().mean()),
            float(gradient(Y))/denom,float((y-x).abs().mean()),float(torch.linalg.vector_norm(raw)),float(raw.abs().max())])
    return np.asarray(rows,dtype=np.float64)


def fit(x,targets):
    x=np.asarray(x,dtype=np.float64);targets=np.asarray(targets,dtype=np.float64)
    mean=x.mean(axis=0);scale=np.maximum(x.std(axis=0,ddof=0),1e-8)
    z=(x-mean)/scale;design=np.column_stack([np.ones(len(x)),z])
    penalty=np.diag([0.]+[LAMBDA]*11)
    beta=np.linalg.solve(design.T@design+penalty,design.T@targets)
    return dict(mean=mean.tolist(),scale=scale.tolist(),intercept=float(beta[0]),coefficients=beta[1:].tolist(),ridge_lambda=LAMBDA,
        objective='sum squared residuals + lambda * sum squared standardized-feature coefficients; unregularized intercept')


def predict(x,model):
    return ((np.asarray(x,dtype=np.float64)-np.asarray(model['mean']))/np.asarray(model['scale']))@np.asarray(model['coefficients'])+model['intercept']


def choose(predictions,base_step):
    return int(np.argmax(np.asarray(predictions)[:base_step+1]))
