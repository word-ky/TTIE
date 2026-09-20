import numpy as np
EPS=1e-12

def knee(values,first_safe,interior,motions):
    steps=list(range(first_safe,interior+1));loss=np.asarray(values,dtype=np.float64)[steps]
    d=np.asarray(motions,dtype=np.float64);assert len(d)==len(steps) and d[0]==0
    s=np.cumsum(d,dtype=np.float64);dl=float(loss[0]-loss[-1])
    u=np.clip((loss[0]-loss)/dl,0.,1.) if dl>EPS else None
    v=s/s[-1] if s[-1]>EPS else None
    a=u-v if u is not None and v is not None else None
    if first_safe==interior:chosen=interior;reason='singleton'
    elif dl<=EPS:chosen=interior;reason='objective_degenerate'
    elif s[-1]<=EPS:chosen=interior;reason='motion_degenerate'
    else:chosen=max(zip(a.tolist(),steps))[1];reason='maximum_advantage'
    serial=lambda x:x.tolist() if x is not None else [None]*len(steps)
    return dict(k_FS=first_safe,k_lambda=interior,k_knee=chosen,steps=steps,D_L=dl,u_k=serial(u),d_k=d.tolist(),s_k=s.tolist(),v_k=serial(v),a_k=serial(a),reason=reason)

def gpu_motions(images,first_safe,interior):
    import torch
    values=[0.]
    with torch.no_grad():
        previous=images[first_safe].cuda().double()
        for k in range(first_safe+1,interior+1):
            current=images[k].cuda().double();values.append(float((current-previous).square().mean().sqrt()));previous=current
    return values
