"""Frozen numerical diagnostics only; no model, renderer or optimizer objects."""
import numpy as np
from scipy.stats import spearmanr

def diagnose(g,v,ref,mse0,mse1):
    g=np.asarray(g,dtype=np.float64).reshape(-1);v=np.asarray(v,dtype=np.float64).reshape(-1);ref=np.asarray(ref,dtype=np.float64).reshape(-1)
    assert g.shape==v.shape==ref.shape==(64,)
    # Bias-corrected first Adam moment: mhat=g, vhat=g^2; accepted lr and eps.
    replay=-.05*g/(np.sqrt(g*g)+1e-8);error=float(np.max(np.abs(replay-v)))
    assert error<1e-7,'INHERITED_ADAM_REPLAY_TOLERANCE'
    active=np.abs(g)>1e-8;assert active.any(),'NO_COORDINATES_ABOVE_ADAM_EPS'
    L=float(np.dot(ref,v));A=float(mse1-mse0);R=A-L
    values=dict(L=L,A=A,R=R,predicted_gradient_norm=float(np.linalg.norm(g)),reference_gradient_norm=float(np.linalg.norm(ref)),step_norm=float(np.linalg.norm(v)),saturation_fraction=float(np.mean(np.abs(v[active])/.05>=.9)),adam_replay_max_error=error)
    assert all(np.isfinite(x) for x in values.values())
    return dict(**values,linear_descent=L<0,actual_harm=A>0,overshoot_flip=L<0 and A>0,coordinates_above_eps=int(active.sum()))

def classify(linear,total,flips,harmful,saturation):
    if linear/total<.90:return 'transferred detail direction itself is insufficient; close the direct detail-step branch'
    if flips/harmful<.5 or saturation<.80:return 'optimizer-scale/curvature mismatch not established; close the direct detail-step branch'
    return 'optimizer-scale/curvature mismatch is supported as a source-only diagnosis'

def summarize(rows):
    n=len(rows);linear=sum(r['linear_descent'] for r in rows);harm=sum(r['actual_harm'] for r in rows);flips=sum(r['overshoot_flip'] for r in rows)
    keys=['L','A','R','predicted_gradient_norm','reference_gradient_norm','step_norm','saturation_fraction']
    distributions={k:{name:float(np.quantile([r[k] for r in rows],q)) for name,q in [('median',.5),('p10',.1),('p90',.9)]} for k in keys}
    correlations={k:float(spearmanr([r[k] for r in rows],[r['step_norm'] for r in rows]).statistic) for k in ['predicted_gradient_norm','reference_gradient_norm']};assert all(np.isfinite(v) for v in correlations.values())
    return dict(eligible_anchors=n,linear_descent=linear,linear_descent_fraction=linear/n,actual_harm=harm,overshoot_flip=flips,overshoot_fraction_of_harm=flips/harm,statistics=distributions,spearman_with_step_norm=correlations,adam_replay_max_error=max(r['adam_replay_max_error'] for r in rows),classification=classify(linear,n,flips,harm,distributions['saturation_fraction']['median']))
