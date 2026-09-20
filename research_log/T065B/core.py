"""One fixed class-balanced logistic head; exact T065-A features reused."""
import numpy as np
from research_log.T065A.core import features,FEATURES,base_choose,summarize,RHO
LAMBDA=1e-3


def class_info(labels):
    labels=np.asarray(labels,dtype=np.float64);safe=int(labels.sum());unsafe=len(labels)-safe
    return dict(safe=safe,unsafe=unsafe,n=len(labels),safe_weight=len(labels)/(2*safe) if safe else None,unsafe_weight=len(labels)/(2*unsafe) if unsafe else None)


def sigmoid(logits):
    return 1/(1+np.exp(-np.clip(logits,-30,30)))


def fit(x,labels,normalization):
    x=np.asarray(x,dtype=np.float64);y=np.asarray(labels,dtype=np.float64);counts=class_info(y)
    if not counts['safe'] or not counts['unsafe']:return None,[]
    z=(x-np.asarray(normalization['mean']))/np.asarray(normalization['scale']);a=np.column_stack([np.ones(len(x)),z])
    weights=np.where(y==1,counts['safe_weight'],counts['unsafe_weight']);penalty=np.diag([0.]+[LAMBDA]*11)
    beta=np.zeros(12,dtype=np.float64);trace=[]
    for iteration in range(50):
        logits=np.clip(a@beta,-30,30);p=sigmoid(logits)
        loss=float(np.sum(weights*(np.logaddexp(0,logits)-y*logits))+.5*LAMBDA*(beta[1:]@beta[1:]))
        g=a.T@(weights*(p-y))+penalty@beta
        H=a.T@((weights*p*(1-p))[:,None]*a)+penalty
        step=np.linalg.solve(H,g);before=beta.copy();beta=beta-step;norm=float(np.max(np.abs(step)))
        trace.append(dict(iteration=iteration,loss=loss,beta_before=before.tolist(),newton_step=step.tolist(),step_inf=norm,beta_after=beta.tolist()))
        if norm<1e-12:break
    model=dict(mean=normalization['mean'],scale=normalization['scale'],intercept=float(beta[0]),coefficients=beta[1:].tolist(),class_info=counts,
        coefficient_l2=LAMBDA,threshold=.5,max_iterations=50,logit_clip=[-30,30],initialization='zero',stop='Newton step infinity norm <1e-12; else50iterations',
        objective='sum class-weighted logistic NLL + lambda * coefficient norm squared /2; unregularized intercept')
    return model,trace


def predict(x,model):
    z=(np.asarray(x,dtype=np.float64)-np.asarray(model['mean']))/np.asarray(model['scale'])
    return sigmoid(z@np.asarray(model['coefficients'])+model['intercept'])


def choose(probabilities,base_step):
    return max([0]+[k for k in range(base_step+1) if probabilities[k]>=.5])


def confusion(labels,probabilities):
    y=np.asarray(labels,dtype=bool);p=np.asarray(probabilities)>=.5
    return dict(safe_pred_safe=int(np.sum(y&p)),safe_pred_unsafe=int(np.sum(y&~p)),unsafe_pred_safe=int(np.sum(~y&p)),unsafe_pred_unsafe=int(np.sum(~y&~p)))
