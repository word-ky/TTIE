import numpy as np
from research_log.T065B.core import fit,predict,choose,class_info,confusion,sigmoid


def test_balanced_weights_and_empty_support():
    info=class_info([1,1,1,0]);assert info['safe_weight']*3==info['unsafe_weight']==2
    assert fit(np.zeros((2,11)),[1,1],dict(mean=[0.]*11,scale=[1.]*11))==(None,[])


def test_weighted_intercept_is_balanced_and_fixed_newton_rule():
    x=np.zeros((4,11));normal=dict(mean=[0.]*11,scale=[1.]*11);m,t=fit(x,[1,1,1,0],normal)
    np.testing.assert_array_equal(predict(x,m),.5)
    assert len(t)==1 and t[0]['step_inf']==0 and m['intercept']==0
    assert m['mean']==normal['mean'] and m['scale']==normal['scale']


def test_logistic_signal_stationarity_and_determinism():
    rng=np.random.default_rng(65066);x=rng.normal(size=(100,11));y=(x[:,0]+.2*x[:,1]>0).astype(int)
    norm=dict(mean=x.mean(0).tolist(),scale=x.std(0).tolist());m,t=fit(x,y,norm);m2,t2=fit(x,y,norm)
    assert m==m2 and t==t2 and len(t)<=50
    assert (len(t)==50 or t[-1]['step_inf']<1e-12) and np.mean((predict(x,m)>=.5)==y)>.95
    for row in t:np.testing.assert_allclose(np.array(row['beta_before'])-row['newton_step'],row['beta_after'],rtol=0,atol=0)
    assert np.isfinite(sigmoid([-1000,1000])).all()


def test_rollback_keeps_base_latest_safe_or_identity():
    assert choose([.9,.1,.6],2)==2
    assert choose([.9,.5,.2,.99],2)==1
    assert choose([.1,.2,.3],2)==0
    assert confusion([1,1,0,0],[.7,.2,.8,.1])==dict(safe_pred_safe=1,safe_pred_unsafe=1,unsafe_pred_safe=1,unsafe_pred_unsafe=1)


def test_independent_logistic_fit_and_trace_equations():
    from research_log.T065B.verify import independent_fit,check_newton
    rng=np.random.default_rng(77);x=rng.normal(size=(120,11));y=(x[:,0]+rng.normal(size=120)>.5).astype(float)
    norm=dict(mean=x.mean(0).tolist(),scale=x.std(0).tolist());m,t=fit(x,y,norm);other=independent_fit(x,y,norm)
    np.testing.assert_allclose(m['coefficients'],other['coefficients'],rtol=0,atol=1e-8)
    assert abs(m['intercept']-other['intercept'])<1e-8
    assert check_newton(t,x,y,norm,m)<1e-8
