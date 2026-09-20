import numpy as np
import torch
from research_log.T065A.core import features,fit,predict,choose,LAMBDA


def test_features_identity_scaling_progress_and_raw_state():
    x=torch.tensor([[[[0.,.01],[.99,1.]]]],dtype=torch.float64).repeat(1,3,1,1)
    images=[x,x*.5];states=[torch.zeros(1,3,2,2),torch.ones(1,3,2,2)]
    f=features(x,images,states,[4.,2.])
    expected0=[0,0,0,.5,np.std([0,.01,.99,1]),.5,.5,1,0,0,0]
    expected1=[1/27,1,1,.25,np.std([0,.01,.99,1])*.5,0,.5,.5,.25,np.sqrt(12),1]
    np.testing.assert_allclose(f,[expected0,expected1],rtol=0,atol=1e-14)
    z=features(torch.zeros_like(x),[torch.zeros_like(x)],[states[0]],[1.])
    assert np.isfinite(z).all() and z[0,7]==0 and z[0,6]==1


def test_ridge_fixed_penalty_population_scale_and_free_intercept():
    x=np.ones((28,11))*2;x[:,0]=np.arange(28);y=3*x[:,0]+7
    model=fit(x,y);std=x[:,0].std(ddof=0)
    np.testing.assert_allclose(model['coefficients'][0],3*std*28/(28+LAMBDA),rtol=0,atol=1e-12)
    assert model['scale'][1:]==[1e-8]*10
    np.testing.assert_allclose(model['intercept'],y.mean(),rtol=0,atol=1e-12)
    np.testing.assert_allclose(predict(x,model).mean(),y.mean(),rtol=0,atol=1e-12)
    np.testing.assert_allclose(model['coefficients'][1:],0,rtol=0,atol=1e-12)


def test_selector_prefix_and_earliest_tie():
    assert choose([0,4,4,10],2)==1
    assert choose([0,4,4,10],0)==0
    assert choose([-1,-2,-3],2)==0


def test_independent_features_and_fit_agree():
    from research_log.T065A.verify import independent_features,independent_fit
    rng=np.random.default_rng(65065);x=torch.from_numpy(rng.random((1,3,8,9)))
    images=[x*.7,x*.9,x];states=torch.from_numpy(rng.normal(size=(3,1,3,2,2)));totals=[3.,2.,2.1]
    np.testing.assert_allclose(features(x,images,states,totals),independent_features(x,images,states,totals),rtol=0,atol=2e-14)
    a=rng.normal(size=(80,11));b=rng.normal(size=80);m=fit(a,b);n=independent_fit(a,b)
    for key in ['mean','scale','intercept','coefficients']:np.testing.assert_allclose(m[key],n[key],rtol=0,atol=1e-12)
