import numpy as np
import torch
from research_log.T066A.core import features,normalization,train_fold,fit,predict,choose
from research_log.T065A.core import features as snapshot


def test_temporal_features_startup_curvature_and_acceleration():
    x=torch.ones(1,3,2,2,dtype=torch.float64)*.1;images=[x,x+.1,x+.3]
    states=[torch.zeros(1,3,2,2),torch.ones(1,3,2,2),torch.ones(1,3,2,2)*3];totals=[4.,3.,2.5]
    f=features(x,images,states,totals)
    np.testing.assert_array_equal(f[:,:11],snapshot(x,images,states,totals))
    expected=[[0]*8,[1/1.5,0,.1,0,np.sqrt(12),0,.1,0],[1,.5/1.5,.2,.1,2*np.sqrt(12),np.sqrt(12),.2,0]]
    np.testing.assert_allclose(f[:,11:],expected,rtol=0,atol=1e-14)


def test_old_normalization_preserved_and_new_features_training_only():
    x=np.ones((6,19));x[4:,11:]=1000;old=dict(mean=list(range(11)),scale=[2.]*11)
    n=normalization(x[:4],old);assert n['mean'][:11]==old['mean'] and n['scale'][:11]==old['scale']
    assert n['mean'][11:]==[1.]*8 and n['scale'][11:]==[1e-8]*8
    y=[0,1,0,1,0,1];m,t=train_fold(x,y,[0,0,1,1,2,2],2,old)
    assert m['mean'][11:]==[1.]*8 and m['class_info']['n']==4


def test_19d_solver_and_rollback():
    rng=np.random.default_rng(66066);x=rng.normal(size=(120,19));y=(x[:,11]+rng.normal(size=120)>.4).astype(float)
    old=dict(mean=[0.]*11,scale=[1.]*11);m,t=fit(x,y,normalization(x,old))
    assert len(m['coefficients'])==19 and len(t)<=50 and np.isfinite(predict(x,m)).all()
    assert choose([.8,.2,.6,.1],3)==2 and choose([.1,.2],1)==0


def test_independent_temporal_features_and_19d_fit():
    from research_log.T066A.verify import independent_features,independent_fit,independent_normalization
    rng=np.random.default_rng(66067);x=torch.from_numpy(rng.random((1,3,5,6)));images=[x,x*.8,x*.5];states=torch.from_numpy(rng.random((3,1,3,2,2)))
    np.testing.assert_allclose(features(x,images,states,[4.,3.,2.]),independent_features(x,images,states,[4.,3.,2.]),rtol=0,atol=1e-14)
    a=rng.normal(size=(140,19));y=(a[:,12]+rng.normal(size=140)>.4).astype(float);old=dict(mean=[0.]*11,scale=[1.]*11)
    norm=normalization(a,old);m,t=fit(a,y,norm);n=independent_fit(a,y,independent_normalization(a,old))
    np.testing.assert_allclose(m['coefficients'],n['coefficients'],rtol=0,atol=1e-7)
