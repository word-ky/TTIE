import numpy as np
from skimage.metrics import structural_similarity
from ttie.ssim_transfer import rgb_ssim,cluster_bootstrap

def test_identity_perturbation_and_full_skimage_map():
    rng=np.random.default_rng(13);x=rng.uniform(.1,.8,(27,31,3));y=x+.07
    assert rgb_ssim(x,x)==1.
    assert rgb_ssim(x,y)<1.
    _,full=structural_similarity(x,y,channel_axis=-1,data_range=1.,gaussian_weights=True,
        sigma=1.5,win_size=11,K1=.01,K2=.03,use_sample_covariance=False,full=True)
    # skimage's scalar discards the border. The task explicitly says no crop,
    # so cross-check the mean over its full returned per-channel SSIM map.
    assert abs(rgb_ssim(x,y)-float(full.mean()))<1e-12
    edge=y.copy();edge[:5]*=.5
    _,full=structural_similarity(x,edge,channel_axis=-1,data_range=1.,gaussian_weights=True,
        sigma=1.5,win_size=11,K1=.01,K2=.03,use_sample_covariance=False,full=True)
    assert abs(rgb_ssim(x,edge)-float(full.mean()))<1e-12

def test_bootstrap_keeps_all_rows_per_image_and_multiplicity():
    ids=np.array([10,10,20,30,30,30]);d=np.array([.2,.4,-.1,.1,.3,.5])
    report,means,draws=cluster_bootstrap(ids,d)
    expected=np.random.default_rng(7).integers(0,3,size=(10000,3));assert np.array_equal(draws,expected)
    for i in (0,1,17,9999):
        actual=np.concatenate([d[ids==[10,20,30][j]] for j in expected[i]])
        assert abs(means[i]-actual.mean())<1e-15
    assert report['ci95']==np.quantile(means,[.025,.975],method='linear').tolist()
