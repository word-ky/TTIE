"""Fixed full-image RGB SSIM and paired source-image cluster bootstrap."""
import numpy as np
from scipy.ndimage import gaussian_filter

METRIC=dict(data_range=1.,window=11,sigma=1.5,K1=.01,K2=.03,
            covariance='population',channels='RGB mean',crop=0,
            padding='reflect (half-sample symmetric)',dtype='float64')
BOOTSTRAP=dict(resamples=10000,seed=7,rng='numpy.default_rng PCG64',
               unit='source image',interval='two-sided percentile 95%',quantile_method='linear')

def rgb_ssim(image,reference):
    """HWC RGB arrays at original frozen resolution, no normalization/crop."""
    x=np.asarray(image,dtype=np.float64);y=np.asarray(reference,dtype=np.float64)
    filt=lambda v:gaussian_filter(v,sigma=(1.5,1.5,0),truncate=3.5,mode='reflect')
    ux=filt(x);uy=filt(y)
    vx=filt(x*x)-ux*ux;vy=filt(y*y)-uy*uy;vxy=filt(x*y)-ux*uy
    score=((2*ux*uy+.01**2)*(2*vxy+.03**2))/((ux*ux+uy*uy+.01**2)*(vx+vy+.03**2))
    return float(score.mean())

def cluster_bootstrap(image_ids,deltas):
    ids=np.asarray(image_ids);d=np.asarray(deltas,dtype=np.float64);clusters=np.unique(ids)
    sums=np.asarray([d[ids==i].sum() for i in clusters]);counts=np.asarray([sum(ids==i) for i in clusters])
    draws=np.random.default_rng(7).integers(0,len(clusters),size=(10000,len(clusters)))
    # Sample each source image with all its rows, including multiplicity.
    means=sums[draws].sum(1)/counts[draws].sum(1)
    low,high=np.quantile(means,[.025,.975],method='linear').tolist()
    return dict(**BOOTSTRAP,clusters=clusters.tolist(),rows=len(d),mean=float(d.mean()),median=float(np.median(d)),
                ci95=[low,high],positive=low>0),means,draws
