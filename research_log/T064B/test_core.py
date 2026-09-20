import math
import torch
from research_log.T064B.core import perturb,sensitivity,choose,candidates,calibrate


def test_fixed_seed_shape_and_clipped_antithetic_statistic():
    x=torch.tensor([0.,1.,.5,.2]).reshape(1,1,2,2)
    p,m=perturb(x);torch.manual_seed(99);p2,m2=perturb(x)
    assert torch.equal(p,p2) and torch.equal(m,m2)
    assert p.min()>=0 and p.max()<=1 and m.min()>=0 and m.max()<=1
    assert sensitivity(p,m,p,m)==1
    assert sensitivity(p*2,m*2,p,m)==2
    assert sensitivity(p*0,m*0,p,m)==0


def test_largest_prefix_only_and_identity_fallback():
    assert choose([1,3,2,.5],2,2)==2
    assert choose([1,3,2,.5],2,.9)==0
    assert choose([1,math.nan,2,.5],2,1)==0
    assert candidates([dict(sensitivity=[1,3,2,.5],base_step=2)])==[1.,2.,3.]


def test_all_five_gates_and_smaller_threshold_tie():
    rows=[dict(sensitivity=[1,2,3,4],base_step=3)]*100
    quality=[dict(psnr=[0,14,14,1],ssim=[.5]*4,t026_psnr=10,t036_psnr=10,t036_ssim=.5)]*100
    table,best,passed=calibrate(rows,quality)
    assert passed and best['tau']==2 and best['mean_psnr']==14
    assert not table[-1]['eligible']
    weak=[dict(psnr=[10,11,11,10],ssim=[.5]*4,t026_psnr=10,t036_psnr=10,t036_ssim=.5)]*100
    assert calibrate(rows,weak)[1:]==(None,False)
